"""Behaviour-cloning pretraining of the two IPPO actor-critics from scripted-expert demonstrations.

Run as `python -m callosum.training.bc --demos runs/demos/X.pt --partner-obs full --exp-name NAME`
(flags: `--help`, defaults: `callosum.configs.bc.BCConfig`). No simulator is needed (the demo file
carries the observation layout), so it runs on CPU in the macOS project venv as well as on the GPU
server.

Why: IPPO from scratch never finds a grasp on FaceTurn-v0 (diagnostics of a 10M-step run: no arm
ever grasps anything). The scripted expert (`callosum.experts.face_turn_expert`) does, so the
policies start from its demonstrations (`scripts/collect_demos.py`) and PPO fine-tunes from there
(`ippo --checkpoint <run>/bc.pt`, optionally with `--critic-warmup-iters` and the auxiliary demo
loss `--demos/--bc-coef`).

What is trained, per agent (holder `so101_pg-0`, rotator `so101_pg-1`), on the same per-agent
inputs IPPO builds (`_agent_obs.AgentObsBuilder` with the demo file's layout and `--partner-obs`;
raw, unnormalised observations, as in IPPO):

* the actor mean: MSE to the demonstrated 6-D actions;
* the critic: MSE to the discounted Monte-Carlo return of the demo rewards (`gamma`, `reward_scale`
  of the IPPO run; success is a true terminal, value 0 after it, as in the trainer). The targets
  are divided by their standard deviation while training and the factor is folded back into the
  critic's last layer afterwards, so the saved critic is on the real return scale;
* `actor_logstd` is not learned: it is set to `--actor-logstd`.

Validation: whole episodes are held out (`val_fraction`); the printed numbers are the actor MSE
(overall and for the gripper dimension, with the share of correct gripper signs) and the critic
RMSE in return units, on the training and the validation split. Outputs in
`<runs_dir>/<exp_name>/`: `bc.pt` (a format-1 checkpoint, see `_checkpoint`), `config.json` and
`bc_log.json` (the loss curve).
"""

import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from callosum.configs.bc import BCConfig, parse_args
from callosum.training._agent_obs import AgentObsBuilder
from callosum.training._checkpoint import AGENT_NAMES, save_weights_checkpoint
from callosum.training._demos import (
    agent_demo_tensors,
    discounted_returns,
    layout_from_meta,
    load_demos,
    success_mask,
)
from callosum.training._ppo_core import ActorCritic

GRIPPER_DIM = 5  # last entry of the 6-D action: the absolute gripper target


@dataclasses.dataclass
class BCData:
    """The demonstrations as per-agent training tensors.

    Attributes:
        obs: per agent, `(S, obs_dim)` policy inputs.
        actions: per agent, `(S, 6)` demonstrated actions.
        returns: `(S,)` Monte-Carlo returns (the critic's targets, same for both agents).
        episode_id: `(S,)` episode index per transition.
        builder: the per-agent input builder (for the checkpoint's field names).
    """

    obs: list[torch.Tensor]
    actions: list[torch.Tensor]
    returns: torch.Tensor
    episode_id: torch.Tensor
    builder: AgentObsBuilder

    @property
    def size(self) -> int:
        return int(self.returns.shape[0])

    def subset(self, mask: torch.Tensor) -> "BCData":
        """The transitions where `mask` (a `(S,)` bool tensor) is True."""
        return BCData(
            [o[mask] for o in self.obs],
            [a[mask] for a in self.actions],
            self.returns[mask],
            self.episode_id[mask],
            self.builder,
        )


def prepare_data(demos: dict[str, Any], cfg: BCConfig) -> BCData:
    """Per-agent inputs, actions and critic targets from a loaded demo file.

    Keeps only the successful episodes when `cfg.only_success`. The returns are computed over
    the kept episodes (so they are unaffected by what was dropped).
    """
    meta = demos["meta"]
    uids = tuple(meta["agent_uids"])
    builder = AgentObsBuilder(layout_from_meta(meta), cfg.partner_obs, uids)
    obs, actions = agent_demo_tensors(demos, builder, cfg.only_success)
    keep = success_mask(demos, cfg.only_success)
    episode_id = demos["episode_id"][keep]
    returns = discounted_returns(
        demos["rewards"][keep].numpy(), episode_id.numpy(), cfg.gamma, cfg.reward_scale
    )
    return BCData(obs, actions, torch.as_tensor(returns, dtype=torch.float32), episode_id, builder)


def split_episodes(data: BCData, val_fraction: float, seed: int) -> tuple[BCData, BCData | None]:
    """Hold out whole episodes: `(train, val)`, `val` is `None` without validation episodes.

    About `val_fraction` of the episodes (at least one if the fraction is positive and there are
    two or more episodes) go to validation, chosen by a seeded permutation.
    """
    episodes = torch.unique(data.episode_id)
    num_val = round(len(episodes) * val_fraction)
    if val_fraction > 0 and len(episodes) >= 2:
        num_val = max(1, min(num_val, len(episodes) - 1))
    else:
        num_val = 0
    if num_val == 0:
        return data, None
    generator = torch.Generator().manual_seed(seed)
    chosen = episodes[torch.randperm(len(episodes), generator=generator)[:num_val]]
    is_val = torch.isin(data.episode_id, chosen)
    return data.subset(~is_val), data.subset(is_val)


@torch.no_grad()
def evaluate_losses(
    agents: list[ActorCritic], data: BCData, value_scale: float
) -> dict[str, float]:
    """Actor MSE (overall and gripper), gripper-sign accuracy and critic RMSE per agent.

    The critic is evaluated in return units (`value_scale` is the factor the training targets
    were divided by while the critic's last layer is still unscaled).
    """
    out: dict[str, float] = {}
    for name, agent, obs, actions in zip(AGENT_NAMES, agents, data.obs, data.actions):
        mean = agent.actor_mean(obs)
        err = (mean - actions) ** 2
        out[f"{name}/actor_mse"] = err.mean().item()
        out[f"{name}/gripper_mse"] = err[:, GRIPPER_DIM].mean().item()
        agree = torch.sign(mean[:, GRIPPER_DIM]) == torch.sign(actions[:, GRIPPER_DIM])
        out[f"{name}/gripper_sign_acc"] = agree.float().mean().item()
        value = agent.get_value(obs).squeeze(-1) * value_scale
        out[f"{name}/critic_rmse"] = ((value - data.returns) ** 2).mean().sqrt().item()
    return out


def fold_value_scale(agent: ActorCritic, scale: float) -> None:
    """Multiply the critic's output by `scale` by scaling its last linear layer in place."""
    last = agent.critic[-1]
    with torch.no_grad():
        last.weight.mul_(scale)
        last.bias.mul_(scale)


def train_bc(
    agents: list[ActorCritic],
    train: BCData,
    val: BCData | None,
    cfg: BCConfig,
    log=print,
) -> dict[str, Any]:
    """Fit the agents on `train` (actor MSE, critic MSE to scaled returns), in place.

    Returns the statistics: `value_scale`, the final train / val losses and the per-logged-epoch
    history. The critics' last layers are scaled back to return units before returning.
    """
    device = train.returns.device
    value_scale = max(1.0, float(train.returns.std()) if train.size > 1 else 1.0)
    actor_opts = [torch.optim.Adam(a.actor_mean.parameters(), lr=cfg.learning_rate) for a in agents]
    critic_opts = [torch.optim.Adam(a.critic.parameters(), lr=cfg.learning_rate) for a in agents]
    targets = train.returns / value_scale
    history: list[dict[str, Any]] = []
    for epoch in range(1, cfg.epochs + 1):
        lr = cfg.learning_rate * (1.0 - (epoch - 1.0) / cfg.epochs if cfg.anneal_lr else 1.0)
        for opt in (*actor_opts, *critic_opts):
            opt.param_groups[0]["lr"] = lr
        order = torch.randperm(train.size, device=device)
        for start in range(0, train.size, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            for i, agent in enumerate(agents):
                obs = train.obs[i][idx]
                actor_loss = ((agent.actor_mean(obs) - train.actions[i][idx]) ** 2).mean()
                actor_opts[i].zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor_mean.parameters(), cfg.grad_clip)
                actor_opts[i].step()
                critic_loss = ((agent.get_value(obs).squeeze(-1) - targets[idx]) ** 2).mean()
                critic_opts[i].zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), cfg.grad_clip)
                critic_opts[i].step()
        if epoch % cfg.log_every == 0 or epoch in (1, cfg.epochs):
            row = {"epoch": epoch, "lr": lr}
            row.update(
                {f"train/{k}": v for k, v in evaluate_losses(agents, train, value_scale).items()}
            )
            if val is not None:
                row.update(
                    {f"val/{k}": v for k, v in evaluate_losses(agents, val, value_scale).items()}
                )
            history.append(row)
            log(_format_row(row))
    for agent in agents:
        fold_value_scale(agent, value_scale)
    return {"value_scale": value_scale, "history": history, "final": history[-1]}


def _format_row(row: dict[str, Any]) -> str:
    """One stdout line: the epoch, then the actor MSE and critic RMSE of each agent per split."""
    parts = [f"epoch {row['epoch']:>4}"]
    for split in ("train", "val"):
        if f"{split}/{AGENT_NAMES[0]}/actor_mse" not in row:
            continue
        cells = []
        for name in AGENT_NAMES:
            short = name[-1]
            cells.append(
                f"{short}: act {row[f'{split}/{name}/actor_mse']:.4f}"
                f" grip {row[f'{split}/{name}/gripper_sign_acc']:.3f}"
                f" crit {row[f'{split}/{name}/critic_rmse']:.2f}"
            )
        parts.append(f"{split} [" + " | ".join(cells) + "]")
    return " | ".join(parts)


def run(cfg: BCConfig) -> Path:
    """Load the demos, train both agents, write the checkpoint; returns the run directory."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(
        "cuda"
        if cfg.device == "cuda" or (cfg.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    demos = load_demos(cfg.demos)
    meta = demos["meta"]
    data = prepare_data(demos, cfg)
    train, val = split_episodes(data, cfg.val_fraction, cfg.seed)
    run_name = cfg.exp_name or f"bc__{cfg.seed}__{int(time.time())}"
    run_dir = Path(cfg.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    episodes = int(torch.unique(data.episode_id).numel())
    print(
        f"demos {cfg.demos}: {demos['meta']['num_episodes']} episodes in file, using {episodes} "
        f"({data.size} transitions; train {train.size}, val {0 if val is None else val.size}) | "
        f"partner_obs {cfg.partner_obs} | device {device} | collected at {meta.get('git_commit')} | "
        f"episode limit {meta.get('max_episode_steps')}"
    )
    for i, name in enumerate(AGENT_NAMES):
        print(
            f"{name}: obs {data.builder.obs_dims[i]} -> act 6 | "
            + ", ".join(data.builder.fields[i])
        )
    print(
        f"returns: mean {data.returns.mean():.1f}, std {data.returns.std():.1f}, "
        f"max {data.returns.max():.1f} (gamma {cfg.gamma})"
    )

    train = BCData(
        [o.to(device) for o in train.obs],
        [a.to(device) for a in train.actions],
        train.returns.to(device),
        train.episode_id.to(device),
        train.builder,
    )
    if val is not None:
        val = BCData(
            [o.to(device) for o in val.obs],
            [a.to(device) for a in val.actions],
            val.returns.to(device),
            val.episode_id.to(device),
            val.builder,
        )
    agents = [ActorCritic(d, 6).to(device) for d in data.builder.obs_dims]
    for agent in agents:
        agent.actor_logstd.data.fill_(cfg.actor_logstd)

    start = time.time()
    stats = train_bc(agents, train, val, cfg)
    print(f"trained {cfg.epochs} epochs in {time.time() - start:.0f}s")
    # train_bc scaled the critics back to return units: report the final losses on the real scale.
    final = {
        f"{split}/{k}": v
        for split, d in (("train", train), ("val", val))
        if d is not None
        for k, v in evaluate_losses(agents, d, 1.0).items()
    }
    save_weights_checkpoint(
        run_dir / "bc.pt",
        [a.cpu() for a in agents],
        dataclasses.asdict(cfg),
        data.builder.obs_dims,
        list(meta["agent_uids"]),
        data.builder.fields,
        extra={
            "final": final,
            "value_scale": stats["value_scale"],
            "demos": cfg.demos,
            # The episode length of the demos: the IPPO warm start warns if its env differs.
            "demo_max_episode_steps": meta.get("max_episode_steps"),
        },
    )
    (run_dir / "bc_log.json").write_text(json.dumps(stats["history"], indent=2))
    print(f"saved {run_dir / 'bc.pt'}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: `python -m callosum.training.bc [flags]`."""
    sys.stdout.reconfigure(line_buffering=True)
    run(parse_args(argv))


if __name__ == "__main__":
    main()
