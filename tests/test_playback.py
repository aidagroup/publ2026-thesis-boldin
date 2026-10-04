"""Playing back an IPPO checkpoint in an env with other agent uids (`callosum.training._playback`).

Pure torch on CPU: the env is replaced by structured observations shaped like the real
wrist-camera FaceTurn env's (`so101_pg_wristcam-*` agent keys, same fields as the training env).
"""

import pytest

torch = pytest.importorskip("torch")  # CI installs only the dev extra; run with `make dev`

from callosum.configs.ippo import IPPOConfig
from callosum.training._agent_obs import AGENT_UIDS, AgentObsBuilder, obs_layout
from callosum.training._checkpoint import save_checkpoint
from callosum.training._playback import (
    CheckpointPolicy,
    EpisodeSummary,
    rename_agent_keys,
    run_name_of,
    state_of,
)
from callosum.training._ppo_core import ActorCritic

CAM_UIDS = ("so101_pg_wristcam-0", "so101_pg_wristcam-1")
N = 1


def structured_obs(uids: tuple[str, str], seed: int = 0) -> dict:
    """A FaceTurn-like structured state observation (flat width 57), random content."""
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(N, *shape, generator=gen)

    return {
        "agent": {
            uids[0]: {"qpos": rnd(7), "qvel": rnd(7)},
            uids[1]: {"qpos": rnd(7), "qvel": rnd(7)},
        },
        "extra": {
            "agent_a_tcp_pose": rnd(7),
            "agent_b_tcp_pose": rnd(7),
            "cube_pose": rnd(7),
            "face_angle": rnd(),
            "face_pose": rnd(7),
        },
    }


def flat(structured: dict) -> torch.Tensor:
    parts: list[torch.Tensor] = []

    def walk(node: dict) -> None:
        for value in node.values():
            walk(value) if isinstance(value, dict) else parts.append(value.reshape(N, -1))

    walk(structured)
    return torch.cat(parts, dim=-1)


def make_checkpoint(tmp_path, partner_obs: str = "full", obs_dims=None):
    """Checkpoint of two random actors trained on the plain `so101_pg` uids."""
    cfg = IPPOConfig(partner_obs=partner_obs, num_envs=1, num_steps=8, num_minibatches=2)
    reference = AgentObsBuilder(obs_layout(structured_obs(AGENT_UIDS)), partner_obs, AGENT_UIDS)
    dims = obs_dims or reference.obs_dims
    torch.manual_seed(0)
    agents = [ActorCritic(d, 6) for d in dims]
    path = tmp_path / "myrun" / "best.pt"
    path.parent.mkdir()
    save_checkpoint(path, agents, cfg, list(dims), list(AGENT_UIDS), 3, 300)
    return path, agents, reference


def bind(policy: CheckpointPolicy, structured: dict) -> None:
    bounds = [(-torch.ones(6), torch.ones(6))] * 2
    policy.bind(structured, flat(structured), CAM_UIDS, bounds)


def test_rename_agent_keys_keeps_order_and_content() -> None:
    cam = structured_obs(CAM_UIDS)
    plain = rename_agent_keys(cam, CAM_UIDS, AGENT_UIDS)
    assert list(plain) == ["agent", "extra"]
    assert list(plain["agent"]) == list(AGENT_UIDS)
    assert plain["agent"][AGENT_UIDS[1]]["qpos"] is cam["agent"][CAM_UIDS[1]]["qpos"]
    assert obs_layout(plain) == obs_layout(structured_obs(AGENT_UIDS))


def test_rename_agent_keys_rejects_unknown_uids() -> None:
    with pytest.raises(KeyError):
        rename_agent_keys(structured_obs(AGENT_UIDS), CAM_UIDS, AGENT_UIDS)


def test_run_name_is_the_directory_name(tmp_path) -> None:
    assert run_name_of(tmp_path / "faceturn_s1" / "best.pt") == "faceturn_s1"


def test_state_of_accepts_flat_and_state_plus_rgb_obs() -> None:
    x = torch.arange(6.0).reshape(1, 6)
    assert torch.equal(state_of(x), x)
    assert torch.equal(state_of({"state": x, "sensor_data": {}}), x)


def test_policy_rebuilds_the_actors_from_the_checkpoint(tmp_path) -> None:
    path, agents, _ = make_checkpoint(tmp_path, partner_obs="none")
    policy = CheckpointPolicy(path)
    assert policy.cfg.partner_obs == "none" and policy.run_name == "myrun"
    assert policy.obs_dims == [36, 36] and policy.action_dims == [6, 6]
    for old, new in zip(agents, policy.agents, strict=True):
        assert not new.training
        for a, b in zip(old.parameters(), new.parameters(), strict=True):
            assert torch.equal(a, b)


@pytest.mark.parametrize("partner_obs", ["full", "none"])
def test_actions_match_the_training_input_pipeline(tmp_path, partner_obs) -> None:
    """In the camera env (other uids) the policy sees exactly what the trainer would feed it."""
    path, agents, reference = make_checkpoint(tmp_path, partner_obs=partner_obs)
    policy = CheckpointPolicy(path)
    cam = structured_obs(CAM_UIDS, seed=4)
    bind(policy, cam)
    assert policy.builder.fields == reference.fields
    assert policy.builder.columns == reference.columns

    state = flat(cam)
    actions = policy.act(state)
    assert list(actions) == list(CAM_UIDS)
    for uid, agent, x in zip(CAM_UIDS, agents, reference(state), strict=True):
        expected = torch.clamp(agent.actor_mean(x), -1, 1)
        assert torch.allclose(actions[uid], expected)


def test_deterministic_by_default_and_stochastic_on_request(tmp_path) -> None:
    path, _, _ = make_checkpoint(tmp_path)
    cam = structured_obs(CAM_UIDS)
    det = CheckpointPolicy(path)
    bind(det, cam)
    first, second = det.act(flat(cam)), det.act(flat(cam))
    assert all(torch.equal(first[u], second[u]) for u in CAM_UIDS)
    sto = CheckpointPolicy(path, deterministic=False)
    bind(sto, cam)
    assert not torch.equal(sto.act(flat(cam))[CAM_UIDS[0]], sto.act(flat(cam))[CAM_UIDS[0]])


def test_actions_are_clamped_to_the_action_bounds(tmp_path) -> None:
    path, _, _ = make_checkpoint(tmp_path)
    policy = CheckpointPolicy(path)
    cam = structured_obs(CAM_UIDS)
    tight = [(-torch.full((6,), 1e-3), torch.full((6,), 1e-3))] * 2
    policy.bind(cam, flat(cam), CAM_UIDS, tight)
    assert all(a.abs().max() <= 1e-3 for a in policy.act(flat(cam)).values())


def test_bind_rejects_an_env_whose_state_differs_from_training(tmp_path) -> None:
    # Checkpoint trained on a state without face_angle / face_pose (35 inputs) vs. a FaceTurn env.
    path, _, _ = make_checkpoint(tmp_path, obs_dims=[35, 35])
    policy = CheckpointPolicy(path)
    with pytest.raises(ValueError, match="trained with"):
        bind(policy, structured_obs(CAM_UIDS))


def test_bind_rejects_a_flat_state_that_does_not_match_the_layout(tmp_path) -> None:
    path, _, _ = make_checkpoint(tmp_path)
    policy = CheckpointPolicy(path)
    cam = structured_obs(CAM_UIDS)
    shuffled = flat(cam)[:, torch.randperm(57, generator=torch.Generator().manual_seed(1))]
    with pytest.raises(ValueError):
        policy.bind(cam, shuffled, CAM_UIDS, [(-torch.ones(6), torch.ones(6))] * 2)


def test_act_before_bind_raises(tmp_path) -> None:
    path, _, _ = make_checkpoint(tmp_path)
    with pytest.raises(RuntimeError):
        CheckpointPolicy(path).act(torch.zeros(1, 57))


def test_episode_summary_tracks_return_success_and_face_angle() -> None:
    summary = EpisodeSummary()
    summary.observe({"success": torch.tensor([False]), "face_angle": torch.tensor([0.0])})
    summary.update(0.5, {"success": torch.tensor([False]), "face_angle": torch.tensor([1.0])})
    summary.update(0.25, {"success": torch.tensor([True]), "face_angle": torch.tensor([1.5708])})
    summary.update(0.25, {"success": torch.tensor([False]), "face_angle": torch.tensor([1.2])})
    assert summary.steps == 3 and summary.total_return == pytest.approx(1.0)
    assert summary.success_once and not summary.success
    assert summary.max_face_angle_deg == pytest.approx(90.0, abs=0.01)
    assert summary.face_angle_deg == pytest.approx(68.75, abs=0.01)
    assert "success False" in summary.status() and "68.8 deg" in summary.status()
    assert "once: True" in summary.text() and "3 steps" in summary.text()
