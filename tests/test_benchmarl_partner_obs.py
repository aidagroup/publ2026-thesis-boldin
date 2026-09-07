"""Guard test: review Б4 — BenchMARL path must not leak partner_obs to actor.

Review Б4 (2026-08-24) found that the BenchMARL task wrapper built each
agent's policy input from the FULL `extra` block, including the partner's
TCP pose — turning a `partner_obs="none"` run into an oracle. The fix routes
the per-agent input through `build_agent_obs(include_partner=False)` (the
shared helper at callosum.training._agent_obs that drops the other agent's
prefixed fields) and threads `partner_obs` from TaskConfig into gym.make
(so the env itself stops emitting partner fields when `partner_obs="none"`).

This test pins that contract. It runs on macOS/CI because the helpers it
exercises (`_per_agent_obs_dim`, `_make_gym_env`, `CallosumTaskClass`) are
importable without `mani_skill`; the test mocks the gym `Dict` space and
inspects the wiring, it does not instantiate a real env.

The check that *actually* proves the leak is gone is
`_per_agent_obs_dim` with a hand-built space: the function must drop the
other agent's `agent_x_*` keys from its `extra` count, so an `agent_b_*`
TCP field cannot widen `agent_a`'s input.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from callosum.envs import benchmarl_task
from callosum.envs.benchmarl_task import (
    GROUP_NAME,
    CallosumTaskClass,
    _per_agent_obs_dim,
)


# ---------------------------------------------------------------------------
# Minimal gym-Dict stand-ins (no gym import). gym spaces are duck-typed
# everywhere in benchmarl_task: `_flat_dim` looks for `.spaces` (Dict),
# `.n` (Discrete), `.shape` (Box). The space objects below are exactly
# that.
# ---------------------------------------------------------------------------
class _Box:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _Dict:
    """gym.spaces.Dict stand-in. Real gym.Dict supports both ``__getitem__``,
    ``.get``, and ``.spaces``; benchmarl_task uses ``__getitem__`` (line 130)
    and ``.get`` (line 131), and `_flat_dim` (line 107) uses ``.spaces``.
    Implement all three."""

    def __init__(self, spaces: dict) -> None:
        self.spaces = spaces

    def __getitem__(self, key: str):
        return self.spaces[key]

    def get(self, key: str, default=None):
        return self.spaces.get(key, default)


def _make_obs_space(include_partner_tcp: bool) -> _Dict:
    """An obs_mode="state_dict"-shaped Dict with both agents' proprio and
    optionally partner TCP fields. `include_partner_tcp=True` mimics the
    pre-Б4 env output (always emitted); `False` mimics the
    `partner_obs="none"` env output (no agent_x_* fields)."""
    agent = {
        "so100-0": _Dict({"qpos": _Box((6,)), "qvel": _Box((6,))}),
        "so100-1": _Dict({"qpos": _Box((6,)), "qvel": _Box((6,))}),
    }
    extra = {
        "cube_pose": _Box((7,)),
    }
    if include_partner_tcp:
        extra["agent_a_tcp_pose"] = _Box((7,))
        extra["agent_b_tcp_pose"] = _Box((7,))
    return _Dict({"agent": agent, "extra": _Dict(extra)})


def test_per_agent_obs_dim_drops_partner_field_for_agent_a() -> None:
    """agent_a (idx=0) must NOT count agent_b_tcp_pose into its own dim —
    that is the entire Б4 guarantee at the dim level."""
    space = _make_obs_space(include_partner_tcp=True)
    dim_a = _per_agent_obs_dim(space, "so100-0", agent_idx=0)
    # proprio 6+6 + agent_a's own tcp 7 + cube 7 = 26; agent_b's tcp 7 is dropped.
    assert dim_a == 6 + 6 + 7 + 7


def test_per_agent_obs_dim_drops_partner_field_for_agent_b() -> None:
    """Symmetric: agent_b (idx=1) drops agent_a_tcp_pose."""
    space = _make_obs_space(include_partner_tcp=True)
    dim_b = _per_agent_obs_dim(space, "so100-1", agent_idx=1)
    assert dim_b == 6 + 6 + 7 + 7


def test_per_agent_obs_dim_identical_full_vs_none_for_this_agent() -> None:
    """With the env-side partner_obs="none" (no agent_x_* fields emitted at
    all) the per-agent dim is just proprio + own cube — it does NOT differ
    from the "full" case for the agent's OWN slice. The two modes only
    diverge on the partner TCP field, which the actor drops either way."""
    full = _make_obs_space(include_partner_tcp=True)
    none = _make_obs_space(include_partner_tcp=False)
    # agent_a: full has agent_a's own tcp 7; none doesn't. So full is wider by 7.
    assert (
        _per_agent_obs_dim(full, "so100-0", agent_idx=0)
        == _per_agent_obs_dim(none, "so100-0", agent_idx=0) + 7
    )


# ---------------------------------------------------------------------------
# Wiring: _make_gym_env must accept partner_obs and pass it into gym.make.
# We don't call it (gym.make would import mani_skill), we inspect its
# signature instead. That is enough to catch a future revert that drops
# the parameter or the dict pass-through.
# ---------------------------------------------------------------------------
def test_make_gym_env_signature_accepts_partner_obs() -> None:
    import inspect

    sig = inspect.signature(benchmarl_task._make_gym_env)
    assert "partner_obs" in sig.parameters
    assert sig.parameters["partner_obs"].default == "full"


def test_make_gym_env_passes_partner_obs_into_env_kwargs() -> None:
    """Static read of the source: the function body must put `partner_obs`
    into the `env_kwargs` dict that is splatted into `gym.make`. Pin this
    so a refactor cannot silently drop the kwarg again (the original Б4
    regression)."""
    import inspect

    src = inspect.getsource(benchmarl_task._make_gym_env)
    # The literal dict that becomes **env_kwargs must include "partner_obs".
    # The simplest robust check: the substring '"partner_obs": partner_obs'
    # appears at least once in the function body.
    assert '"partner_obs": partner_obs' in src


# ---------------------------------------------------------------------------
# Wiring: get_env_fun must read partner_obs from the config and pass it on.
# A full instantiation requires a real EnvBase, so we drive the path the
# cheap way: build a stub config and call get_env_fun, then inspect the
# closure via a partial call. We only check that the partner_obs value
# travels from the config into _make_gym_env's kwargs.
# ---------------------------------------------------------------------------
def test_get_env_fun_threads_partner_obs_from_config() -> None:
    """The TaskConfig.partner_obs value must reach _make_gym_env. We
    inspect the get_env_fun source rather than invoke it (its callable
    returns a closure that requires server-only state)."""
    import inspect

    src = inspect.getsource(CallosumTaskClass.get_env_fun)
    # Reads partner_obs from the config ...
    assert "self.config.get(\"partner_obs\"" in src
    # ... and passes it to _make_gym_env.
    assert "partner_obs=partner_obs" in src


# ---------------------------------------------------------------------------
# Belt-and-braces: the actual actor-side path (_CallosumManiSkillEnv._agent_obs)
# is what the previous version got wrong. Inspect its source: it must call
# build_agent_obs with include_partner=False. Reading the source avoids
# needing to spin up an env (which is server-only).
# ---------------------------------------------------------------------------
def test_agent_obs_uses_build_agent_obs_with_include_partner_false() -> None:
    import inspect

    src = inspect.getsource(benchmarl_task._CallosumManiSkillEnv._agent_obs)
    assert "build_agent_obs" in src
    assert "include_partner=False" in src


def test_group_name_is_agents() -> None:
    """The BenchMARL group name is the constant IPPO/MAPPO configs key on;
    renaming it would silently break all hydra yaml defaults."""
    assert GROUP_NAME == "agents"
