"""Naming convention of the per-step diagnostics an env publishes in its `info` dict.

Pure Python (no mani_skill/torch import). An env marks a diagnostic by prefixing its `info` key
with `DIAG_PREFIX` (`diag_holder_grasp`, ...); the trainer averages every such key generically
(`callosum.training._diag`) and an env without any (TwoSO101-v0) simply has none. Diagnostics
never feed back into the reward or the observation.
"""

DIAG_PREFIX = "diag_"
"""Prefix of the `info` keys that are diagnostics (per-env tensors of shape `(num_envs,)`)."""


def diag_name(info_key: str) -> str | None:
    """The diagnostic's short name (`diag_face_angle` -> `face_angle`), or None for other keys."""
    if info_key.startswith(DIAG_PREFIX) and len(info_key) > len(DIAG_PREFIX):
        return info_key[len(DIAG_PREFIX) :]
    return None
