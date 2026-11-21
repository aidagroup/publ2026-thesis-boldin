"""Reusable, experiment-agnostic building blocks of the thesis repo.

`callosum.common` holds the pieces every experiment shares: the config/CLI plumbing,
the PPO machinery and categorical head, checkpointing, non-finite guards, metric
logging/aggregation, the Bi-JEPA module and the generic partner-visibility rule.
Experiment-specific code lives under `callosum/experiments/<name>/`; importing this
package must not pull in a simulator (mujoco or SAPIEN).
"""
