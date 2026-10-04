"""Robot (agent) definitions for ManiSkill.

Like `callosum.envs`, this package deliberately does not import its submodules (they need
`mani_skill`, which is Linux-only), so `import callosum.robots` stays clean on macOS/CI.
Import `callosum.robots.so101_parallel_gripper` explicitly to register the agent.
"""
