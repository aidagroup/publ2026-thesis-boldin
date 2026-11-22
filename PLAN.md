# PLAN.md

Working plan, pinned 2026-10-08. The full step-by-step spec is [docs/implementation-plan.md](docs/implementation-plan.md); this file is the short "what next" list. Update it when a block is done.

## Standing rules

- Model roles: Haiku subagents write the code, Sonnet reviews (diff, tests, lint) and commits/pushes, Opus plans and does the global review after a whole stage. Docs-only edits can be done directly.
- Never run simulation, probes, demo collection, BC or IPPO training on the Mac. Locally only `ruff`, small `pytest` and code reading; give the user server commands.
- Commit subject `type: msg` (the hook adds the branch prefix), no attribution lines. Push to the step branch routinely; merge into `main` only server-verified, stable work.
- Simulation only. Reply to the user in English; docs stay in Russian, code and comments in English.

## Thesis direction

Main line: skill-level MARL (step 2.3, `FaceTurnSkills-v0`). Each arm picks a discrete scripted primitive; both agents know the goal and the cube-solving algorithm. Partner modes: `none` (control), `oracle`, `predicted` (Bi-JEPA). Secondary line: joint-level IPPO + BC (step 2.1), kept as motivation (the BC rotator cannot see the holder's jaw close) and as reusable infrastructure.

## Now: finish persistent storage (branch `step/2.1-ippo-so101`)

Goal: everything permanent lives in `$HOME` (100 GB); only temporary files use `/tmp`.

1. [ ] Finish the storage change (uncommitted: `scripts/setup_server.sh`, `update_server.sh`, `diagnose_vulkan.sh`, `docs/setup.md`, `docs/server-runbook.md`, `CLAUDE.md`, `README.md`). The storage subagent (id a9861cae649f8e696, ran on Sonnet) died on a rate limit before the runbook was finished; resume it after the limit resets (2:30pm Moscow) or re-delegate.
   - Layout: checkout `~/callosum`, data root `~/.callosum` (venv, uv cache, python, HF, ManiSkill, SAPIEN, mesa, XDG cache), `~/callosum-runs` behind the `runs/` symlink.
   - Scratch layout only via explicit `CALLOSUM_SCRATCH` / `CALLOSUM_LAYOUT=scratch` or when `$HOME` is low on space; drop the `JUPYTERHUB_USER` trigger.
   - Clean up old leftovers: the `/tmp` exports in `~/.callosum-env.sh`, the bashrc hook, the dangling `~/.sapien` symlink.
   - Docs: every `/tmp/callosum` becomes `~/callosum` / `$CALLOSUM_REPO`.
2. [ ] Review the diff: `bash -n` on all scripts, fresh-install and leftover-cleanup paths, no remaining `/tmp/callosum`. Then `ruff`, `pytest -q`, commit, push.
3. [ ] Give the user the server commands:
   - fresh install: clone `https://github.com/aidagroup/publ2026-thesis-boldin.git` (branch `step/2.1-ippo-so101`) into `~/callosum`, then `bash scripts/update_server.sh`;
   - check `ls ~/callosum-runs/ft_full_s1/`; if `latest.pt` exists, resume with `python -m callosum.training.ippo --resume runs/ft_full_s1` (detached, log appended with `>>`); otherwise relaunch the fine-tune: `--checkpoint runs/bc_full/bc.pt --demos runs/demos/faceturn_a.pt --critic-warmup-iters 10 --bc-coef 1.0 --bc-decay-iters 100 --learning-rate 1e-4 --partner-obs full --total-timesteps 10000000 --seed 1 --exp-name ft_full_s1`.

## Pending reports from the user

- Fine-tune progress: `ever hold`, `ever rot`, `eval_succ` for `ft_full_s1`.
- Friction probe (`--face-friction 20/10/5`) to see whether the remaining ~29% demo failures come from body rotation.

## Next: step 2.3 skill-level env (branch `step/2.3-skill-level-marl`)

Branch from `main` once 2.1 is stable, or from 2.1. Scope is v1 only (cube on the table).

- Primitives. Holder: wait / approach / grasp / release / lift / lower / retreat. Rotator: wait / approach / grasp / turn90 / turn180 / release / retreat. Face joint range extended to 180°; the goal (90° or 180°) is drawn at reset and observed by both agents.
- Synchronous decisions every ~10 sim steps (`decision_period`); a chosen primitive continues until the choice changes, and a change interrupts it.
- Coordination must be non-trivial (all in config): random primitive speeds per episode and arm, grasp failure probability, random partner start delay. Otherwise `none` equals `oracle` and the ablation measures nothing.
- Engineering: batched GPU IK (or per-reset precompute). The current per-env Python IK takes ~290 s per 256-env reset.
- Trainer: existing IPPO with a categorical head; one trainer step per decision.
- Work order: (1) pure logic `callosum/envs/_skills.py` plus Mac unit tests; (2) env plus a scripted conductor that must reach ~100% success; (3) categorical head and IPPO runs in `none` and `oracle`.
- Done when (server): env builds and resets in acceptable time, conductor ~100%, `oracle` above ~90%, `none` clearly worse, metrics logged (success, time to success, useless actions, partner-primitive prediction accuracy).

## Then

1. Merge `step/3.1-bijepa-module` (expect doc conflicts in `docs/implementation-plan.md` and `docs/thesis/03-method-bijepa.md`).
2. Step 3.2: integrate Bi-JEPA into the skill-level policy (`predicted` mode with the same execution-time inputs as `none`).
3. Step 3.3: ablation `none / oracle / predicted`, at least 3 seeds; headline metric is the fraction of the oracle-none gap closed.
4. v2 primitives (holder flips 90/180, rotator side approach).
5. Merge to `main` only after server verification; the branch 1.6 video re-render is still unverified.
