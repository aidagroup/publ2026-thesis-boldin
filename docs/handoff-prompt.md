# Handoff prompt

Paste everything below the line into a fresh session to resume this work.

---

You are my reviewer and co-engineer on a master's thesis project. I am not
fluent in ML jargon — expand abbreviations and explain terms when they first
appear. Answer in Russian. Be concise: I would rather have one measured number
than three paragraphs of reasoning. Do not re-derive things that are already
written down below.

## The project

`callosum` — two SO-100 robot arms solve a Rubik's cube cooperatively, and
coordinate **decentrally** via Bi-JEPA: each agent predicts its partner's
latent state instead of exchanging messages. The ablation triple is
`partner_input ∈ {none, oracle, predicted}`. Goal: a demo good enough for
ICRA.

- Repo: `aidagroup/callosum`, working branch `step/2.1-ippo`.
- Push needs the `github` SSH alias (plain `github.com` resolves to my old
  account) and the noreply commit email; the account blocks pushes that expose
  a real address.
- Never touch the local `backup/pre-identity-rewrite/*` branches.
- Never commit `.codegraph/` or `.codegraphy/`.
- `AGENTS.md`, `docs/thesis/status.md` and `docs/thesis/README.md` are mine —
  leave them alone.

## Where the code runs

My dev box is macOS: **ManiSkill and SAPIEN cannot be imported here.** Anything
touching the simulator is verified on the server, so it must be written
carefully and guarded by tests that do not import `mani_skill`.

The server is a lab JupyterHub. **No SSH, no GitHub.** Code goes over as an
archive:

```bash
make bundle                       # ~264 KB, sources only
make bundle VENDOR=1              # + the PhysX .so, only if scratch was wiped
```

Upload it, then on the server:

```bash
cd ~ && rm -rf callosum && tar xzf callosum.tar.gz && cd callosum
bash scripts/setup_jupyterhub.sh && source .callosum-env.sh
```

`runs/` is a symlink to `~/callosum-runs` so results survive re-extraction.
`docs/jupyterhub-runbook.md` has the whole story, including the Vulkan
loader fix that makes rendering work.

**Trick worth knowing:** to answer any ManiSkill/SAPIEN question locally,
download and unpack the wheel and read the source —
`pip download --no-deps mani_skill==3.0.1` then unzip. That has settled several
questions that would otherwise have cost a server round trip.

## The task, as currently built

`RubikCube-v0` (`callosum/envs/rubik.py`), on top of `FaceTurn-v0`:

1. The **holder** pinches the cube's rear two layers from its own side and
   lifts it to 9 cm.
2. The **rotator** grips a nub on the axis of the front layer and rolls its
   wrist 90°.
3. A completed turn applies a move to a logical `RubikState`
   (`callosum/envs/_rubik.py`, full 3×3×3, 42 tests) and snaps the hinge back
   to 0, so the same hinge serves any number of turns.

Constants that are **measured, not chosen** — re-derive with
`scripts/measure_gripper.py` and `scripts/solve_waypoints.py`:

| Quantity | Value | Why it is that value |
|---|---|---|
| Fixed blade's gripping face | x = +0.0079 in the Fixed_Jaw frame | from the URDF collision pads |
| Widest object that sits ON the wrist_roll axis | 1.58 cm | anything wider is dragged around an arc instead of spun |
| Gripper sweep radius, jaws closed | 4.26 cm | why the cube must be lifted: on the table the palm goes 1.4 cm under it |
| `ARM_BASE_OFFSET` | 0.34 m | a horizontal tool cannot bring the pocket closer than ~0.26 m to its own base |
| `LIFT_HEIGHT` | 0.090 m | 4.7 cm of table clearance during the turn |

## Working style that has paid off

- **Measure, don't reason.** Four separate rounds were lost to plausible
  arguments about geometry. Every one was settled by a number.
- **Every claim falsifiable without the server.** `tests/test_grasp_waypoints.py`
  is the guard rail; it caught two regressions by negative control.
- **Add the missing measurement before guessing again.** The `dq_h`/`dq_r`
  columns in `scripts/probe_grasp.py` (how far each arm still is from its
  commanded joint angles) diagnosed a stall in one run that three prior runs
  could not distinguish from a bad waypoint.
- Terse comments. Explain a number or a decision; skip the essay.
- Spawn subagents for bulk implementation and for research, then review their
  work myself — both agents this session flagged assumptions that turned out
  to be real bugs, and reviewing those flags was where the value was.

## Open items, most damaging first

1. **Only one cube face is reachable.** The holder grips the cube in one
   orientation for the whole episode, so the rotator can only ever turn the
   face pointing at it. A depth-1 scramble is solvable **16.7 %** of the time
   (only `F`/`F'`), depth-2 is **0 %** by construction (the scramble generator
   forbids repeating a face), depth-3 ≈ 0.33 %.
   I chose the fix "the holder re-orients the cube between turns", but the
   check I ran says **none of the six 90°/180° re-orientations is reachable**
   while keeping the grip: rotating the cube rotates the holder's tool off
   horizontal, and the arm cannot hold those directions at that position
   (best case 5 mm / 9° short; worst 100°). A relaxed sweep — is there ANY
   holder pose presenting another face, with the cube's position free — was
   started and killed by the OOM killer. **That sweep is the next thing to
   run**, on a coarser grid. If it also comes back empty, the honest options
   are: change the rig geometry, or scope the task to the reachable face set
   and say so in the thesis.

2. **The turn sign is probably inverted.** By kinematic derivation, a positive
   joint angle is counter-clockwise seen from outside the face, while
   `rubik.py` maps positive → the clockwise move. Every applied move would be
   wrong. Cheap to check: one scripted turn, read back `cube_colours`, compare
   against the expected `F` permutation.

3. **Scramble RNG is broken on partial resets.** `rubik.py` uses
   `self._episode_rng`, which is a single global stream and is not reseeded at
   all when `seed=None` — the ordinary vectorised auto-reset path. Should be
   `self._batched_episode_rng[env_idx]`.

4. **The Bi-JEPA target is near-trivial.** 324 of each agent's 364 observation
   dimensions are the shared cube-colour field, bit-identical between the two
   agents. A low `jepa_loss` would therefore prove very little about
   cross-agent modelling. Needs narrowing or encoding before the ablation
   means anything.

5. **Grip margins rest on an un-audited default.** The SO-100's gripper
   `force_limit = 100 N·m` is boilerplate shared with the arm joints, not a
   real servo figure. All the comfortable margins (200–1500× on holding the
   cube's weight, 23–155× on the reaction torque) come from it. Worth an
   ablation with a realistic cap before trusting any grasp-stability result.

6. **`is_grasping` checks the wrong axis.** It tests contact-force alignment
   against each jaw link's local **y**, but the SO-100's jaws close along
   **x**; it passes only because the threshold is 110°. Not urgent, but a
   one-line contact-force printout on the server would settle it.

## Immediate next step

The last server run stalled because the scripted controller drove each joint
at its own rate independently: the short joints arrived ~50 steps early and
the arm swept the gripper **9.5 cm below the tabletop**, where it jammed. That
is fixed (`synchronized_step` in `callosum/envs/_scripted_expert.py`, straight
joint-space path, +1.54 cm clearance) and committed, **but has not been re-run
on the server**.

So:

```bash
uv run python scripts/probe_grasp.py
```

`dq_h` should fall to ~0 by the end of the `reach` phase. If it does and
`held` is still 0, the next suspect is `is_grasping` (item 6). If `held`
appears but `lift` does not climb, it is grip force (item 5).
