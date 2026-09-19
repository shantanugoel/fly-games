# Self-learning plan: remove the teacher, replace the walker

Goal: an end state with **no authored behaviour in the loop**.

```
fly brain (frozen) -> 12 command rates -> readout (learned from outcome) -> buttons
                                 ^
                    its own rollouts + MidStart states it reached itself
```

The brain stays frozen. The readout is the only learnable surface, and it is
learned from **what happened**, not from an action label I wrote by hand.

## Why this is the right shape

- The readout is ~48 parameters (12 command rates x 4 labels). At that size
  evolutionary search is competitive with gradients and far easier to get right.
- The simulator is deterministic: identical trajectories across seeds and
  processes. Fitness has no noise, which is what usually makes ES painful.
- Rollouts are cheap since the clock fix: ~500 decisions in ~3 s (1600 samples
  in 8.5 s measured). A population of 16 is ~45 s per generation.
- It removes the last scripted supervision, which is the line this project has
  been walking since the escape-reflex override was deleted.

## Organising rule for "careful"

**Nothing is deleted until its replacement is measured at least as good, and the
two must coexist long enough to be compared on one metric.** Every phase ends
with the suite green and a commit. Do not collapse phases.

---

## Phase 0 - the sensorium gate

Random-policy rollouts. For each decision, measure how well the 12 command rates
predict future delta-x over a K-decision window (AUC or mutual information).

This decides whether Phases 1-4 are worth writing, and the rollouts are the same
ones Phase 1 needs, so nothing is wasted.

Three outcomes, three different projects:

| result | meaning |
|---|---|
| signal in command rates | self-learning is viable, proceed to Phase 1 |
| signal only in the 4 sensory channels | the readout should read encoder drive, not motor urges - stop and redesign |
| no signal anywhere | widen `CHANNELS` with real cell types first; learning is downstream of sensing |

Evidence that this gate is needed, all measured:

- ~3 spikes fire at rest across all 12 command groups.
- `forward` is zero on nearly every decision, which is why any "quiet means
  halt" rule stops Mario dead (656 of 708 frames standing still).
- After the side-duplication collapse there are effectively ~4 usable channels,
  not 8.
- `Dm15/Dm16/Dm18` (350 neurons) are dead at every injected voltage while
  `T4a-T5d` (6,790) respond, so pool size does not predict drive.

**Acceptance:** a number for predictiveness per channel, written into this file,
and a decision recorded on which branch applies.

---

## Phase 1 - self-play collection, additive

`src/fly_games/engine.py:554` in `collect_fly_samples`:

```python
fine_action, _ = game.scripted(obs)
```

Add a source rather than replacing it:

```
train --source {teacher,self}      # default teacher: behaviour unchanged
```

`self` draws from the current readout's distribution (uniform when no readout
exists yet) and records `(feature, action, x_after)` per decision. The teacher
path is untouched, so both can be A/B'd on identical metrics.

**Acceptance:** `--source teacher` reproduces today's numbers exactly;
`--source self` produces samples and trains without error.

---

## Phase 2 - outcome labelling

```
score[i] = x[i+K] - x[i]           # K ~ 8 decisions
death    -> penalty larger than any achievable progress gain
keep top quantile -> fit (feature -> action taken)
```

`K` is not optional. Backing off to get a run-up *loses* x for ~10 decisions and
then gains much more, because jump height scales with running speed. A one-step
label would learn "never retreat" and reproduce the wall bug with data behind it.

New metric: fitness = max_x / deaths / completion, evaluated on a **frozen seed
set never used for selection**. Keep leave-one-episode-out accuracy only while
teacher labels still exist; it stops being comparable once exploration changes
the action distribution.

**Acceptance:** self-trained readout reaches at least x=898 (the current
teacher-trained distance) on the frozen eval seeds, and the eval set is committed
so it cannot be tuned against.

---

## Phase 3 - replace the walker, then delete it

`_seek_start` (`engine.py:153`) walks with `scripted` to a target x, captures,
restores. That is navigation, not supervision, but it is still authored
behaviour carrying the fly past hard parts.

Replacement: **capture during self-play.**

- During any rollout, when x crosses a target and `alive and grounded`, call
  `dump_state()` and persist to `states/mario/<x>.npz` with metadata. Gate on
  alive+grounded exactly as the reference `capture_state.py` does.
- `--start-at` loads the nearest captured blob; falls back to walking only when
  no blob exists.
- Curriculum grows: record deepest captured x, extend it each generation as the
  readout improves. Bootstrapping is expected - an untrained fly cannot get deep,
  so the curriculum lengthens with competence.

Verified enablers: `dump_state`/`load_state` round-trip exactly (all 64 KB of RAM
identical after restore), and `--start-at 880` now yields `offset 883`.

**Deletion trigger:** keep the scripted walker until self-captures cover the
range actually trained on. Then delete the walk and keep the blob loader.

**Acceptance:** every `--start-at` value used in training resolves to a
self-captured blob, with the walker path unreachable.

---

## Phase 4 - delete the teacher and `fly-hand`

Only after Phases 1-3 measure at least as good.

1. **`fly-hand`** (`cli.py:23` `POLICIES`, and `hand_decode` in each game's
   `fly.py`) - gone. Decide the no-readout behaviour explicitly: **refuse and
   print the `train` command.** Silent substitution is how you get a demo that is
   secretly not the fly.
2. **`scripted`** - out of `games/*/game.py`, `POLICIES`, and README. Move a
   minimal version into `tests/` as a fixture: several tests use it as the
   deterministic competent policy, and tests asserting *progress* (`moved`,
   `distinct_x`) will not hold under seeded random actions. A test-only stub is
   legitimate - a fixture is not shipped behaviour.
3. **`coarse_of`** supervision for Mario - gone. The label is the action; the
   outcome does the selecting.

Achievability ceiling, previously supplied by `scripted`, can be filled by MPC
(`dump_state`/`load_state` search) - no authored behaviour, and arguably a
stronger ceiling.

**Acceptance:** `grep -rn "scripted\|fly_hand_decode" src/` returns nothing;
suite green; README documents the new interface.

---

## Practical constraints from this codebase

- **The fly is one animal.** `decision_lock` exists because only one thing may
  step it. Population rollouts need separate **processes**, not threads.
- **Assert determinism in a test** (same seed -> identical trajectory) so a future
  refactor cannot silently break fitness comparability.
- Changing the snapshot shape requires bumping the content-addressed point-cloud
  `?rev=` so browsers stop serving stale tissue.

## Guardrails, each learned the hard way in one session

- **Re-grep after every edit.** One edit silently dropped `--hold` while its
  neighbours survived; a reflex-override edit never applied and an A/B test
  measured unmodified code; `--start-at` was capped at x=594 while reporting
  success. Edits fail atomically and silently.
- Commit each phase after the measurement, not after the code.
- **The numbers will get worse before they get real.** `fly-hand` and `scripted`
  are what made a broken system look alive: the reflex override survived 150
  decisions and was nearly read as the fly being competent, when it was my
  threshold. Self-learning scoring below the teacher at first is the expected
  signature of removing a crutch - not a reason to re-add one.
- Do not supervise on hand-authored action labels. That was the lesson from
  `super-mario-bros-rl`, which derives reward from RAM `full_x` and never from a
  authored action.

## State at handoff

`scripted 2476 / fly 898 / fly-hand 595`, 182 tests passing, HEAD `f2c214c`.

Known open items not in this plan: Kung Fu and Doom readouts were fitted under
the broken `hold_frames=1` default and need refitting; `hold_frames` may want a
per-game sweep.