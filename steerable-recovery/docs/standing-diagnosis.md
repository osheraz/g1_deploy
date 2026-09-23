# Standing hold: planner or tracker?

The Y mode uses the original motion encoder and the **same tracker**. It can
remove continued recovery/steering intent. It cannot independently guarantee
balance or remove a tracker feedback failure.

## Run the offline probes

No camera, DDS, odometry or robot connection is opened. Outputs go to
`logs/deployment_validation/standing_cause_20260922/`, not into the recordings.

```bash
cd /home/osher/hv2

# S50R: stood, ankle reversal, fell, stood again.
PYTHONPATH=g1_deploy/steerable-recovery .venv/bin/python -m \
  steerable_recovery.tools.diagnostics.standing_cause \
  paper/real_world/2_steering/gallery_turn_03_right_1790044594325550007 \
  --events 347 --interval 5.5 8.2 \
  --output logs/deployment_validation/standing_cause_20260922/steering

# S25R: terrain recovery and repeated standing ankle oscillation.
PYTHONPATH=g1_deploy/steerable-recovery .venv/bin/python -m \
  steerable_recovery.tools.diagnostics.standing_cause \
  paper/real_world/1_new_stairs/robot_s25r_diffusion_tensorrt_1790027059201917979 \
  --events 440 --interval 7.5 12.8 \
  --output logs/deployment_validation/standing_cause_20260922/terrain
```

`--events` is the later **control tick** of each adjacent before/after pair.
`--interval` is elapsed policy time in seconds. Pick intervals before inspecting
intervention results. The default paths above are the two previously identified
failure events, not newly selected successes.

The tool verifies trace integrity and model weights, reproduces the recorded
target, checks chronological measured/prior-issued history, and computes:

- All eight old/new combinations of history, native code and depth; exact
  Shapley attribution distributes interactions rather than adding isolated swaps.
- All combinations of the five history fields, then of the ten prior-action
  time rows. Single-joint probes are explicitly nonadditive.
- The exact same histories/depth with a constant original-encoder standing code
  or the preceding native code. **No fake history reset.**
- PyTorch FP32 vs captured TensorRT, including the depth encoder.
- Original P42 reconstruction of native codes, with correct S[t+1]/z[t] timing
  for q comparisons. Reconstruction is an imperfect diagnostic, not a proof
  that the motion is physically feasible.

The standing-code output sequence is a collection of **one-step probes**. Its
actions are not fed back into subsequent recorded history. It must never be
described as a simulated recovery or the trajectory that pressing Y would create.

## Controlled test design

1. **Interface check — completed on these recordings.** Baseline targets must
   match within 1e-4 rad; history rows must match actual measurements and prior
   issued actions. A failure here blocks model attribution until the interface
   discrepancy is explained. Do not change normalization or zero the history.

2. **Intent vs feedback — completed on these recordings.** Substitute inputs
   independently and jointly, including original-encoder standing intent.
   A large reversal with constant standing code establishes that a changing
   planner output is not necessary for that particular tracker response. It
   does not establish why the robot reached that state.

3. **Physical counterfactual — nominal paired simulations exist; reproducing
   the hardware failure remains outstanding.** First reproduce the failure's
   ankle-target/measurement sequence, not merely a visually similar fall.
   Save a branch point 0.3–0.5 s before the first growing oscillation, and a
   second point after genuinely quiet standing. Compare these four arms:

   | Arm | What changes at the branch |
   |---|---|
   | A | Continue the original planner/steering mode |
   | B | Unguided planning with the same model and sampling settings |
   | C | Current Y implementation: encoded standing with the existing blend |
   | D | Same handoff, with an encoded standing reference from a verified stable teacher segment |

   Use prone and supine on flat ground, then raised support and the platform-edge
   contact configuration from the failed recording. Every pair must share the
   full physics state (including actuator/warm-start state), robot/history
   buffers, current code/plan cursor, heading, depth delivery/filter state,
   controller settings, friction/DR draw and random-generator states. An exact
   repeated prefix is acceptable; verify equality before the branch. Restoring
   qpos alone or just setting the same seed is insufficient. Keep default sim
   depth timing unchanged; test measured hardware delay separately and label it.

   Record both the first 0.5 s and at least 5 s after the transition: falls,
   ankle/whole-body target changes, measured speed/acceleration, torque and
   contact impulses where available, support loss, slip and pelvis drift.
   Standing height alone is not the outcome. Include all attempted pairs.

4. **Handoff improvement — test only after step 3 identifies a failing case.**
   Compare the current fixed 0.5-s transition against one initialized with
   measured joint velocity and duration based on pose/velocity mismatch.
   Keep the eventual reference identical. Separately compare immediate Y with
   a brief settling-aware transition, using measured speed and body rotation,
   not just upright posture. These are candidate experiments, **not current
   robot behavior or approved thresholds**. Do not suppress/reset prior-action
   history to make an offline trace look quiet; that changes the tracker input
   contract and needs its own closed-loop evaluation/training.

5. **Decision and hardware confirmation.** If C prevents a reproduced failure
   when activated early but not late, call it prevention/avoidance. If C still
   fails under stationary intent, prioritize tracker/actuator/contact mismatch
   and standing/contact training; changing the planner alone is insufficient.
   If D improves over C, use the verified reference and retest transitions.
   If only the transition changes help, fix that boundary. Confirm a selected
   change on repeated operator-controlled robot runs with all traces recorded;
   successful nominal simulation does not close the hardware question.

No new real-robot command behavior was introduced by this investigation.
