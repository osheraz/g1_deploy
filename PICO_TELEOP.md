# PICO 3-point teleop (deploy)

Live VR teleop for the reference-tracking WBC: stream **head + 2 controllers** from a
PICO headset (XRoboToolkit) straight into the policy as a sparse keypoint goal — **no IK**.
This is the SONIC/CHIP route (`../others/_WBC_SUMMARY/14_PICO_teleop_obs.md`).

## Pieces

- `policy/pico_reader.py` — `PicoReader`: background thread over `xrobotoolkit_sdk`,
  reads head + L/R controller poses, transforms Unity Y-up → Z-up (mirrors GMR
  `XRobotStreamer`), exposes `latest()` → `{head, left_wrist, right_wrist: (pos, quat_wxyz),
  *_grip, *_trig}`. Inert if the SDK is absent so the loop never blocks.
- `policy/pico_reference_provider.py` — `PicoReferenceProvider`: turns the 3 poses + the
  robot pelvis-anchor pos/quat into the goal obs blocks with the exact env getter math
  (`motion_anchor_pos_b/ori_b`, `ref_body_pos_b/ori_b`). Returns `{obs_name: block}`;
  `PolicyRunner` merges it so the `sorted(keys)` concat picks up the trained keys.

## Contract (must match the trained policy)

- `tracked_order` == trained `tracked_body_names`, in order. Default `(head, left_wrist, right_wrist)`.
- `emit` == the goal-obs keys in the trained obs config.
- Anchor body = `pelvis`. Quats are wxyz at the provider boundary.

## Not yet wired (intentional)

The current ONNX (`g1_23dof_tracker.yaml`) trained on dense `future_ref_motion_proto` — not
drivable by 3 sparse poses. PICO support is the deploy half of a policy trained on the sparse
keypoint obs (`+exp=reference_tracking_3point`). Once that ONNX exists:

1. `g1_23dof_pico.yaml` with `pico_teleop: true`, `pico_tracked_order`, `pico_emit`.
2. `PolicyRunner.__init__`: if `pico_teleop`, start `PicoReader` + `PicoReferenceProvider`
   instead of the clip `ReferenceProvider`.
3. `compute_action`: `values.update(provider.blocks(reader.latest(), ref_anchor, robot_anchor))`.
   World→pelvis calibration (headset origin ↔ robot pelvis) is the bench-test step.

## Next to-do

1. Train the 3-point tracker (`+exp=reference_tracking_3point`) — gated on the anchor-reward sigma fix.
2. Export ONNX; MuJoCo sim2sim gate (`tools/test_tracker_mujoco.py`).
3. Wire steps 1–3 above; bench-test `PicoReader` on a real headset.
4. Decide lower-body: learned stance (chair-push) vs base-velocity channel (walk-while-teleop).
5. Warm-start CHIP compliance on it: `ref_body_pos_b_hindsight` + `cmp_inv_k` from grip + history.
6. Map grip/trigger → tactile hand close + commanded stiffness.
