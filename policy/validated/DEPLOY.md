# Validated deploy ONNX (sim2sim 817-pass, g1_tactile)

- tracker_armtight.onnx  — deployable tracker. body 0.086, wrist 0.11, arms move. 283-dim proto obs.
  Use configs/g1_23dof_armtight.yaml (clip-replay via ReferenceProvider). Drop-in for g1_23dof_tracker.
- compliant_v3.onnx      — compliant (chair-push). tight free 0.092 / yields pushed 0.118 / smooth.
  Same 283 proto obs. NEEDS the low-gain arm gains on the real robot (g1_23dof_compliant: arm kp ~30-40%).

Future at deploy: clip-replay = real future (works now). Live teleop = distillation (future-privileged
teacher -> current-obs student) OR a planner; extrapolation was tried+dropped. See plans/3POINT_PROGRESS.md.
