# 3. Config

Configs live in [`configs/`](../configs/), selected with `--config`.

| Config | Use |
| --- | --- |
| `g1_23dof.yaml` | fall recovery |
| `g1_23dof_sonic.yaml` | SONIC tracker + tactile, run with `deploy_eskin.py` |

Set these for a first run:

```yaml
use_depth_obs: false
use_tactile: false
policy_path: <your checkpoint>
urdf_path: <your g1_23dof_rev_1_0.urdf>
```

`policy_path` and `urdf_path` default to paths under `~/HumanoidVerse2/`, which you
probably don't have.

`obs_dim` / `num_obs` / the obs scales must match the trained config. Mismatched, the
policy still runs but behaves wrong. Don't tune them to make shapes fit.

## Depth as policy input

Only relevant if the policy was trained with depth. To just look at the camera, see
[4. Cameras](04_cameras.md).

[`core/common/depth_client.py`](../core/common/depth_client.py) turns each frame into the obs vector:
`cv2.imdecode` → center-crop to square (`640×480` → `480×480`) → `×depth_scale` to meters →
resize to `64×64` (`INTER_AREA`) → clip to `[min_d, max_d]` → normalize to `[0,1]` →
**subtract `0.5`** → flatten to `depth_obs_dim`.

```yaml
use_depth_obs: true
depth_server_ip: "192.168.123.164"   # robot onboard PC
depth_server_port: 5556
min_d: 0.1                           # meters, from sim ego_camera.clipping_range
max_d: 2.0
depth_scale: 0.001                   # raw units -> meters
depth_img_height: 64                 # obs resolution
depth_img_width: 64
depth_obs_dim: 4096                  # must equal height*width
depth_raw_height: 480                # what the server actually sends
depth_raw_width: 640
```

`min_d`/`max_d` come from `ego_camera.clipping_range` in sim and must match training. The
final range is `[-0.5, 0.5]`, not `[0, 1]`.

---

Next: [4. Cameras](04_cameras.md)
