#!/usr/bin/env python3
"""Compare MuJoCo depth against the training Warp renderer at identical saved poses.

No policy execution, viewer, robot or sensor connections. --legacy-mesh reproduces
pre-fix ray geometry for diagnosis; it is not a policy/deployment option.
"""
import argparse
import json
from pathlib import Path
import sys

from steerable_recovery.paths import ROOT
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path,help='Folder containing config.json and trajectory.npz')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stride',type=int,default=25)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--legacy-mesh',action='store_true')
    args=parser.parse_args()
    if args.stride<1:parser.error('--stride must be positive')
    import numpy as np
    import torch
    import mujoco
    from omegaconf import OmegaConf
    from motion_diffusion.inference.bundle import load_bundle, asset_path
    from motion_diffusion.inference.policy import load_controller
    from motion_diffusion.inference.mujoco import scene_model, MujocoNativePlant, MujocoDepth
    from motion_diffusion.terrain.planner_depth import PlannerDepthRenderer,depth_view
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads((args.episode/'config.json').read_text())
    bundle,manifest=load_bundle(config['bundle']);row=manifest['models'][config['model']]
    controller=load_controller(dict(controller_checkpoint=str(asset_path(bundle,row['controller'])),
        tracker_device='cpu',pd_sidecar=str(asset_path(bundle,row['pd_binding']))))
    training=OmegaConf.load(asset_path(bundle,row['controller_training_config']))
    plant=MujocoNativePlant(scene_model(config['robot'],**config['scene']),controller.contract,config['pd'],
                           OmegaConf.to_container(training.robot,resolve=True))
    names=[mujoco.mj_id2name(plant.model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(1,plant.model.nbody)]
    capture=dict(config['camera']);capture['robot_urdf']=str((ROOT/capture['robot_urdf']).resolve())
    from motion_diffusion.inference.mujoco_terrain import terrain_geometry
    vertices,triangles=terrain_geometry(plant.model)
    renderer=PlannerDepthRenderer(vertices=vertices,triangles=triangles,
        body_names=names,config=capture,device=args.device)
    rows=[];images=[]
    with np.load(args.episode/'trajectory.npz') as trajectory:
        for tick in range(0,len(trajectory['qpos']),args.stride):
            plant.data.qpos[:]=trajectory['qpos'][tick];mujoco.mj_forward(plant.model,plant.data)
            sensor=MujocoDepth(plant,capture,42)
            if args.legacy_mesh:
                from types import SimpleNamespace
                sensor.scene=SimpleNamespace(model=plant.model,data=plant.data,sync=lambda _:None)
            sensor.capture_tick(0,plant.observation());actual=sensor.records[0]
            pos=torch.tensor(plant.data.xpos[1:].copy(),device=args.device,dtype=torch.float32)[None]
            quat=torch.tensor(plant.data.xquat[1:,[1,2,3,0]].copy(),device=args.device,dtype=torch.float32)[None]
            expected=renderer.render(pos,quat)
            a=depth_view(actual['depth_mm'],actual['pixel_status'],max_range_m=2.,range_mode='saturate').numpy()
            b=depth_view(expected['depth_mm'][0].cpu(),expected['pixel_status'][0].cpu(),max_range_m=2.,range_mode='saturate').numpy()
            difference=np.abs(a[0]-b[0]);common=(a[1]*b[1]).astype(bool)
            rows.append(dict(tick=tick,mask_disagreement=float((a[1]!=b[1]).mean()),
                depth_mae_m=float(difference.mean()),max_depth_error_m=float(difference.max()),
                common_depth_max_m=float(difference[common].max()) if common.any() else 0.,
                over_1cm_fraction=float((difference>.01).mean()),
                mujoco_near_fraction=float((actual['pixel_status'].numpy()==2).mean()),
                status_disagreement=float((actual['pixel_status'].numpy()!=expected['pixel_status'][0].cpu().numpy()).mean())))
            images.append(np.stack([a,b]))
    report=dict(episode=str(args.episode.resolve()),legacy_mesh=args.legacy_mesh,poses=len(rows),rows=rows,
        scope='Identical physical terrain/link poses and camera calibration; MuJoCo raycaster versus training URDF/Warp; no image corruption',
        max_mask_disagreement=max(r['mask_disagreement'] for r in rows),
        max_depth_error_m=max(r['max_depth_error_m'] for r in rows))
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output/'depth_images.npz',images=np.stack(images),ticks=[r['tick'] for r in rows])
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
