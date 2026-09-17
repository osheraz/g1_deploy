#!/usr/bin/env python3
"""Show sampled Z and recovery variation with exactly matched initial conditions.

Only the planner seed varies. Pose, physics, camera delivery seed, weights,
backend and inference recipe are fixed. This measures diversity; a small seed
panel is not proof that a checkpoint learned distinct semantic modes.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def compare(directory,model,output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths=sorted(Path(directory).glob(model+'_*/trajectory.npz'))
    if len(paths)<2:raise ValueError('At least two completed episodes are required')
    configs=[json.loads((p.parent/'config.json').read_text()) for p in paths]
    conditions=[];traces=[];plans=[];summaries=[]
    for path in paths:
        with np.load(path.parent/'first_condition.npz') as a:conditions.append({k:a[k].copy() for k in a.files})
        with np.load(path) as a:traces.append({k:a[k].copy() for k in a.files})
        plans.append(json.loads((path.parent/'plans.json').read_text())[0])
        summaries.append(json.loads((path.parent/'summary.json').read_text()))
    common=[k for k in conditions[0] if k not in ('state_noise','code_noise')]
    for c in conditions[1:]:
        for k in common:np.testing.assert_array_equal(c[k],conditions[0][k],err_msg=f'Condition changed: {k}')
    for cfg in configs[1:]:
        for k in ('model','controller_sha256','planner_sha256','inference','backend','planner_precision','engine_mode','pd','camera'):
            if cfg[k]!=configs[0][k]:raise ValueError(f'Comparison configuration differs: {k}')
    seeds=[c['seed'] for c in configs]
    z=np.array([p['native_code'] for p in plans]);state=np.array([p['state'] for p in plans])
    def pairwise(x):
        delta=x[:,None]-x[None,:]
        return np.sqrt(np.mean(delta**2,axis=tuple(range(2,delta.ndim))))
    count=min(len(t['q']) for t in traces)
    z_dist=pairwise(z);q_dist=pairwise(np.array([t['q'][:count] for t in traces]))
    report=dict(model=model,seeds=seeds,identical_initial_condition=True,checked_condition_arrays=common,
        zero_initial_velocity=all(not np.any(c['initial_qvel']) for c in conditions),
        distinct_initial_code_noise=len({c['code_noise'].tobytes() for c in conditions}),
        distinct_sampled_code_trajectories=len({x.tobytes() for x in z}),
        first_plan_code_rms_distance=z_dist.tolist(),rollout_joint_rms_distance_rad=q_dist.tolist(),
        upright_hold_1s=[s['upright_hold_1s'] for s in summaries],
        first_upright_hold_s=[s['first_upright_hold_s'] for s in summaries],
        interpretation='Different noise and Z/physical trajectories demonstrate stochastic diversity. They do not by themselves establish separate semantic modes or their probabilities.',
        reset=[s['reset'] for s in summaries])
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(output/'sampled_first_plans.npz',seeds=seeds,native_code=z,state=state,
        state_noise=np.array([c['state_noise'] for c in conditions]),code_noise=np.array([c['code_noise'] for c in conditions]))
    colors=plt.get_cmap('tab10')(np.arange(len(seeds))%10)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    flat=z.reshape(len(z),-1);centered=flat-flat.mean(0)
    u,s,v=np.linalg.svd(centered,full_matrices=False);pc=u[:,:2]*s[:2]
    for i,(seed,t) in enumerate(zip(seeds,traces)):
        axes[0,0].scatter(pc[i,0],pc[i,1],color=colors[i]);axes[0,0].annotate(str(seed),pc[i,:2])
        axes[0,1].plot((np.arange(len(state[i]))+1)*.02,state[i,:,3],color=colors[i],label=str(seed))
        axes[1,0].plot(t['time'],t['root_position'][:,2],color=colors[i])
        axes[1,1].plot(t['time'],t['q'][:,3],color=colors[i])
    axes[0,0].set(title='First sampled Z horizon: PCA (not a mode classifier)',xlabel='PC1',ylabel='PC2')
    axes[0,1].set(title='First plan: left knee relative to default',xlabel='Horizon time (s)',ylabel='rad')
    axes[0,1].legend(title='Planner seed',ncol=2)
    axes[1,0].set(title='Executed recovery: pelvis height',xlabel='Policy time (s)',ylabel='m')
    axes[1,1].set(title='Executed recovery: left knee',xlabel='Policy time (s)',ylabel='rad')
    for a in axes.flat:a.grid(alpha=.2)
    fig.suptitle(f'{model}: identical resting pose and sensor condition; different planner noise')
    fig.savefig(output/'diversity.png',dpi=160);plt.close(fig)
    return configs,traces,seeds,report


def video(configs,traces,seeds,output):
    import mujoco
    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from motion_diffusion.behavioral.mujoco import flat_model
    model=flat_model(configs[0]['mjcf']);data=mujoco.MjData(model)
    width,height=400,360;columns=min(4,len(seeds));rows=(len(seeds)+columns-1)//columns
    model.vis.global_.offwidth=width;model.vis.global_.offheight=height
    camera=mujoco.MjvCamera();camera.distance=2.4;camera.azimuth=135;camera.elevation=-18
    opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,0,0,0,0]
    pelvis=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
    try:font=ImageFont.truetype('DejaVuSans.ttf',16)
    except OSError:font=ImageFont.load_default()
    def frame(tick):
        canvas=np.zeros((rows*height,columns*width,3),np.uint8)
        for i,(cfg,trace,seed) in enumerate(zip(configs,traces,seeds)):
            j=min(tick,len(trace['qpos'])-1);data.qpos[:]=trace['qpos'][j]
            mujoco.mj_forward(model,data)
            camera.lookat[:]=data.xpos[pelvis];camera.lookat[2]=max(.35,camera.lookat[2])
            renderer.update_scene(data,camera=camera,scene_option=opt)
            tile=Image.fromarray(renderer.render());draw=ImageDraw.Draw(tile)
            draw.rectangle((0,0,width,27),fill=(15,18,24))
            draw.text((8,3),f'Seed {seed} | t={j*.02:.2f}s',fill='white',font=font)
            row,col=divmod(i,columns);canvas[row*height:(row+1)*height,col*width:(col+1)*width]=np.asarray(tile)
        return canvas
    with mujoco.Renderer(model,height=height,width=width) as renderer:
        with imageio.get_writer(str(output/'samples.mp4'),fps=25,codec='libx264',quality=7,macro_block_size=1) as writer:
            initial=frame(0)
            imageio.imwrite(output/'resting_starts.png',initial)
            for _ in range(25):writer.append_data(initial)
            for tick in range(0,max(len(t['qpos']) for t in traces),2):
                picture=frame(tick);writer.append_data(picture)
                if tick==150:imageio.imwrite(output/'recoveries_at_3s.png',picture)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('diffusion','flow'),default='diffusion')
    p.add_argument('--start',choices=('prone','supine','left_side','right_side','seated'),default='supine')
    p.add_argument('--samples',type=int,default=8)
    p.add_argument('--seconds',type=float,default=8.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--pose-seed',type=int,default=123)
    p.add_argument('--sensor-seed',type=int,default=123)
    p.add_argument('--results-dir',type=Path,help='Analyze existing matching sim_diffusion episodes instead of running new ones')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if not 2<=args.samples<=12:p.error('Use 2 to 12 samples for a readable video grid')
    os.environ.setdefault('MUJOCO_GL','egl')
    output=(args.output or ROOT/'logs/native_recovery'/f'sample_diversity_{time.time_ns()}').resolve()
    output.mkdir(parents=True,exist_ok=False)
    directory=args.results_dir or output/'rollouts'
    if args.results_dir is None:
        command=[sys.executable,str(ROOT/'g1_deploy/sim_diffusion.py'),'--model',args.model,
                 '--start',args.start,'--episodes',str(args.samples),'--seconds',str(args.seconds),
                 '--seed',str(args.seed),'--pose-seed',str(args.pose_seed),'--sensor-seed',str(args.sensor_seed),
                 '--output',str(directory)]
        with (output/'run.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    configs,traces,seeds,report=compare(directory,args.model,output)
    video(configs,traces,seeds,output)
    print(json.dumps(dict(output=str(output),identical_condition=report['identical_initial_condition'],
        distinct_Z=report['distinct_sampled_code_trajectories'],seeds=seeds),indent=2))

if __name__=='__main__':main()
