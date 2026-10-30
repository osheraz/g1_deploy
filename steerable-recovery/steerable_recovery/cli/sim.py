#!/usr/bin/env python3
"""Run a published recovery pair in MuJoCo; use Isaac for paper evaluation."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import json
import os
import queue
from pathlib import Path
import sys
import time

from steerable_recovery.paths import ROOT, CONFIG_ROOT, recovery_asset
sys.path.insert(0,str(ROOT))


def arguments(argv=None):
    from motion_diffusion.inference.mujoco_terrain import SIM_TERRAINS, DEFAULT_CACHE
    from steerable_recovery.settings import sampling_seed, MODEL_NAMES
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model',choices=MODEL_NAMES,
        default='ours')
    parser.add_argument('--blend-ticks',type=int,help='Latent-plan overlap in 50 Hz control ticks; default uses the bundle recipe')
    parser.add_argument('--replan-hz',type=float,help='Override saved cadence on whole 50 Hz ticks; e.g. 2.5, 5, 10. Simulation time, not a compute guarantee')
    parser.add_argument('--guidance-scale',type=float,help='Override goal steering strength; does not change the trajectory or motor PD gains')
    parser.add_argument('--goal-json',type=Path,help='Shared Stanley steering cost; simulator pelvis feedback, PyTorch planner and selected tracker backend')
    parser.add_argument('--noise-policy',choices=('fresh','fixed_episode'),
        help='fresh: new noise every plan; fixed_episode: sample once per episode and reuse; default follows model bundle')
    parser.add_argument('--bundle',type=Path,help='Verified deployment bundle; defaults to the published primary pair')
    parser.add_argument('--robot',type=Path,help='Robot URDF or MJCF; defaults to the training G1-29 URDF')
    parser.add_argument('--terrain',choices=('flat','step',*SIM_TERRAINS),default='flat')
    parser.add_argument('--terrain-seed',type=int,default=42)
    parser.add_argument('--terrain-geometry',type=Path,help='Measured static-box geometry.json in metres/Z up; replaces flat terrain')
    parser.add_argument('--terrain-cache',type=Path,default=DEFAULT_CACHE)
    parser.add_argument('--depth-view',action=argparse.BooleanOptionalAction,default=True,
                        help='Show the actual planner depth input in the live window')
    parser.add_argument('--step-height',type=float,default=.15,help='Stair height in metres')
    parser.add_argument('--step-edge',type=float,default=.25,help='Front edge X in metres relative to initial pelvis; stair extends toward +X')
    parser.add_argument('--step-length',type=float,default=1.2,help='Stair tread length along X in metres')
    parser.add_argument('--step-width',type=float,default=2.,help='Stair width along Y in metres')
    parser.add_argument('--seconds',type=float,help='Episode duration; default: unlimited with --viewer, 10 s headless. Use 0 for unlimited')
    parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--start',choices=('standing','prone','supine','left_side','right_side','seated','random'),default='supine')
    parser.add_argument('--postures',nargs='+',choices=('standing','prone','supine','left_side','right_side','seated','random'),
                        help='Headless sweep: run --episodes seeds for each posture')
    parser.add_argument('--joint-jitter',type=float,default=0.,help='Seeded joint perturbation amplitude in radians, at most .3')
    parser.add_argument('--n-joint-jitter',type=float,default=.15,help='Joint perturbation (rad, at most .3) applied after pressing N')
    parser.add_argument('--n-xy-m',type=float,default=.1,help='N: pelvis XY offset range in metres')
    parser.add_argument('--n-tilt-deg',type=float,default=5.,help='N: roll/pitch offset range in degrees')
    parser.add_argument('--n-yaw-deg',type=float,default=15.,help='N: yaw offset range in degrees')
    parser.add_argument('--backend',choices=('pytorch','tensorrt','onnx'),help='Selected model default: P11 PyTorch TF32; legacy models TensorRT FP32')
    parser.add_argument('--planner-precision',choices=('fp32','tf32'),default='fp32',help='TensorRT only; PyTorch uses the saved bundle precision')
    parser.add_argument('--engine-mode',choices=('standard','buffered','cuda_graph'),default='cuda_graph')
    parser.add_argument('--seed',type=sampling_seed,default=42,
                        help='Planner seed: integer or random for fresh noise each launch; N advances the seed')
    parser.add_argument('--seed-list',type=int,nargs='+',help='Cycle these planner seeds with N, or run them headless')
    parser.add_argument('--standing-hold',action='store_true',help='Experimental Y key: bypass planner with encoded standing intent; reset to resume recovery')
    parser.add_argument('--standing-hold-at',type=float,help='Simulation experiment: request standing intent at this time (seconds)')
    parser.add_argument('--standing-hold-on-upright',action='store_true',help='Simulation experiment: request standing intent after .3 s continuously upright')
    parser.add_argument('--standing-hold-frame',choices=('local','world'),default='local',help='local needs no XYZ feedback; world uses simulator pelvis position to restore the captured location')
    parser.add_argument('--standing-hold-blend-s',type=float,default=.5,help='Smooth desired pose transition duration, seconds')
    parser.add_argument('--start-state',type=Path,help='Exact resting MuJoCo snapshot; flat terrain, N changes only noise')
    parser.add_argument('--recorded-start',type=Path,help='Imported recorded reset with its exact terrain tile; N changes only noise')
    parser.add_argument('--start-pose-jitter-deg',type=float,default=0.,help='Perturb saved joint targets, position and settle before control; N samples another pose, R repeats')
    parser.add_argument('--start-pose-seed',type=sampling_seed,help='Pose jitter RNG seed, independent of planner noise; default random when jitter is enabled')
    parser.add_argument('--noise-only',action='store_true',help='N changes only planner noise; preserve pose and sensor seed')
    parser.add_argument('--pose-seed',type=int,help='Initial posture/joint seed; N advances it and samples a nearby start')
    parser.add_argument('--sensor-seed',type=int,help='Fix camera delivery randomness independently of planner seed')
    parser.add_argument('--depth-startup',choices=('prefill','cold','stationary'),default='prefill',
                        help='Default: acquire delivered depth history at rest before control, matching Stanley; cold/stationary retain older diagnostics')
    parser.add_argument('--tracker-startup',choices=('repeat','measured'),default='repeat',
                        help='Tracker history only: repeat the first measurement (saved recipe) or retain passive pre-start measurements')
    parser.add_argument('--tracker-preroll-s',type=float,
                        help='Passive pre-start physics, in 20 ms ticks; defaults to .2 with measured tracker history, otherwise 0')
    parser.add_argument('--command-limits',choices=('bounded','training'),default='bounded',
                        help='Bound extreme PD setpoints and target rates (default), or reproduce original training action/torque clips')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--viewer',action='store_true',help='Interactive MuJoCo window; requires a display')
    parser.add_argument('--fixed-camera',action='store_true',help='Keep the view at the starting location so pelvis translation stays visible')
    parser.add_argument('--video',action='store_true',help='Save an MP4 per run; supports headless EGL')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    import math
    if not math.isfinite(args.standing_hold_blend_s) or not 0<args.standing_hold_blend_s<=5:
        parser.error('--standing-hold-blend-s must be positive and at most 5 s')
    if args.standing_hold_at is not None and (not math.isfinite(args.standing_hold_at) or args.standing_hold_at<.02):
        parser.error('--standing-hold-at must be finite and at least .02 s')
    if args.standing_hold_at is not None or args.standing_hold_on_upright:args.standing_hold=True
    if args.guidance_scale is not None and (not args.goal_json or not math.isfinite(args.guidance_scale) or args.guidance_scale<=0):
        parser.error('--guidance-scale requires --goal-json and a finite positive value')
    if not math.isfinite(args.start_pose_jitter_deg) or args.start_pose_jitter_deg<0:
        parser.error('--start-pose-jitter-deg must be finite and nonnegative')
    if (args.start_pose_jitter_deg or args.start_pose_seed is not None) and not args.start_state:
        parser.error('Saved-pose jitter requires --start-state')
    if args.start_pose_jitter_deg and args.start_pose_seed is None:args.start_pose_seed=sampling_seed('random')
    if args.seed_list:
        if any(s<0 or s>=2**63 for s in args.seed_list):parser.error('Seeds must be in [0, 2**63)')
        if len(set(args.seed_list))!=len(args.seed_list):parser.error('--seed-list requires distinct seeds')
        args.seed=args.seed_list[0];args.episodes=len(args.seed_list)
    if args.start_state:
        if args.terrain!='flat' or args.postures or args.joint_jitter or args.tracker_preroll_s:
            parser.error('--start-state fixes a flat resting state; no posture sweep, jitter or extra physics preroll')
        args.noise_only=True
        args.start_snapshot=json.loads(args.start_state.expanduser().resolve().read_text())
    if args.recorded_start:
        if args.start_state or args.postures or args.joint_jitter or args.tracker_preroll_s or args.tracker_startup!='repeat':
            parser.error('--recorded-start preserves the recorded reset; no other reset or pre-physics options')
        from motion_diffusion.inference.bundle import sha256
        args.recorded_snapshot=json.loads(args.recorded_start.resolve().read_text())
        if args.recorded_snapshot.get('schema')!='hv2.recorded_sim_start.v1':parser.error('Invalid recorded reset schema')
        terrain=args.recorded_snapshot['terrain']
        if 'geometry' in terrain:
            path=Path(terrain['geometry'])
            args.terrain_geometry=recovery_asset(path)
            if sha256(args.terrain_geometry)!=terrain['sha256']:parser.error('Measured terrain hash mismatch')
            args.terrain='flat'
        else:
            path=Path(terrain['cache'])
            args.terrain_cache=recovery_asset(path)
            if sha256(args.terrain_cache)!=terrain['sha256']:parser.error('Recorded terrain hash mismatch')
            args.terrain='fork3';args.terrain_seed=terrain['seed']
        args.noise_only=True
    if args.terrain_geometry:
        if args.terrain!='flat':parser.error('--terrain-geometry replaces flat terrain; do not combine with another terrain')
        from motion_diffusion.inference.measured_scene import load_geometry
        args.measured_geometry=load_geometry(args.terrain_geometry)
    import math
    if args.seconds is None:args.seconds=None if args.viewer else 10.
    elif not math.isfinite(args.seconds) or args.seconds<0:parser.error('--seconds must be finite and nonnegative')
    elif args.seconds==0:args.seconds=None
    if args.episodes<1:parser.error('--episodes must be positive')
    if args.tracker_preroll_s is None:args.tracker_preroll_s=.2 if args.tracker_startup=='measured' else 0.
    if args.start_state and args.tracker_preroll_s:parser.error('--start-state cannot advance physics before the policy')
    if not math.isfinite(args.tracker_preroll_s) or args.tracker_preroll_s<0:
        parser.error('--tracker-preroll-s must be finite and nonnegative')
    if args.tracker_preroll_s or args.tracker_startup=='measured':
        if args.tracker_preroll_s<.18 or abs(round(args.tracker_preroll_s/.02)*.02-args.tracker_preroll_s)>1.e-8:
            parser.error('Tracker prehistory requires at least .18 s in whole 20 ms ticks')
    if not 0<=args.joint_jitter<=.3:parser.error('--joint-jitter must be between 0 and .3 radians')
    if not 0<=args.n_joint_jitter<=.3:parser.error('--n-joint-jitter must be between 0 and .3 radians')
    if not all(math.isfinite(x) and x>=0 for x in (args.n_xy_m,args.n_tilt_deg,args.n_yaw_deg)):
        parser.error('N offset ranges must be finite and nonnegative')
    if not all(math.isfinite(x) for x in (args.step_height,args.step_edge,args.step_length,args.step_width)):
        parser.error('Step dimensions must be finite')
    if min(args.step_height,args.step_length,args.step_width)<=0:parser.error('Step dimensions must be positive')
    if args.planner_precision!='fp32' and args.backend!='tensorrt':parser.error('--planner-precision tf32 requires explicit --backend tensorrt')
    if args.viewer and args.postures:parser.error('Use posture keys in the viewer; --postures is a headless sweep')
    if args.viewer and not os.environ.get('DISPLAY') and sys.platform.startswith('linux'):
        parser.error('--viewer needs DISPLAY; use --video for a headless SSH session')
    return args


class ViewerControls:
    """The GUI thread only queues events; the control thread owns resets."""
    def __init__(self):self.events=queue.SimpleQueue()
    def key(self,keycode):
        if keycode in (ord('R'),ord('N'),ord('M'),ord('P'),ord('Y'),32,259,*range(ord('1'),ord('8'))):
            self.events.put(keycode)
    def poll(self):
        try:return self.events.get_nowait()
        except queue.Empty:return None


def sample_start_offset(seed,*,xy_m=.1,tilt_deg=5.,yaw_deg=15.):
    """Sample around the original start, independently of planner/camera RNG."""
    import numpy as np
    rng=np.random.default_rng([int(seed),7143])
    return dict(root_xy=rng.uniform(-xy_m,xy_m,2).tolist(),
                root_rpy=np.deg2rad(rng.uniform([-tilt_deg,-tilt_deg,-yaw_deg],[tilt_deg,tilt_deg,yaw_deg])).tolist())


def advance_steering_release(planner, gate, observation, elapsed_s, context, status, *, standing=False):
    """Apply shared release criteria to synchronous scheduling; Y owns its bypass."""
    if standing or context is None:
        return status
    from motion_diffusion.inference.steering import command_finished
    release = gate.update(observation) if gate is not None else None
    if release is None and command_finished(planner.guidance_contract, elapsed_s):
        release = dict(active=False, reason='command_window_elapsed')
    if release is not None and (status is None or status['reason'] != release['reason']):
        status = dict(release, release_elapsed_s=elapsed_s)
        planner.request_replan()
    context['steering_released'] = context['elapsed_s'].new_full((1,), float(status is not None))
    return status


def run_episode(policy,plant,capture,args,output,seed,viewer=None,controls=None):
    import mujoco
    import numpy as np
    import torch
    from humanoidverse2.agents.action_vae.heading import initialize_heading,transport_heading
    from motion_diffusion.inference.mujoco import WarpDepth
    pose_seed=getattr(args,'pose_seed',None)
    sensor_seed=getattr(args,'sensor_seed',None)
    pose_seed=seed if pose_seed is None else pose_seed
    sensor_seed=seed if sensor_seed is None else sensor_seed
    with viewer.lock() if viewer else nullcontext():
        if getattr(args,'recorded_snapshot',None) is not None:
            from motion_diffusion.inference.start_state import restore_recorded_start
            restore_recorded_start(plant,args.recorded_snapshot)
        elif getattr(args,'start_snapshot',None) is not None:
            from motion_diffusion.inference.start_state import restore_start, prepare_jittered_start
            if getattr(args,'start_pose_jitter_deg',0.):
                prepared=prepare_jittered_start(plant,args.start_snapshot,args.start_pose_jitter_deg,args.start_pose_seed)
                (output/'prepared_start.json').write_text(json.dumps(prepared,indent=2)+'\n')
                print(f'  Pose jitter: +/- {args.start_pose_jitter_deg:g} deg, pose seed {args.start_pose_seed}; '
                      'joint positioning finished before camera history and policy reset.',flush=True)
            else:restore_start(plant,args.start_snapshot)
        else:
            plant.reset(args.start,seed=pose_seed,joint_jitter=max(args.joint_jitter,getattr(args,'start_jitter',0.)),
                        **getattr(args,'start_offset',{}))
        initial_tracker_history=None
        if getattr(args,'tracker_preroll_s',0.):
            from motion_diffusion.inference.mujoco import passive_tracker_history
            prehistory=passive_tracker_history(plant,args.tracker_preroll_s)
            np.savez_compressed(output/'tracker_prehistory.npz',**{k:v.numpy() for k,v in prehistory.items()})
            if getattr(args,'tracker_startup','repeat')=='measured':initial_tracker_history=prehistory
        fixed_lookat=plant.data.xpos[plant.pelvis].copy();fixed_lookat[2]=max(.35,fixed_lookat[2])
        if viewer and getattr(args,'fixed_camera',False):viewer.cam.lookat[:]=fixed_lookat
    if viewer:viewer.sync()
    device=policy.controller.device
    sensor=WarpDepth(plant,capture,sensor_seed,device=device) if capture else None
    if sensor is not None and getattr(args,'depth_startup','prefill')=='prefill':
        receipt=sensor.prefill_initial_pose(dict(planner=policy.planner.model.behavioral_config.get('depth_condition',{}),
                                                tracker=policy.controller.depth_config))
        plant.reset_info['depth_startup']=receipt
        print(f'  Depth startup: delivered camera history ready after {receipt["wait_s"]:.2f}s of sensor time; no physics or policy steps.',flush=True)
    if getattr(args,'depth_startup','cold')=='stationary':
        if sensor is None:raise ValueError('--depth-startup stationary requires a depth-conditioned model')
        sensor.prime_stationary()
        print('  Depth startup: stationary camera prehistory; physics and policy still start at tick zero.',flush=True)
    print(f'  Reset: { {k:v for k,v in plant.reset_info.items() if k != "joint_randomization"} }',flush=True)
    device=policy.controller.device
    previous=torch.zeros(1,29,device=device);code=torch.zeros(1,policy.controller.latent_dim,device=device)
    heading=previous_quat=None
    stand=getattr(policy,'standing_reference',None)
    if stand is not None:stand.reset()
    hold_requested=False;standing_ticks=0
    from motion_diffusion.planner.progress_steering import ProgressSteering
    from motion_diffusion.inference.steering import configured_upright_release
    upright_release = configured_upright_release(policy.planner.guidance_contract, plant.names)
    steering_release_status = None
    progress_goal=policy.planner.guidance_contract.get('goal',{})
    progress_steering=(ProgressSteering(progress_goal)
        if progress_goal.get('target',{}).get('path_progress') else None)
    steering_origin=plant.data.xpos[plant.pelvis].copy();steering_heading=None
    frames=[];replans=[];plans=[];status='duration_reached';limits_announced=False
    report_raw_max=0.;report_issued_max=0.;report_limit_ticks=0;report_ticks=0
    end_tick=None if args.seconds is None else round(args.seconds/.02)
    began=time.monotonic()
    reset_help=('R repeat | N new joint pose + noise' if getattr(args,'start_pose_jitter_deg',0.) else
                'R repeat | N next noise (fixed start)' if getattr(args,'noise_only',False) else
                'R repeat | N seed + nearby start | M terrain | P model | 1-7 posture')
    with ExitStack() as stack:
        renderer=writer=None
        camera=mujoco.MjvCamera();camera.distance=2.5;camera.azimuth=135;camera.elevation=-18
        if getattr(args,'fixed_camera',False):camera.lookat[:]=fixed_lookat
        if viewer:
            viewer.set_texts((None,None,f'{getattr(args,"terrain_label",getattr(args,"terrain","flat"))} | {args.start} | policy seed {seed}\n'+
                reset_help+' | SPACE pause\n'+
                ('Y encoded standing hold (experimental); R resets\n' if stand is not None else '')+
                ('' if getattr(args,'noise_only',False) else
                 '1 stand  2 front  3 back  4 left  5 right  6 seated  7 random\n')+
                'Depth: dark=near | light=far | purple=invalid/missing (including <10 cm)',None))
        if args.video:
            import imageio.v2 as imageio
            with viewer.lock() if viewer else nullcontext():
                plant.model.vis.global_.offwidth=960;plant.model.vis.global_.offheight=720
            renderer=mujoco.Renderer(plant.model,height=720,width=960);stack.callback(renderer.close)
            writer=imageio.get_writer(str(output/'rollout.mp4'),fps=50,codec='libx264',quality=7)
            stack.callback(writer.close)
        render_options=mujoco.MjvOption();render_options.geomgroup[:]=[1,1,0,0,0,0]
        try:
            tick=0;paused=False;halted=False;end_announced=False
            while viewer or end_tick is None or tick<end_tick:
                if viewer and not viewer.is_running():status='viewer_closed';break
                # A reset through MuJoCo's built-in UI also resets policy history.
                # MuJoCo can rewind time when bad acceleration resets its state.
                # Once halted on that warning, keep the diagnostic visible until
                # an operator reset; do not mistake it for a UI reset and loop.
                if not halted and abs(float(plant.data.time)-tick*.02)>1e-6:
                    status='reset_requested';break
                event=controls.poll() if controls else None
                if event==32:paused=not paused
                elif event==ord('Y'):
                    if stand is not None:hold_requested=True
                    else:print('Restart with --standing-hold to enable experimental Y.',flush=True)
                elif event is not None:
                    if (getattr(args,'start_snapshot',None) is not None or getattr(args,'recorded_snapshot',None) is not None) and (event==ord('M') or ord('1')<=event<=ord('7')):
                        print('Saved start: use the launcher to select another posture. '+reset_help,flush=True)
                        continue
                    status=('next_model' if event==ord('P') else 'next_terrain' if event==ord('M') else 'reset_requested')
                    if event==ord('N'):
                        sequence=getattr(args,'seed_list',None)
                        args.next_seed=sequence[(sequence.index(seed)+1)%len(sequence)] if sequence else seed+1
                        if getattr(args,'start_pose_jitter_deg',0.):
                            args.start_pose_seed=(args.start_pose_seed+1)%(2**63)
                        if not getattr(args,'noise_only',False):
                            args.pose_seed=pose_seed+1
                            args.start_offset=sample_start_offset(args.pose_seed,xy_m=getattr(args,'n_xy_m',.1),
                                tilt_deg=getattr(args,'n_tilt_deg',5.),yaw_deg=getattr(args,'n_yaw_deg',15.))
                            args.start_jitter=getattr(args,'n_joint_jitter',.15)
                    if ord('1')<=event<=ord('7'):
                        from motion_diffusion.inference.mujoco import START_POSTURES
                        args.start=START_POSTURES[event-ord('1')]
                    break
                if viewer and (paused or halted or (end_tick is not None and tick>=end_tick)):
                    if not paused and not end_announced:
                        viewer.set_texts((None,None,f'Episode stopped ({status}): {args.start} | seed {seed}\n'+
                                          reset_help,None))
                        end_announced=True
                    viewer.sync();time.sleep(.01);continue
                wall_start=time.monotonic()
                obs=plant.observation()
                quat=torch.tensor(obs.quat_xyzw,device=device).reshape(1,4)
                heading=initialize_heading(quat) if heading is None else transport_heading(heading,previous_quat,quat)
                previous_quat=quat.clone()
                guidance_context=None
                if getattr(args,'goal_json',None):
                    from humanoidverse2.agents.action_vae.heading import SCHEMA,heading_quat
                    from humanoidverse2.agents.action_vae.motion_contract import quat_mul_xyzw,quat_conj_xyzw,rot6d_from_quat_xyzw
                    if steering_heading is None:steering_heading=heading.clone()
                    delta=torch.as_tensor(plant.data.xpos[plant.pelvis]-steering_origin,device=device,dtype=torch.float32)
                    c,s=steering_heading.cos(),steering_heading.sin()
                    position=torch.stack((c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1],delta[2].expand_as(c)),-1)
                    guidance_context=dict(heading_schema=SCHEMA,elapsed_s=torch.tensor([tick*.02],device=device),
                        current_yaw_rel_rad=heading-steering_heading,current_position_reset=position,
                        current_height_rel_m=position[:,2])
                    rot6=rot6d_from_quat_xyzw(quat_mul_xyzw(quat_conj_xyzw(heading_quat(heading)),quat))
                    guidance_context.update({f'current_rot6d_{i}':rot6[:,i] for i in range(6)})
                    if progress_steering is not None:
                        was_released=progress_steering.released
                        guidance_context.update(progress_steering.update(quat,position,
                            orientation_stamp=tick*.02,position_stamp=tick*.02))
                        if progress_steering.released and not was_released:
                            print('Requested side roll confirmed; next replan is unguided.',flush=True)
                    if viewer and tick==0:
                        from motion_diffusion.inference.steering import goal_display_segments
                        segments=goal_display_segments(policy.planner.guidance_contract['goal'])
                        ch,sh=float(c[0]),float(s[0])
                        segments=segments@np.array([[ch,sh,0],[-sh,ch,0],[0,0,1]])+steering_origin
                        # Keep a fixed-world goal visible; no camera following.
                        with viewer.lock():
                            scene=viewer.user_scn;scene.ngeom=0
                            for a,b in segments:
                                if scene.ngeom>=scene.maxgeom:break
                                geom=scene.geoms[scene.ngeom]
                                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.9,.1,.8,.8]))
                                mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.006,a,b)
                                scene.ngeom+=1
                sensor_args={};availability=[]
                if sensor:
                    sensor.capture_tick(tick,obs)
                    if tick==0:
                        camera_info=sensor.startup_info
                        plant.reset_info['camera']=camera_info
                        from hv2_common.sensors.depth import is_v3
                        semantics=('Near pixels are masked; healthy empty rays saturate at 2 m.'
                                   if is_v3(capture) else 'These pixels stay masked.')
                        print(f'  Camera: {100*camera_info["camera_clearance_above_terrain_m"]:.1f} cm above terrain; '
                              f'{100*camera_info["too_near_fraction"]:.0f}% too close '
                              f'(<{100*camera_info["min_valid_depth_m"]:g} cm), '
                              f'{100*camera_info["no_hit_fraction"]:.0f}% no return. {semantics}',flush=True)
                    for prefix,config in (('',policy.planner.model.behavioral_config.get('depth_condition',{})),
                                          ('tracker_',policy.controller.depth_config)):
                        if not config.get('enabled'):continue
                        depth=sensor.prepare(tick,config,device,heading)
                        sensor_args.update({prefix+'depth_'+key:value for key,value in depth.items()})
                        if not prefix:
                            availability=depth['available'][0].cpu().numpy()
                            if viewer and getattr(args,'depth_view',True):
                                from motion_diffusion.inference.mujoco_terrain import depth_panel
                                panel=depth_panel(depth['frames'][0].detach().cpu().numpy(),availability,config['max_depth_m'])
                                viewer.set_images((mujoco.MjrRect(12,12,panel.shape[1],panel.shape[0]),panel))
                override=None
                if stand is not None:
                    position=torch.as_tensor(plant.data.xpos[plant.pelvis],device=device,dtype=torch.float32)[None]
                    joints=torch.as_tensor(obs.q,device=device,dtype=torch.float32)[None]
                    knees=[plant.names.index(side+'_knee_joint') for side in ('left','right')]
                    hips=[plant.names.index(side+'_hip_pitch_joint') for side in ('left','right')]
                    upright=(-obs.gravity_body[2]>=.92 and abs(obs.q[knees]).max()<=1.
                             and abs(obs.q[hips]).max()<=.7
                             and position[0,2].item()-plant.terrain_height(position[0,:2].cpu().numpy())>=.55)
                    standing_ticks=standing_ticks+1 if upright else 0
                    if args.standing_hold_at is not None and tick*.02>=args.standing_hold_at:hold_requested=True
                    if args.standing_hold_on_upright and standing_ticks>=15:hold_requested=True
                    if hold_requested and not stand.active:
                        stand.activate(tick*.02,joints,quat,position,heading,code)
                        print(f'  Standing intent enabled at {tick*.02:.2f}s ({stand.frame}); planner bypassed, tracker history retained.',flush=True)
                        if viewer:viewer.set_texts((None,None,
                            f'ENCODED STANDING HOLD ({stand.frame}) | tracker remains active\nR reset | N next seed | SPACE pause',None))
                    if stand.active:override=stand(tick*.02,position,heading)
                steering_release_status = advance_steering_release(
                    policy.planner, upright_release, obs, tick*.02, guidance_context,
                    steering_release_status, standing=override is not None)
                result=policy.step(obs.q[None],obs.dq[None],obs.gyro_body[None],obs.gravity_body[None],previous,
                    pelvis_quat=quat,previous_executed_code=code,reset_mask=torch.tensor([tick==0],device=device),
                    control_tick=torch.tensor([tick],device=device),episode_seeds=[seed],
                    guidance_context=guidance_context,
                    native_code_override=override,
                    **({'initial_tracker_history':initial_tracker_history} if tick==0 and initial_tracker_history is not None else {}),
                    **sensor_args)
                raw_max=float(policy.controller.last_unclipped_action.abs().max())
                action=result['requested_action'][0].cpu().numpy()
                replanned=bool(result['replanned'][0])
                if replanned:
                    replans.append(tick)
                    plan=dict(tick=tick,state=result['predicted_state'][0].cpu().numpy().tolist())
                    if hasattr(policy,'planner'):
                        plan['native_code']=policy.planner.last_predicted_codes[0].cpu().numpy().tolist()
                    plans.append(plan)
                if tick==0 and hasattr(policy,'planner') and override is None:
                    # Freeze the actual condition/noise used by the first plan,
                    # so comparisons can prove that only sampling noise changed.
                    audit={f'history_{key}':value.detach().cpu().numpy().copy()
                           for key,value in policy.planner.hist.items()}
                    audit.update({key:value.detach().cpu().numpy().copy() for key,value in sensor_args.items()})
                    audit.update(state_noise=policy.planner.state_noise.detach().cpu().numpy().copy(),
                        code_noise=policy.planner.action_noise.detach().cpu().numpy().copy(),
                        initial_qpos=plant.data.qpos.copy(),
                        initial_qvel=plant.data.qvel.copy())
                    if getattr(policy,'history',None) is not None:
                        audit['tracker_history']=policy.history.flatten().detach().cpu().numpy().copy()
                    np.savez_compressed(output/'first_condition.npz',**audit)
                # Continue the policy after limiting a finite command; feedback
                # must contain what the plant actually received.
                qpos=plant.data.qpos.copy();root=plant.data.xpos[plant.pelvis].copy()
                terrain_z=plant.terrain_height(root[:2])
                with viewer.lock() if viewer else nullcontext():
                    issued,target,torque,clipped=plant.step(action)
                command_limit=getattr(plant,'last_command_limit',None)
                position_limited=0 if command_limit is None else int(command_limit['position_limited'].sum())
                rate_limited=0 if command_limit is None else int(command_limit['rate_limited'].sum())
                limited=position_limited>0 or rate_limited>0
                if limited and not limits_announced:
                    changed=np.flatnonzero(command_limit['position_limited']|command_limit['rate_limited'])
                    details='; '.join(f'{plant.names[i]} {command_limit["requested_target"][i]:+.2f}->{target[i]:+.2f} rad'
                                     for i in changed)
                    print(f'  Command targets limited; policy continues: {details}',flush=True)
                    limits_announced=True
                report_raw_max=max(report_raw_max,raw_max);report_limit_ticks+=int(limited);report_ticks+=1
                report_issued_max=max(report_issued_max,float(np.max(np.abs(issued))))
                previous=torch.tensor(issued,device=device).reshape(1,29)
                code=result['native_code'].detach()
                frames.append(dict(time=tick*.02,q=obs.q,dq=obs.dq,gravity=obs.gravity_body,
                    standing_hold=stand is not None and stand.active,
                    pelvis_quat_xyzw=obs.quat_xyzw.copy(),causal_heading_rad=float(heading[0]),
                    qpos=qpos,root_position=root,terrain_height_m=terrain_z,
                    action=issued,target=target,native_code=code[0].cpu().numpy(),
                    raw_action_max=raw_max,saturated_joints=int((np.abs(action)>=plant.clip).sum()),
                    command_position_limited_joints=position_limited,command_rate_limited_joints=rate_limited,
                    torque_peak=torque,torque_clipped=clipped,depth_available=np.asarray(availability,dtype=bool),
                    wall_ms=(time.monotonic()-wall_start)*1000))
                lookat=plant.data.xpos[plant.pelvis].copy();lookat[2]=max(.35,lookat[2])
                if renderer:
                    if not getattr(args,'fixed_camera',False):camera.lookat[:]=lookat
                    renderer.update_scene(plant.data,camera=camera,scene_option=render_options)
                    writer.append_data(renderer.render())
                if viewer:
                    if not getattr(args,'fixed_camera',False):
                        with viewer.lock():viewer.cam.lookat[:]=lookat
                    viewer.sync()
                    remaining=.02-(time.monotonic()-wall_start)
                    if remaining>0:time.sleep(remaining)
                if tick%50==0:
                    print(f'  t={tick*.02:5.1f}s height_above_terrain={root[2]-terrain_z:.3f}m upright={-obs.gravity_body[2]:.3f} '
                          f'raw_action_max={report_raw_max:.2f} issued_action_max={report_issued_max:.2f} '
                          f'limited_ticks={report_limit_ticks}/{report_ticks} plans={len(replans)}',flush=True)
                    report_raw_max=0.;report_issued_max=0.;report_limit_ticks=0;report_ticks=0
                if any(int(w.number)>0 for w in plant.data.warning):
                    status='physics_warning'
                    if viewer:halted=True
                    else:break
                tick+=1
        except KeyboardInterrupt:
            status='interrupted'
        except Exception:
            status='failed'
            raise
        finally:
            if frames:
                arrays={key:np.asarray([f[key] for f in frames]) for key in frames[0]}
                arrays['final_qpos']=plant.data.qpos.copy()
                np.savez_compressed(output/'trajectory.npz',**arrays)
                # Sensor-only archive; consumes no goals and changes no planner behavior.
                from steerable_recovery.hardware.measured_pose import simulation_packet
                with (output/'measured_pose.jsonl').open('w') as stream:
                    for frame in frames:
                        packet=simulation_packet(frame['root_position'],frame['pelvis_quat_xyzw'].tolist(),frame['time'])
                        packet.update(imu_quat_xyzw=frame['pelvis_quat_xyzw'].tolist(),
                            geometry_valid=True,tracking_valid=False,offline_replay=True,clock_domain='simulation',
                            yaw_disagreement_rad=0.,tilt_disagreement_rad=0.)
                        stream.write(json.dumps(packet)+'\n')
            upright=[f['root_position'][2]-f['terrain_height_m']>=.55 and -f['gravity'][2]>=.75 for f in frames]
            held=0;first_hold=None
            for i,good in enumerate(upright):
                held=held+1 if good else 0
                if held>=50 and first_hold is None:first_hold=(i+1)*.02
            summary=dict(status=status,reset=plant.reset_info,planner_seed=seed,sensor_seed=sensor_seed,
                standing_hold_frame=None if stand is None else stand.frame,
                standing_hold_start_s=None if stand is None else stand.start_time,
                steering_progress=None if progress_steering is None else progress_steering.diagnostics(),
                steering_release=steering_release_status,
                upright_steering_release=None if upright_release is None else upright_release.contract,
                tracker_startup=getattr(args,'tracker_startup','repeat'),tracker_preroll_s=getattr(args,'tracker_preroll_s',0.),
                depth_startup=getattr(args,'depth_startup','cold'),
                frames=len(frames),simulated_seconds=len(frames)*.02,
                upright_hold_1s=first_hold is not None,first_upright_hold_s=first_hold,
                final_second_upright_fraction=float(np.mean(upright[-50:])) if upright else None,
                upright_metric='pelvis height above terrain under pelvis XY >= .55 m and upright cosine >= .75 for 50 control ticks; diagnostic, not training success',
                wall_seconds=time.monotonic()-began,replan_ticks=replans,
                min_pelvis_height_m=min((float(f['root_position'][2]) for f in frames),default=None),
                final_pelvis_height_m=float(plant.data.xpos[plant.pelvis,2]),
                final_pelvis_height_above_terrain_m=float(plant.data.xpos[plant.pelvis,2]-plant.terrain_height(plant.data.xpos[plant.pelvis,:2])),
                saturated_action_ticks=sum(f['saturated_joints']>0 for f in frames),
                command_limit_mode=getattr(plant,'command_limit_mode','training'),
                command_limited_ticks=sum(f['command_position_limited_joints']>0 or f['command_rate_limited_joints']>0 for f in frames),
                max_issued_action=max((float(np.max(np.abs(f['action']))) for f in frames),default=None),
                max_raw_action=max((f['raw_action_max'] for f in frames),default=None),
                final_upright_cos=float(-plant.observation().gravity_body[2]),
                physics_warnings={str(mujoco.mjtWarning(i)):int(w.number) for i,w in enumerate(plant.data.warning) if w.number})
            if frames and getattr(args,'goal_json',None):
                from motion_diffusion.planner.pose_waypoints import waypoint_tracking_metrics
                from motion_diffusion.features.rotation import quat_to_matrix
                from humanoidverse2.agents.action_vae.heading import heading_matrix
                goal=policy.planner.guidance_contract['goal']
                h0=heading_matrix(torch.tensor(frames[0]['causal_heading_rad']))
                p=torch.tensor(np.asarray([f['root_position'] for f in frames]),dtype=torch.float32)
                relative=(p-torch.tensor(steering_origin,dtype=p.dtype))@h0
                if progress_steering is not None:
                    summary['steering']=dict(method='progress_xyz',**progress_steering.diagnostics(),
                        note='Reference timestamps are geometric labels; no timed-path error metric.')
                elif goal['target'].get('pose_waypoints'):
                    q=torch.tensor(np.asarray([f['pelvis_quat_xyzw'] for f in frames]),dtype=torch.float32)
                    summary['steering']=waypoint_tracking_metrics(relative,h0.T@quat_to_matrix(q),goal['target']['pose_waypoints'])
                elif goal['target'].get('axis_waypoints'):
                    from motion_diffusion.planner.axis_waypoints import interpolate_axis_waypoints
                    spec=goal['target']['axis_waypoints'];axis=0 if spec['axis']=='x' else 1
                    times=torch.tensor([f['time'] for f in frames]);end=goal.get('execution',{}).get('release_after_s',float('inf'))
                    desired=interpolate_axis_waypoints(spec,times)
                    excess=((relative[:,axis]-desired).abs()-spec['deadband_m']).clamp_min(0)
                    index=int((times-spec['times_s'][-1]).abs().argmin())
                    summary['steering']=dict(axis=spec['axis'],target_m=spec['values_m'][-1],deadband_m=spec['deadband_m'],
                        at_target_time_s=float(times[index]),at_target_time_m=float(relative[index,axis]),
                        active_excess_rmse_m=float(excess[times<end].square().mean().sqrt()),
                        final_relative_xyz_m=relative[-1].tolist(),release_after_s=end)
                if 'steering' in summary:summary['steering']['feedback']='simulator truth; hardware estimator not exercised'
                (output/'steering_goal.json').write_text(json.dumps(goal,indent=2)+'\n')
            (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            (output/'plans.json').write_text(json.dumps(plans)+'\n')
    return summary


def main(argv=None):
    args=arguments(argv)
    # N varies diffusion noise independently of pose and camera delivery.
    if args.pose_seed is None:args.pose_seed=args.seed
    args.start_offset={};args.start_jitter=0.
    if args.sensor_seed is None:args.sensor_seed=args.seed
    if args.video and not args.viewer:os.environ.setdefault('MUJOCO_GL','egl')
    import torch
    import yaml
    from omegaconf import OmegaConf
    from motion_diffusion.inference.policy import NativeRecoveryPolicy
    from motion_diffusion.inference.policy import load_controller
    from motion_diffusion.inference.steering import load_hardware_planner
    from motion_diffusion.inference.bundle import load_bundle, asset_path, replan_ticks_for_hz
    from motion_diffusion.inference.export_pd import derive_pd
    from motion_diffusion.inference.engines import configure_controller, configure_planner
    from motion_diffusion.inference.mujoco import DEFAULT_ROBOT, scene_model, MujocoNativePlant
    torch.set_num_threads(1)
    config=yaml.safe_load((CONFIG_ROOT / 'models.yaml').read_text())
    from steerable_recovery.settings import model_settings
    output=(args.output or ROOT/'logs/native_recovery'/f'mujoco_{args.terrain}_{time.time_ns()}').resolve()
    output.mkdir(parents=True,exist_ok=False)
    print(f'Output: {output}',flush=True)
    runs={}
    for name in (args.model,):
        selected=model_settings(config,name)
        backend=args.backend or selected.get('default_backend','pytorch')
        bundle,manifest=load_bundle(args.bundle or selected['bundle'])
        if name not in manifest['models']:
            raise ValueError(f'Bundle does not contain {name}; supply its verified bundle with --bundle')
        row=manifest['models'][name]
        controller_path=asset_path(bundle,row['controller']);planner_path=asset_path(bundle,row['planner'])
        training=OmegaConf.load(asset_path(bundle,row['controller_training_config']))
        print(f'Loading {name} planner and tracker...',flush=True)
        controller=load_controller(dict(controller_checkpoint=str(controller_path),tracker_device=args.device,
            pd_sidecar=str(asset_path(bundle,row['pd_binding']))))
        inference=dict(row['reference_inference'])
        if args.goal_json:
            from motion_diffusion.inference.steering import with_guidance_scale
            inference['motion_goal']=with_guidance_scale(json.loads(args.goal_json.read_text()),args.guidance_scale)
        if args.noise_policy is not None:inference['noise_policy']=args.noise_policy
        ticks=replan_ticks_for_hz(args.replan_hz,horizon=row['inspected_contract']['F'])
        if ticks is not None:inference['replan_every_ticks']=ticks
        if args.blend_ticks is not None:
            if not 0 <= args.blend_ticks <= inference['replan_every_ticks']:
                raise ValueError('Blend must be between zero and the replan period')
            inference['blend_ticks']=args.blend_ticks
        planner=load_hardware_planner(planner_path,args.device,seed=args.seed,sampling_rng='paired_episode_v2',**inference)
        if backend!='pytorch':
            def engine(key,precision):
                if key not in selected['exports']:
                    raise ValueError(f'No compiled export for {name}/{key}; use --backend pytorch')
                directory=Path(selected['exports'][key])
                if not directory.is_absolute():directory=ROOT/directory
                return dict(directory=str(directory),backend=backend,precision=precision,execution_mode=args.engine_mode,
                            validation_mode='measure' if precision=='tf32' else 'strict')
            configure_controller(controller,engine('controller','fp32'))
            if not args.goal_json:configure_planner(planner,planner_path,engine(name,args.planner_precision))
        policy=NativeRecoveryPolicy(controller,planner)
        if args.standing_hold:
            from motion_diffusion.inference.standing import StandingReference
            policy.standing_reference=StandingReference(torch.load(controller_path,map_location='cpu',weights_only=True),
                controller,blend_s=args.standing_hold_blend_s,frame=args.standing_hold_frame)
        validation={key:engine.validation_report for key,engine in (
            ('denoiser',planner.model.inference_backend),
            ('depth',None if planner.model.depth_encoder is None else planner.model.depth_encoder.inference_backend))
            if engine is not None}
        if any(not v['passed'] for v in validation.values()):
            print('Experimental precision: neural parity tolerance exceeded; measuring closed-loop behavior.',flush=True)
        robot=OmegaConf.to_container(training.robot,resolve=True)
        pd=derive_pd(controller.contract,dict(robot=robot,simulator=dict(config=dict(sim=OmegaConf.to_container(training.simulator.config.sim,resolve=True)))))
        scene=dict(terrain=args.terrain,step_height=args.step_height,step_edge=args.step_edge,
                   step_length=args.step_length,step_width=args.step_width,
                   terrain_seed=args.terrain_seed,terrain_cache=str(args.terrain_cache.resolve()))
        if args.terrain_geometry:scene['terrain_geometry']=str(args.terrain_geometry.resolve())
        print('Preparing MuJoCo scene...',flush=True)
        model=scene_model(args.robot or DEFAULT_ROBOT,**scene)
        plant=MujocoNativePlant(model,controller.contract,pd,robot,command_limits=args.command_limits)
        print(f'Command limits: {args.command_limits}; raw_action_max reports the maximum since the previous line (50 control ticks/s).',flush=True)
        depth_configs=[c for c in (controller.depth_config,planner.model.behavioral_config.get('depth_condition',{})) if c.get('enabled')]
        capture=depth_configs[0]['capture'] if depth_configs else None
        if any(c['capture']!=capture for c in depth_configs):raise ValueError('Tracker/planner camera mounts differ')
        print(f'{name}: horizon={planner.model.F}, replan={planner.replan_every_ticks}, blend={planner.blend_ticks} ticks; '
              f'{inference["sampler"]}{inference["num_inference_steps"]}; {backend}; '
              f'{50/planner.replan_every_ticks:g} Hz planning; noise={planner.noise_policy}; start={args.start}',flush=True)
        if args.goal_json:
            print(f'Steering: {args.goal_json.name}; PyTorch planner / {backend} tracker; simulator pelvis truth, not measured odometry.',flush=True)
        if args.terrain=='step':print(f'Stair: height {args.step_height:g} m, edge X={args.step_edge:g} m, '
            f'tread {args.step_length:g} m, width {args.step_width:g} m; pelvis starts at XY=(0,0).',flush=True)
        viewer=None;controls=None
        def open_viewer(model,plant):
            import mujoco.viewer
            controls=ViewerControls()
            view=mujoco.viewer.launch_passive(model,plant.data,key_callback=controls.key)
            with view.lock():
                view.cam.distance=3.;view.cam.azimuth=135;view.cam.elevation=-25
                view.opt.geomgroup[:]=[1,1,0,0,0,0]
            return view,controls
        if args.viewer:
            viewer,controls=open_viewer(model,plant)
        episode=0;seed=args.seed;initial_start_pose_seed=args.start_pose_seed
        postures=args.postures or [args.start]
        while episode<args.episodes*len(postures) or viewer:
            if not viewer:
                args.start=postures[episode//args.episodes]
                seed=args.seed_list[episode%args.episodes] if args.seed_list else args.seed+episode%args.episodes
                if args.start_pose_jitter_deg:args.start_pose_seed=(initial_start_pose_seed+episode)%(2**63)
            directory=output/f'{name}_{episode:03d}';directory.mkdir()
            from motion_diffusion.inference.mujoco_terrain import SIM_TERRAINS, sample_terrain_mesh
            tile_info=sample_terrain_mesh(args.terrain,scene['terrain_seed'],args.terrain_cache)[2] if args.terrain in SIM_TERRAINS else None
            args.terrain_label=(f'{tile_info["family"]} tile {tile_info["tile"]} terrain seed {scene["terrain_seed"]}' if tile_info else args.terrain)
            if args.terrain_geometry:args.terrain_label=args.measured_geometry.get('name','measured lab')
            if tile_info:print(f'Terrain: {args.terrain_label}',flush=True)
            (directory/'config.json').write_text(json.dumps(dict(model=name,seed=seed,start=args.start,joint_jitter=args.joint_jitter,
                pose_seed=args.pose_seed,sensor_seed=args.sensor_seed,start_offset=args.start_offset,start_jitter=args.start_jitter,
                depth_startup=args.depth_startup,tracker_startup=args.tracker_startup,tracker_preroll_s=args.tracker_preroll_s,
                backend=backend,planner_precision=(args.planner_precision if backend=='tensorrt' else inference['numerical_precision']),engine_mode=args.engine_mode,
                camera_randomization=False,
                fixed_start_snapshot=getattr(args,'start_snapshot',None),noise_only=args.noise_only and not args.start_pose_jitter_deg,
                recorded_gallery_start=getattr(args,'recorded_snapshot',None),
                start_pose_jitter_deg=args.start_pose_jitter_deg,start_pose_seed=args.start_pose_seed,
                seed_list=args.seed_list,requested_seconds=args.seconds,fixed_camera=args.fixed_camera,
                standing_hold=dict(enabled=args.standing_hold,at_s=args.standing_hold_at,
                    on_upright=args.standing_hold_on_upright,frame=args.standing_hold_frame,
                    blend_s=args.standing_hold_blend_s,latent_handoff='quintic_from_last_executed_code',
                    controller_encoder='original_posterior_and_frozen_corpus_normalizer',
                    target='default_joints_zero_velocity_upright_captured_heading'),
                scene=scene,
                terrain_tile=tile_info,depth_backend='shared_training_warp',
                reference_inference=row['reference_inference'],requested_replan_hz=args.replan_hz,requested_blend_ticks=args.blend_ticks,
                actual_replan_hz=50/planner.replan_every_ticks,
                neural_validation=validation,
                robot=str((args.robot or DEFAULT_ROBOT).resolve()),bundle=str(bundle),
                controller_sha256=row['controller']['sha256'],planner_sha256=row['planner']['sha256'],
                inference=inference,sampling_rng='paired_episode_v2',pd=pd,camera=capture,
                effort_limits=plant.effort.tolist(),physics_dt=float(model.opt.timestep),control_dt=.02,
                command_limit_mode=args.command_limits,command_limits=None if plant.command_limiter is None else plant.command_limiter.contract(),
                timing='synchronous simulation clock; inference pauses physics',
                plant='SONIC MuJoCo asset, saved nominal PD/armature/friction/effort, terrain friction1; no domain randomization'),indent=2)+'\n')
            try:result=run_episode(policy,plant,capture,args,directory,seed,viewer,controls)
            except BaseException:
                if viewer:viewer.close()
                raise
            runs[directory.name]=result
            (output/'summary.json').write_text(json.dumps(runs,indent=2)+'\n')
            episode+=1
            seed=vars(args).pop('next_seed',seed if viewer else seed+1)
            if result['status']=='next_terrain':
                if args.recorded_start or args.terrain_geometry:
                    print('Recorded starts and measured geometry fix the terrain; provide another input to change it.',flush=True)
                    continue
                if viewer:viewer.close()
                if args.terrain not in SIM_TERRAINS:args.terrain='fork3'
                args.terrain_seed+=1
                scene.update(terrain=args.terrain,terrain_seed=args.terrain_seed)
                print('Loading next fork3 terrain; keeping model weights and policy seed.',flush=True)
                model=scene_model(args.robot or DEFAULT_ROBOT,**scene)
                plant=MujocoNativePlant(model,controller.contract,pd,robot,command_limits=args.command_limits)
                viewer,controls=open_viewer(model,plant)
                continue
            if result['status'] in ('interrupted','viewer_closed','next_model','physics_warning'):
                if viewer:viewer.close()
                if result['status']!='next_model':return
                break
        if viewer:viewer.close()
        del policy,planner,controller
        if torch.cuda.is_available():torch.cuda.empty_cache()
    print(f'Finished. Results: {output}',flush=True)


if __name__=='__main__':main()
