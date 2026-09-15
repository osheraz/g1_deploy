"""Diffusion recovery on G1: one config, default sensor-only inference.

python g1_deploy/deploy_diffusion.py
python g1_deploy/deploy_diffusion.py --check
python g1_deploy/deploy_diffusion.py --default-pose --actuate --release-controller
python g1_deploy/deploy_diffusion.py --actuate --release-controller
Motor mode: enter zero torque after robot connection. A starts policy from the current pose; A again stops.
Before starting only, START optionally moves to default over 2 s. SELECT stops.
With --start-pose prone, START slowly positions first; A then starts recovery.
Default-pose-only mode also waits for START, then holds until SELECT/A/Ctrl+C.
--actuate automatically releases the built-in controller for startup zero torque;
--release-controller remains a compatibility flag. Place the robot at rest or supported before launch.
"""
import argparse
import math
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'g1_deploy'))
sys.path.insert(0,str(ROOT/'dependencies/isaac_utils'))

from record_motion import DEFAULT_CAMERA, RESULTS_ROOT


def sampling_seed(value):
    if value == 'random':
        import secrets
        return secrets.randbits(63)
    try:
        seed=int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Seed must be an integer or random') from exc
    if not 0<=seed<2**63:
        raise argparse.ArgumentTypeError('Seed must be in [0, 2**63)')
    return seed


def model_settings(settings, model):
    """Resolve an explicit pair without changing the legacy diffusion/flow aliases."""
    from copy import deepcopy
    selected=deepcopy(settings)
    selected.update(selected.get('models',{}).get(model,{}))
    path=Path(selected['bundle'])
    selected['bundle']=str(path if path.is_absolute() else ROOT/path)
    return selected


def hardware_settings(config, contract):
    from motion_diffusion.behavioral.hardware.sensors import motor_indices
    indices=motor_indices(contract['dof_names'])
    configured=config['action_joint2motor_idx']
    if len(configured)!=29 or set(configured)!=set(range(29)):
        raise ValueError('Diffusion deployment requires all 29 physical joints')
    order=[configured.index(i) for i in indices]
    return dict(tracker_device=config['tracker_device'],planner_device=config['planner_device'],
        robot=dict(interface=config['network_interface'],domain_id=0,tick_period_s=.001,
            tick_clock=None,max_command_age_s=config['max_command_age_s'],require_remote_start=True),
        camera=config['camera'],pd=dict(training_pd=None,mode_machine=config['mode_machine'],
            lower_joint_limits=[config['dof_lower'][i] for i in order],
            upper_joint_limits=[config['dof_upper'][i] for i in order],
            max_joint_velocity_rad_s=config['max_joint_velocity'],
            max_abs_requested_action=contract['action_clip']))


def run_default_pose(config, model_contract, *, release_controller=None):
    from motion_diffusion.behavioral.hardware.unitree import UnitreeHG,NativePDContract,wait_for_policy_start
    contract=dict(model_contract)
    if contract.get('deployment_pd') not in (None,config['pd']['training_pd']):
        raise ValueError('Default-pose PD binding differs from the decoder')
    contract['deployment_pd']=config['pd']['training_pd']
    pd=NativePDContract(contract,config['pd']);settings=config['robot']
    robot=UnitreeHG(interface=settings['interface'],dof_names=contract['dof_names'],
        capture_clock=None,tick_period_s=settings['tick_period_s'],enable_actuation=False,
        pd_contract=pd,max_command_age_s=settings['max_command_age_s'],domain_id=settings['domain_id'])
    try:
        deadline=time.monotonic()+5.
        while True:
            try:robot.observation();break
            except RuntimeError:
                if robot.fault or time.monotonic()>deadline:raise
                time.sleep(.01)
        pd.check_startup_pose(robot.observation())
        released=False
        if config.get('startup_zero_torque',False):
            robot.arm_a_stop(after_press=float('inf'))
            if release_controller is not None:release_controller();released=True
            robot.enter_zero_torque()
            print('Zero torque active: tau=0, kp=0, kd=0. START: move to default.',flush=True)
        def activate():
            pd.check_startup_pose(robot.observation())
            if release_controller is not None and not released:release_controller()
            pd.check_startup_pose(robot.observation())
            robot.enable_motor_output()
        wait_for_policy_start(robot,pd,pose_only=True,activate=activate)
    finally:robot.close()


def main():
    from core.config import _load_config
    from motion_diffusion.behavioral.hardware import bundle
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'g1_deploy/configs/diffusion.yaml')
    parser.add_argument('--model',choices=['p11_diffusion','r8_f50r_flow','r8_s25_diffusion','r2_f50r_flow','p05_diffusion','diffusion','flow'],
        default='p11_diffusion')
    parser.add_argument('--replan-hz',type=float,help='Planner rate, e.g. 2.5, 5 or 10 Hz; tracker stays 50 Hz; existing asynchronous horizon/age checks apply')
    parser.add_argument('--noise-policy',choices=('fresh','fixed_episode'),default='fresh',
        help='fresh: new noise every plan (default); fixed_episode: sample once at policy start and reuse for replanning')
    parser.add_argument('--seed',type=sampling_seed,default=42,
        help='Planner sampling seed: integer or random. Saved in the session; default 42')
    parser.add_argument('--start-pose',help='prone or a prone/supine joint-preset JSON: START moves to it, then A starts recovery; requires --actuate')
    parser.add_argument('--start-pose-jitter-deg',type=float,default=0.,help='Random joint offsets up to +/- this many degrees before START positioning; default 0')
    parser.add_argument('--start-pose-seed',type=sampling_seed,help='Pose jitter seed, separate from planner --seed; default random when jitter is enabled')
    parser.add_argument('--check',action='store_true',help='Verify selected weights, PD binding and synthetic inference; no sensors or motors')
    parser.add_argument('--backend',choices=['pytorch','tensorrt','onnx'],help='Defaults to the selected model profile')
    parser.add_argument('--planner-precision',choices=['fp32','tf32'],default='fp32',
        help='TensorRT engine precision only; PyTorch uses the bundle precision (P11: TF32)')
    parser.add_argument('--engine-mode',choices=['standard','buffered','cuda_graph'],default='cuda_graph')
    parser.add_argument('--command-limits',choices=['bounded','training'],default='bounded',
        help='Bound PD target jumps from the measured start (default); training selects the original direct targets')
    parser.add_argument('--actuate',action='store_true')
    parser.add_argument('--preflight',action='store_true',help='Repeat live startup checks without motor output or takeover')
    parser.add_argument('--release-controller',action='store_true',
        help='Compatibility flag; --actuate releases the built-in service for startup zero torque automatically')
    parser.add_argument('--default-pose',action='store_true',
        help='Wait for START to move to default and hold; SELECT/A stops; requires --actuate; no policy')
    parser.add_argument('--duration-s',type=float,default=30.)
    parser.add_argument('--output',type=Path,help='Session directory; default: logs/real_world/robot_MODEL_BACKEND_TIMESTAMP')
    parser.add_argument('--record-sensors',action=argparse.BooleanOptionalAction,default=None,
        help='Record raw LiDAR + robot state; default for actuation/live odometry, opt-in for sensor-only inference')
    parser.add_argument('--record-depth',action=argparse.BooleanOptionalAction,default=None,
        help='Save original-resolution lossless depth for video and exact model replay; defaults on with model/sensor recording')
    parser.add_argument('--record-model',action=argparse.BooleanOptionalAction,default=True,
        help='Save exact planner/tracker inputs, RNG, outputs and code provenance for offline model replay (default on)')
    parser.add_argument('--existing-lidar-driver',action='store_true',help='Use an already-running Livox driver')
    parser.add_argument('--record-label',default='supported_hoist',help='Operator description stored with the capture')
    parser.add_argument('--external-camera',nargs='?',const=DEFAULT_CAMERA,
        help=f'Record a fixed external USB camera through a v4l2-ctl child of the sensor recorder (kernel-only MJPEG dump '
             f'with driver timestamps; the control process never touches it). No value selects {DEFAULT_CAMERA}')
    parser.add_argument('--camera-size',default='1920x1080',help='MJPG capture size; see check_camera.py --list-modes')
    parser.add_argument('--camera-fps',type=int,default=30,help='requested rate; the driver clamps to what the mode offers and the report shows the rate set')
    parser.add_argument('--live-odometry',action='store_true',help='Run DLIO alongside deployment and log live XYZ; depth remains the planner input')
    parser.add_argument('--odometry-port',type=int,default=15560)
    parser.add_argument('--odometry-mode',choices=['monitor','required'],default='monitor',
        help='Monitor labels unqualified estimates; required enforces the existing steering receiver contract')
    parser.add_argument('--odometry-mount',type=Path,default=ROOT/'g1_deploy/config/lidar/mount_candidate_20260912.json',
        help='Explicit diagnostic mount candidate used by live odometry')
    parser.add_argument('--odometry-clock',type=Path,help='Independent clock calibration JSON; never infer qualification from arrival jitter')
    args=parser.parse_args()
    if not math.isfinite(args.start_pose_jitter_deg) or args.start_pose_jitter_deg<0:
        parser.error('--start-pose-jitter-deg must be finite and nonnegative')
    if (args.start_pose_jitter_deg or args.start_pose_seed is not None) and not args.start_pose:
        parser.error('Pose jitter requires --start-pose')
    if args.start_pose and (not args.actuate or args.default_pose or args.check or args.preflight):
        parser.error('--start-pose requires --actuate and a normal recovery session')
    config_path=args.config.resolve()
    settings=model_settings(_load_config(config_path),args.model)
    if args.backend is None:args.backend=settings.get('default_backend','tensorrt')
    if args.check and (args.actuate or args.release_controller or args.default_pose or args.preflight or args.live_odometry):
        parser.error('--check does not open sensors or control the robot')
    if args.record_sensors is None:args.record_sensors=args.actuate or args.live_odometry
    if args.record_depth is None:args.record_depth=args.record_sensors or args.record_model
    if args.backend!='tensorrt' and args.planner_precision!='fp32':parser.error('TF32 engine precision requires TensorRT')
    if args.default_pose and not args.actuate:parser.error('--default-pose moves motors: add --actuate explicitly')
    if args.preflight and (args.actuate or args.default_pose):parser.error('--preflight is sensor-only')
    if args.release_controller and not args.actuate:parser.error('--release-controller requires --actuate; it changes robot control ownership')
    if args.record_sensors and not args.default_pose and not args.preflight and not 0<args.duration_s<=300:
        parser.error('Recorded policy sessions require --duration-s in (0, 300]')
    if args.external_camera:
        if not args.record_sensors or args.preflight or args.default_pose:parser.error('--external-camera needs a recorded policy session')
        if not Path(args.external_camera).exists():parser.error(f'Camera device {args.external_camera} not found')
        if not 1<=args.camera_fps<=60 or not re.fullmatch(r'\d+x\d+',args.camera_size):parser.error('Camera fps 1..60 and size WxH')
    if args.live_odometry:
        if args.default_pose or args.preflight:parser.error('--live-odometry uses normal sensor-only or actuated deployment')
        if not args.record_sensors or not 0<args.duration_s<=300:parser.error('Live odometry evaluation requires recording and duration in (0, 300]')
        if not 1<=args.odometry_port<=65535:parser.error('Invalid odometry UDP port')
        if not args.odometry_mount.is_file():parser.error('Missing odometry mount calibration')
        if args.odometry_clock is not None:
            from motion_diffusion.behavioral.hardware.pelvis_odometry import load_clock_calibration
            load_clock_calibration(args.odometry_clock)
    # DDS, depth and journaling share this interpreter. The default 5 ms GIL
    # timeslice is a quarter of a control tick; give I/O threads shorter turns.
    sys.setswitchinterval(.001)
    mode_name='Default-pose check' if args.default_pose else f'{args.model.capitalize()} / {args.backend}'
    print(f'{mode_name} — '+('model check, no sensors or motor output.' if args.check else
          'motor control enabled; startup enters zero torque before waiting for your buttons.' if args.actuate else
          f'sensor-only test ({args.duration_s:g} seconds), no motor output.'),flush=True)
    root,manifest=bundle.load_bundle(settings['bundle']);row=manifest['models'][args.model]
    import torch
    torch.set_num_threads(1)
    checkpoint=torch.load(bundle.asset_path(root,row['controller']),map_location='cpu',weights_only=True,mmap=True)
    hardware=hardware_settings(settings,checkpoint['action_vae']['model_contract'])
    replan_ticks=bundle.replan_ticks_for_hz(args.replan_hz,horizon=row['inspected_contract']['F'])
    config,receipt=bundle.resolve_session(root,manifest,args.model,config_path,hardware_config=hardware,
        replan_ticks=replan_ticks,seed=args.seed,noise_policy=args.noise_policy)
    config['startup_zero_torque']=bool(args.actuate)
    receipt['startup_zero_torque']=bool(args.actuate)
    if args.start_pose:
        from motion_diffusion.behavioral.hardware.start_pose import load_start_pose,perturb_start_pose
        from motion_diffusion.behavioral.hardware.unitree import NativePDContract
        contract=dict(checkpoint['action_vae']['model_contract'],deployment_pd=config['pd']['training_pd'])
        path=ROOT/'g1_deploy/config/poses/prone_ground_01.json' if args.start_pose=='prone' else Path(args.start_pose)
        pd=NativePDContract(contract,config['pd'])
        config['start_pose']=load_start_pose(path,pd)
        if args.start_pose_jitter_deg:
            config['start_pose']=perturb_start_pose(config['start_pose'],pd,args.start_pose_jitter_deg,args.start_pose_seed)
            jitter=config['start_pose']['randomization']
            print(f'Start pose jitter: +/- {jitter["amplitude_deg"]:g} deg; pose seed {jitter["seed"]}. '
                  'Sampled once; START moves slowly to this target. Saved in start_pose.json.',flush=True)
        receipt['start_pose']=config['start_pose']
    print(f'Planner seed: {args.seed}; noise policy: {args.noise_policy} ('+
          ('new noise each plan' if args.noise_policy=='fresh' else 'one noise sample reused; replanning stays active')+').',flush=True)
    if not args.default_pose:
        from motion_diffusion.behavioral.action_limits import deployment_command_limits,PDTargetLimiter
        from omegaconf import OmegaConf
        training=OmegaConf.load(bundle.asset_path(root,row['controller_training_config']))
        parameters=deployment_command_limits(checkpoint['action_vae']['model_contract'],config['pd']['training_pd'],training.robot)
        # Parse and validate once, before opening sensors or enabling motor output.
        command_contract=PDTargetLimiter(**parameters).contract()
        config['command_limits']=dict(mode=args.command_limits,parameters=parameters,
            source_config_sha256=row['controller_training_config']['sha256'])
        receipt['command_limits']=dict(mode=args.command_limits,**command_contract)
    if replan_ticks is not None:
        print(f'Requested planner cadence: {50/replan_ticks:g} Hz; actual replies remain subject to compute/transport latency.',flush=True)
    if args.check:
        result=bundle.inspect_pair(bundle.asset_path(root,row['planner']),
            bundle.asset_path(root,row['controller']),config['planner_options'],row['objective'],sample=True)
        from motion_diffusion.behavioral.hardware.export_pd import derive_pd
        from omegaconf import OmegaConf
        training=OmegaConf.load(bundle.asset_path(root,row['controller_training_config']))
        actual_pd=derive_pd(checkpoint['action_vae']['model_contract'],training)
        if actual_pd!=config['pd']['training_pd']:
            raise ValueError('Bundled PD differs from the original tracker training configuration')
        print(f"Verified {args.model}: checkpoint hashes, paired contracts and CPU synthetic inference.")
        print(f"H{result['H']} / H_z{result['H_z']} / F{result['F']} / S{result['state_dim']} / Z{result['latent_dim']}; PD gains, default pose and action scales match tracker training.")
        options=config['planner_options']
        print(f"Planner: {options['sampler']}{options['num_inference_steps']}, EMA={options['use_ema']}, "
            f"{options['numerical_precision']}, {options['noise_policy']} noise, "
            f"K{options['replan_every_ticks']}/blend{options['blend_ticks']}, {options['guidance']} guidance; "
            f"maximum reply age {config['max_plan_age_ticks']} ticks.")
        return
    live_odometry=None
    if args.live_odometry:
        live_odometry=dict(port=args.odometry_port,mode=args.odometry_mode,
            mount_calibration=str(args.odometry_mount.resolve()),
            clock_calibration=None if args.odometry_clock is None else str(args.odometry_clock.resolve()))
        config.update(pelvis_odometry_port=args.odometry_port,pelvis_odometry_mode=args.odometry_mode)
        config['live_odometry']=live_odometry
    if args.backend!='pytorch' and not args.default_pose:
        for component,key in [('controller','controller'),('planner',args.model)]:
            if key not in settings['exports']:
                parser.error(f'No compiled export for {args.model}/{key}; use --backend pytorch')
            export=Path(settings['exports'][key])
            if not export.is_absolute():export=ROOT/export
            config[component+'_engine']=dict(backend=args.backend,directory=str(export),
                precision=args.planner_precision if component=='planner' else 'fp32',execution_mode=args.engine_mode,
                export_receipt_sha256=bundle.sha256(export/'export.json'))
    mode='default_pose' if args.default_pose else f'{args.model}_{args.backend}'
    output=args.output or RESULTS_ROOT/f'robot_{mode}_{time.time_ns()}'
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    if config.get('start_pose') is not None:
        bundle.write_json(output/'start_pose.json',config['start_pose'])
    if args.record_sensors and args.odometry_mount.is_file():
        import shutil
        mount_snapshot=output/'odometry_mount.json'
        shutil.copyfile(args.odometry_mount,mount_snapshot)
        config['odometry_mount_calibration']=str(mount_snapshot)
        if live_odometry is not None:live_odometry['mount_calibration']=str(mount_snapshot)
    if args.record_depth and not args.default_pose:
        config['depth_recording_directory']=str(output/'sensors/depth')
    if args.record_model and not args.default_pose:
        from motion_diffusion.behavioral.hardware.model_trace import prepare_trace
        config['model_trace_directory']=str(output/'model_trace')
        prepare_trace(output/'model_trace',config,ROOT)
        print('Model recording enabled: exact inputs, depth history, noise and outputs; asynchronous disk writes.',flush=True)
    receipt.update(config_path=str(config_path),backend=args.backend,
        mode='default_pose' if args.default_pose else ('actuation' if args.actuate else 'preflight' if args.preflight else 'sensor_dry_run'),
        release_controller_requested=args.release_controller,
        release_controller_on_startup=bool(args.actuate))
    bundle.write_json(output/'session.json',receipt);bundle.write_json(output/'resolved.json',config)
    print(f'Session: {output}',flush=True)
    if not args.default_pose:
        print(f'Command limits: {args.command_limits}'+
            (' — targets start at measured pose; half-rate for 1 s, gradual release to normal rate by 2 s.'
             if args.command_limits=='bounded' else ' — original direct PD targets.'),flush=True)
    from motion_diffusion.behavioral.hardware.cli import run
    motion_switcher=None;recording=None
    def release_control():
        nonlocal motion_switcher
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from sonic.common.unitree_mode import release_active_motion_mode
        print('Taking low-level control now. Keep the robot supported.',flush=True)
        ChannelFactoryInitialize(config['robot']['domain_id'],config['robot']['interface'])
        motion_switcher=release_active_motion_mode()
        bundle.write_json(output/'controller_release.json',dict(released=True))
    try:
        if args.default_pose:
            run_default_pose(config,checkpoint['action_vae']['model_contract'],
                release_controller=release_control if args.actuate else None)
            return
        if args.record_sensors and not args.preflight:
            from record_motion import PolicyRecording
            if not config['robot'].get('require_remote_start'):
                raise ValueError('A-triggered recording requires require_remote_start')
            recording=PolicyRecording(output/'sensors',interface=config['robot']['interface'],
                seconds=args.duration_s,existing_driver=args.existing_lidar_driver,label=args.record_label,
                live_odometry=live_odometry,automatic_start=not args.actuate,
                camera=None if not args.external_camera else dict(device=args.external_camera,size=args.camera_size,fps=args.camera_fps))
            recording.prepare()
        result=run(config,enable_actuation=args.actuate,duration_s=args.duration_s,
            telemetry_jsonl=output/'telemetry.jsonl.gz',preflight_only=args.preflight,
            release_controller=release_control if args.actuate else None,
            before_policy_ready=recording.wait_until_ready if recording is not None else None,
            on_policy_start=recording.start if recording is not None else None)
    except KeyboardInterrupt as exc:
        bundle.write_json(output/'STOPPED.json',dict(reason=str(exc) or 'operator interrupt'))
        button={'Remote A stop':'A','Remote SELECT stop':'SELECT'}.get(str(exc))
        print(f'Stopped by {button}.' if button else 'Stopped by operator.',flush=True)
        return
    except BaseException as exc:
        bundle.write_json(output/'FAILED.json',dict(error=str(exc)));raise
    finally:
        # run() closes/damps the robot before this potentially expensive save.
        if recording is not None:recording.close()
        if not args.preflight and not args.default_pose:
            from record_motion import session_manifest
            session_manifest(output)
    bundle.write_json(output/'COMPLETE.json',result)
    if args.preflight:
        print(f'Preflight complete: {result["startup_checks"]} live first-plan checks; '
              f'{result["startup_timing_failures"]} timing misses; no motor output.',flush=True)
        return
    print(f'Finished {result["ticks"]/50:.1f} seconds. '
          f'Control step: median {result["control_wall_p50_ms"]:.1f} ms, '
          f'p95 {result["control_wall_p95_ms"]:.1f} ms. '
          f'Command-limit failures: {sum(result["command_limit_violations"].values())}.',flush=True)


if __name__=='__main__':main()
