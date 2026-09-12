"""Diffusion recovery on G1: one config, default sensor-only inference.

python g1_deploy/deploy_diffusion.py
python g1_deploy/deploy_diffusion.py --backend tensorrt
python g1_deploy/deploy_diffusion.py --default-pose --actuate --release-controller
python g1_deploy/deploy_diffusion.py --backend tensorrt --actuate
Motor mode: initially idle. A starts policy from the current pose; A again stops.
Before starting only, START optionally moves to default over 2 s. SELECT stops.
Default-pose-only mode also waits for START, then holds until SELECT/A/Ctrl+C.
--release-controller releases the built-in controller only on A/START takeover;
the hoist must already support the robot.
"""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'g1_deploy'))
sys.path.insert(0,str(ROOT/'dependencies/isaac_utils'))


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
        def activate():
            pd.check_startup_pose(robot.observation())
            if release_controller is not None:release_controller()
            pd.check_startup_pose(robot.observation())
            robot.enable_motor_output()
        wait_for_policy_start(robot,pd,pose_only=True,activate=activate)
    finally:robot.close()


def main():
    from core.config import _load_config
    from motion_diffusion.behavioral.hardware import bundle
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'g1_deploy/configs/diffusion.yaml')
    parser.add_argument('--model',choices=['diffusion','flow'],default='diffusion')
    parser.add_argument('--backend',choices=['pytorch','tensorrt','onnx'],default='tensorrt')
    parser.add_argument('--planner-precision',choices=['fp32','tf32'],default='fp32')
    parser.add_argument('--engine-mode',choices=['standard','buffered','cuda_graph'],default='cuda_graph')
    parser.add_argument('--actuate',action='store_true')
    parser.add_argument('--preflight',action='store_true',help='Repeat live startup checks without motor output or takeover')
    parser.add_argument('--release-controller',action='store_true',
        help='Explicitly release the built-in motion service; requires --actuate and a supported robot')
    parser.add_argument('--default-pose',action='store_true',
        help='Wait for START to move to default and hold; SELECT/A stops; requires --actuate; no policy')
    parser.add_argument('--duration-s',type=float,default=30.)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.backend!='tensorrt' and args.planner_precision!='fp32':parser.error('TF32 engine precision requires TensorRT')
    if args.default_pose and not args.actuate:parser.error('--default-pose moves motors: add --actuate explicitly')
    if args.preflight and (args.actuate or args.default_pose):parser.error('--preflight is sensor-only')
    if args.release_controller and not args.actuate:parser.error('--release-controller requires --actuate; it changes robot control ownership')
    # DDS, depth and journaling share this interpreter. The default 5 ms GIL
    # timeslice is a quarter of a control tick; give I/O threads shorter turns.
    sys.setswitchinterval(.001)
    config_path=args.config.resolve();settings=_load_config(config_path)
    mode_name='Default-pose check' if args.default_pose else f'{args.model.capitalize()} / {args.backend}'
    print(f'{mode_name} — '+('motor control enabled; waiting for your buttons.' if args.actuate else
          f'sensor-only test ({args.duration_s:g} seconds), no motor output.'),flush=True)
    root,manifest=bundle.load_bundle(settings['bundle']);row=manifest['models'][args.model]
    import torch
    torch.set_num_threads(1)
    checkpoint=torch.load(bundle.asset_path(root,row['controller']),map_location='cpu',weights_only=True,mmap=True)
    hardware=hardware_settings(settings,checkpoint['action_vae']['model_contract'])
    config,receipt=bundle.resolve_session(root,manifest,args.model,config_path,hardware_config=hardware)
    if args.backend!='pytorch' and not args.default_pose:
        for component,key in [('controller','controller'),('planner',args.model)]:
            export=Path(settings['exports'][key])
            if not export.is_absolute():export=ROOT/export
            config[component+'_engine']=dict(backend=args.backend,directory=str(export),
                precision=args.planner_precision if component=='planner' else 'fp32',execution_mode=args.engine_mode,
                export_receipt_sha256=bundle.sha256(export/'export.json'))
    mode='default_pose' if args.default_pose else f'{args.model}_{args.backend}'
    output=args.output or ROOT/'logs/native_recovery'/f'robot_{mode}_{time.time_ns()}'
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    receipt.update(config_path=str(config_path),backend=args.backend,
        mode='default_pose' if args.default_pose else ('actuation' if args.actuate else 'preflight' if args.preflight else 'sensor_dry_run'),
        release_controller_requested=args.release_controller)
    bundle.write_json(output/'session.json',receipt);bundle.write_json(output/'resolved.json',config)
    print(f'Session: {output}',flush=True)
    from motion_diffusion.behavioral.hardware.cli import run
    motion_switcher=None
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
                release_controller=release_control if args.release_controller else None)
            return
        result=run(config,enable_actuation=args.actuate,duration_s=args.duration_s,
            telemetry_jsonl=output/'telemetry.jsonl.gz',preflight_only=args.preflight,
            release_controller=release_control if args.release_controller else None)
    except KeyboardInterrupt as exc:
        bundle.write_json(output/'STOPPED.json',dict(reason=str(exc) or 'operator interrupt'))
        button={'Remote A stop':'A','Remote SELECT stop':'SELECT'}.get(str(exc))
        print(f'Stopped by {button}.' if button else 'Stopped by operator.',flush=True)
        return
    except BaseException as exc:
        bundle.write_json(output/'FAILED.json',dict(error=str(exc)));raise
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
