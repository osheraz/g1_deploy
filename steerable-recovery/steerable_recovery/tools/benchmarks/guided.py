"""Measure the shared asynchronous guided runtime with synthetic observations."""
import json
import time
from steerable_recovery.paths import ROOT, CONFIG_ROOT, VALIDATION_ROOT
from steerable_recovery.settings import DIFFUSION_RATES as MODEL_RATES

def benchmark(args, *, goal_path):
    """Actual spawned planner and TRT tracker, synthetic observations, no sensors."""
    import torch
    from core.config import _load_config
    from steerable_recovery.settings import model_settings, hardware_settings, sampling_seed
    from motion_diffusion.behavioral.hardware import bundle
    from motion_diffusion.behavioral.deploy import load_controller
    from motion_diffusion.behavioral.hardware.engines import configure_controller
    from motion_diffusion.behavioral.hardware.execution import PlannerProcess,AsyncNativeRecovery
    torch.set_num_threads(1)
    path=CONFIG_ROOT / 'models.yaml'
    settings=model_settings(_load_config(path),args.model)
    root,manifest=bundle.load_bundle(settings['bundle']);row=manifest['models'][args.model]
    ckpt=torch.load(bundle.asset_path(root,row['controller']),map_location='cpu',weights_only=True,mmap=True)
    hardware=hardware_settings(settings,ckpt['action_vae']['model_contract'])
    hardware['pelvis_odometry_port']=15560
    rate=MODEL_RATES[args.model] if args.replan_hz is None else args.replan_hz
    config,_=bundle.resolve_session(root,manifest,args.model,path,hardware_config=hardware,
        task='motion-command',goal_path=goal_path,
        guidance_scale=args.guidance_scale,blend_ticks=getattr(args,'blend_ticks',None),
        guided_timing=settings.get('guided_timing'),
        seed=sampling_seed(str(args.seed)),replan_ticks=bundle.replan_ticks_for_hz(rate,horizon=row['inspected_contract']['F']))
    actor=load_controller(config)
    export=ROOT/settings['exports']['controller']
    configure_controller(actor,dict(backend='tensorrt',directory=str(export),precision='fp32',execution_mode='cuda_graph'))
    worker=PlannerProcess(config['planner_checkpoint'],device=config['planner_device'],options=config['planner_options'])
    try:
        runtime=AsyncNativeRecovery(actor,worker,max_plan_age_ticks=config['max_plan_age_ticks'])
        cold=runtime.warmup()
        warm=runtime.warmup()
        report=dict(schema='hv2.guided_inference_check.v1',option=getattr(args,'option',None),
            goal_path=str(goal_path),
            model=args.model,replan_hz=rate,blend_ticks=config['planner_options']['blend_ticks'],max_plan_age_ticks=config['max_plan_age_ticks'],
            guidance_scale=config['planner_options']['motion_goal']['guidance_scale'],
            motors_opened=False,sensors_opened=False,cold=cold,warm=warm,
            planner_backend='pytorch',tracker_backend='tensorrt',
            scope='Synthetic pre-clock warmup; not concurrent sensor/recording or dynamic accuracy qualification')
        output=args.output or VALIDATION_ROOT/f'steering_benchmark_{time.time_ns()}'
        output.mkdir(parents=True,exist_ok=False)
        (output/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'Saved {output}/benchmark.json')
        print('Warm planner samples (ms):', [round(v*1000,1) for v in warm['planner']['sample_wall_s']])
    finally:worker.close()
