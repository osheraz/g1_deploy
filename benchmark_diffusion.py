#!/usr/bin/env python3
"""Offline GPU qualification: fixed-input parity, warmed latency and concurrent planner/tracker.

No robot imports, connections or motor commands. Concurrent tests use fixed synthetic
sensor/decoder fixtures, real 50 Hz wall-clock scheduling and the deployment planner IPC.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PROFILES={
    'pytorch':dict(backend='pytorch'),
    'trt_standard':dict(backend='tensorrt',precision='fp32',execution_mode='standard'),
    'trt_buffered':dict(backend='tensorrt',precision='fp32',execution_mode='buffered'),
    'trt_graph':dict(backend='tensorrt',precision='fp32',execution_mode='cuda_graph'),
    'trt_tf32':dict(backend='tensorrt',precision='tf32',execution_mode='cuda_graph'),
}


def engine_config(settings,key,profile,*,tracker=False):
    config=dict(PROFILES[profile])
    if config['backend']=='pytorch':return config
    directory=Path(settings['exports'][key])
    if not directory.is_absolute():directory=ROOT/directory
    config.update(directory=str(directory),precision='fp32' if tracker else config['precision'])
    if profile=='trt_tf32' and not tracker:config['validation_mode']='measure'
    return config


def fixtures(runtime):
    import torch
    from motion_diffusion.behavioral.hardware.export import graph_examples
    model=runtime.model;examples=graph_examples(model=model);pairs=[]
    generator=torch.Generator(device=model.device).manual_seed(17)
    for i,sample in enumerate(examples['denoiser'][3]):
        hs,ha,sm,am,*_=sample;depth=None
        if model.depth_encoder is not None:
            depth_samples=examples['depth'][3]
            frames,metadata,available=depth_samples[i%len(depth_samples)]
            depth=dict(frames=frames,metadata=metadata,available=available)
        request=dict(history=dict(state=hs,action=ha,state_mask=sm,action_mask=am),
                     depth=depth,origin_tick=0,generation=0)
        noise=dict(state=torch.randn(1,model.F,67,device=model.device,generator=generator),
                   action=torch.randn(1,model.F,model.action_dim,device=model.device,generator=generator))
        pairs.append((request,noise))
    return pairs


def concurrent(checkpoint,options,engine,controller,features,requests,seconds):
    import numpy as np
    import torch
    from motion_diffusion.behavioral.hardware.execution import PlannerProcess
    from motion_diffusion.behavioral.hardware.telemetry import latency_summary
    # Fixed fixtures isolate scheduling/compute; no fictitious robot latency.
    with_worker=PlannerProcess(checkpoint,device=str(controller.device),options=options,
                              engine_config=engine,startup_timeout_s=600.)
    tracker=[];scheduled=[];roundtrip=[];sample=[];misses=0;submitted=0
    width=controller.contract['proprio_dim'];interval=options['replan_every_ticks']
    try:
        with_worker.warmup(requests,timeout_s=60.)
        for i in range(10):
            x=features[i%len(features)].to(controller.device)
            controller(x[:,:width],x[:,width:]).cpu()
        start=time.perf_counter();sent_at=None
        for tick in range(round(seconds/.02)):
            deadline=start+tick*.02
            remaining=deadline-time.perf_counter()
            if remaining>0:time.sleep(remaining)
            capture=time.perf_counter()
            reply=with_worker.poll()
            if reply is not None:
                roundtrip.append(time.perf_counter()-sent_at);sample.append(reply['sample_wall_s'])
            if tick%interval==0 and not with_worker.inflight:
                request=dict(requests[submitted%len(requests)],origin_tick=tick)
                sent_at=time.perf_counter()
                with_worker.submit(request);submitted+=1
            x=features[tick%len(features)].to(controller.device)
            action=controller(x[:,:width],x[:,width:]).cpu().numpy()
            if not np.isfinite(action).all():raise FloatingPointError('Nonfinite tracker output')
            ready=time.perf_counter();tracker.append(ready-capture);scheduled.append(ready-deadline)
            misses+=int(ready-deadline>.02)
        # Drain the last reply outside the tracker measurement period.
        timeout=time.perf_counter()+5.
        while with_worker.inflight and time.perf_counter()<timeout:
            reply=with_worker.poll()
            if reply is not None:
                roundtrip.append(time.perf_counter()-sent_at);sample.append(reply['sample_wall_s'])
            else:time.sleep(.001)
    finally:with_worker.close()
    return dict(local_input_to_action=latency_summary(tracker),scheduled_tick_to_action=latency_summary(scheduled),
        planner_request_to_observed_reply=latency_summary(roundtrip),planner_sample=latency_summary(sample),
        deadline_misses_20ms=misses,submitted_plans=submitted,
        scope='Synthetic input availability to CPU action; includes H2D, tracker, D2H, planner IPC and concurrent GPU load. No sensor/network/DDS measurement. Replies polled at 50 Hz.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('diffusion','flow','both'),default='both')
    p.add_argument('--profiles',nargs='+',choices=PROFILES,default=list(PROFILES))
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--concurrent-seconds',type=float,default=20.)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.iterations<5 or args.warmup<1 or args.concurrent_seconds<=0:p.error('Positive warmup/duration and at least 5 iterations required')
    import numpy as np
    import torch
    import yaml
    import tensorrt
    from motion_diffusion.behavioral.hardware.bundle import load_bundle,asset_path
    from motion_diffusion.behavioral.hardware.cli import load_controller
    from motion_diffusion.behavioral.hardware.engines import configure_controller,configure_planner
    from motion_diffusion.behavioral.hardware.execution import sample_snapshot
    from motion_diffusion.behavioral.hardware.export import graph_examples
    from motion_diffusion.behavioral.hardware.telemetry import latency_summary
    from motion_diffusion.behavioral.runtime import load_frozen_planner
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    settings=yaml.safe_load((ROOT/'g1_deploy/configs/diffusion.yaml').read_text())
    root,manifest=load_bundle(settings['bundle'])
    output=args.output or ROOT/'logs/native_recovery'/f'latency_{time.time_ns()}'
    output.mkdir(parents=True,exist_ok=False)
    report=dict(gpu=torch.cuda.get_device_name(args.device),torch=torch.__version__,cuda=torch.version.cuda,
                tensorrt=tensorrt.__version__,arguments=vars(args)|{'output':str(output)},results={},
                live_robot_latency_measured=False)
    def save():(output/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    save()
    for name in (('diffusion','flow') if args.model=='both' else (args.model,)):
        row=manifest['models'][name];checkpoint=asset_path(root,row['planner'])
        options=dict(row['reference_inference'],seed=42,sampling_rng='legacy_v1')
        runtime=load_frozen_planner(checkpoint,args.device,**options)
        controller=load_controller(dict(controller_checkpoint=str(asset_path(root,row['controller'])),
            pd_sidecar=str(asset_path(root,row['pd_binding'])),tracker_device=args.device))
        pairs=fixtures(runtime)
        features=[x[0] for x in graph_examples(controller=controller)['controller'][3]]
        width=controller.contract['proprio_dim']
        with torch.no_grad():
            expected=[sample_snapshot(runtime,r,n)['prediction'] for r,n in pairs]
            expected_actions=[controller(x[:,:width].to(args.device),x[:,width:].to(args.device)).cpu() for x in features]
            for profile in args.profiles:
                print(f'{name}/{profile}',flush=True)
                result={};report['results'][name+'/'+profile]=result
                try:
                    runtime.model.inference_backend=None
                    if runtime.model.depth_encoder is not None:runtime.model.depth_encoder.inference_backend=None
                    controller.inference_backend=None
                    pc=engine_config(settings,name,profile);tc=engine_config(settings,'controller',profile,tracker=True)
                    configure_planner(runtime,checkpoint,pc);configure_controller(controller,tc)
                    result['neural_validation']={key:engine.validation_report for key,engine in (
                        ('denoiser',runtime.model.inference_backend),
                        ('depth',None if runtime.model.depth_encoder is None else runtime.model.depth_encoder.inference_backend))
                        if engine is not None}
                    errors={};passed=True
                    for (request,noise),native in zip(pairs,expected):
                        prediction=sample_snapshot(runtime,request,noise)['prediction']
                        for key,value in prediction.items():
                            errors[key]=max(errors.get(key,0.),float((value-native[key]).abs().max()))
                            passed &= bool(torch.allclose(value,native[key],atol=1e-3,rtol=1e-4))
                    tracker_error=0.
                    for x,native in zip(features,expected_actions):
                        value=controller(x[:,:width].to(args.device),x[:,width:].to(args.device)).cpu()
                        tracker_error=max(tracker_error,float((value-native).abs().max()))
                    result.update(parity_passed=passed and tracker_error<=1e-3,
                                  planner_max_abs_error=errors,tracker_max_abs_error=tracker_error)
                    times=[];tracker_times=[]
                    for i in range(args.warmup+args.iterations):
                        r,n=pairs[i%len(pairs)];sample=sample_snapshot(runtime,r,n)
                        x=features[i%len(features)]
                        began=time.perf_counter()
                        controller(x[:,:width].to(args.device),x[:,width:].to(args.device)).cpu()
                        elapsed=time.perf_counter()-began
                        if i>=args.warmup:times.append(sample['sample_wall_s']);tracker_times.append(elapsed)
                    result.update(planner=latency_summary(times),tracker_input_to_cpu_action=latency_summary(tracker_times))
                    result['concurrent']=concurrent(checkpoint,options,pc,controller,features,
                        [r for r,n in pairs],args.concurrent_seconds)
                    result['status']='completed'
                    np.savez(output/f'{name}_{profile}_latency.npz',planner_s=times,tracker_s=tracker_times)
                except Exception as exc:
                    result.update(status='failed',error=str(exc))
                save();print(json.dumps(result,indent=2),flush=True)
        del runtime,controller
        torch.cuda.empty_cache()
    print(f'Report: {output}/report.json',flush=True)


if __name__=='__main__':main()
