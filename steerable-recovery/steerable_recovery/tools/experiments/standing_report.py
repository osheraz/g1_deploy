"""Summarize all attempted paired standing cases, without dropping failures."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    from .standing_online import metrics
    rows=[]
    for report in sorted(a.root.glob('*/standing_comparison.json')):
        for case in json.loads(report.read_text()):
            case['group']=report.parent.name
            # Recompute common-window metrics, including newly added support heights.
            for arm,values in case['results'].items():
                dest=Path(case['output'])/arm
                case['results'][arm].update(metrics(dest,case['switch_s'],values['seconds']))
            rows.append(case)
    lines=['# S50R manual Y: measured lab scene','',
        'All attempted cases are retained. `planner` is the unchanged plan stream; `encoded` is the deployed manual-Y controller. '
        'Stress cases also include `continue`, which receives the same physical perturbation as Y.', '',
        '| Start/group | Seed | Y (s) | Mode | Fell after Y | Quiet at end | Total XY drift (cm) | Peak late joint speed (rad/s) | Support at Y/end (m) |',
        '|---|---:|---:|---|---|---|---:|---:|---|']
    for c in rows:
        for arm,r in c['results'].items():
            if c['switch_s'] is None:
                lines.append(f'| {c["group"]} | {c["seed"]} | — | {arm} | Y never reached | — | — | — | — |');continue
            lines.append(f'| {c["group"]} | {c["seed"]} | {c["switch_s"]:.2f} | {arm} | {r.get("fell_after_y")} | {r.get("quiet_at_end")} | '
                f'{r.get("drift_cm",float("nan")):.1f} | {r.get("late_peak_speed_rad_s",float("nan")):.3f} | '
                f'{r.get("support_height_at_y_m",0):.3f}/{r.get("support_height_at_end_m",0):.3f} |')
    lines+=['','Quiet at end: upright cosine > 0.9 and every joint speed < 0.5 rad/s throughout the last second, with no post-Y fall. '
        'Fall: upright cosine < 0.6 or pelvis clearance < 0.35 m at any recorded post-Y tick. '
        'Late excludes the first two seconds after Y. Support height is the terrain directly below pelvis XY, not measured foot support. '
        'XY drift includes the initial settling transition. These are explicit simulation diagnostics, not hardware safety certification.']
    (a.root/'results.md').write_text('\n'.join(lines)+'\n')
    (a.root/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=[c for c in rows if c['group'] in ('lab26','lab09','lab20','lab25') and c['seed']==52032 and 'encoded' in c['results']]
    if not cases:return
    fig,axes=plt.subplots(len(cases),2,figsize=(9,2.4*len(cases)),squeeze=False,constrained_layout=True)
    for ax,c in zip(axes,cases):
        switch=c['switch_s'];index=round(switch/.02)
        for arm,color,label in [('planner','#a75531','Continue planner'),('encoded','#1e7296','Press Y')]:
            x=np.load(Path(c['output'])/arm/'trajectory.npz');t=x['time'];root=x['root_position']
            drift=np.linalg.norm(root[:,:2]-root[index,:2],axis=-1)*100
            speed=np.max(np.abs(x['dq']),axis=-1)
            ax[0].plot(t[index:]-switch,drift[index:],color=color,lw=1.7,label=label)
            ax[1].plot(t[index:]-switch,speed[index:],color=color,lw=1.1,label=label)
        ax[0].set_ylabel(f'{c["group"]}\nPelvis displacement (cm)');ax[1].set_ylabel('Max joint speed (rad/s)')
        for x in ax:x.grid(alpha=.2);x.set_xlabel('Seconds after Y');x.spines[['top','right']].set_visible(False)
    axes[0,0].legend(frameon=False);fig.savefig(a.root/'comparison.png',dpi=180);plt.close(fig)
    print(a.root/'results.md');print(a.root/'comparison.png')


if __name__=='__main__':main()
