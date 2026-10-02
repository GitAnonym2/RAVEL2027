from __future__ import annotations

"""Reproduce the submitted Figure 2 from compact reference results."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE=Path(__file__).resolve().parent
MODEL_ORDER=["ravel_dynamic_recruitment_M2","matched_dynamic_delay_K2","ravel_static_recruitment_M2","ravel_nuisance_only","adm_native_selectedK"]
MODEL_LABELS={"ravel_dynamic_recruitment_M2":"RAVEL","matched_dynamic_delay_K2":"Matched\ndelay","ravel_static_recruitment_M2":"RAVEL\nstatic","ravel_nuisance_only":"Nuisance","adm_native_selectedK":"ADM"}
DELAY_COLORS=["#9ECAE1","#756BB1","#7F0000"]


def colored_line(ax,x,y,d,truth=False):
    pts=np.column_stack([x,y]).reshape(-1,1,2); seg=np.concatenate([pts[:-1],pts[1:]],axis=1)
    lc=LineCollection(seg,cmap=LinearSegmentedColormap.from_list("delay",DELAY_COLORS),norm=Normalize(5,35),linewidths=2.0 if truth else 2.4)
    lc.set_array(0.5*(d[:-1]+d[1:]));
    if truth: lc.set_linestyle((0,(4.5,2.4)))
    ax.add_collection(lc); ax.update_datalim(np.column_stack([x,y])); ax.autoscale_view(scalex=True,scaley=False)
    return lc


def mechanism_column(fig, gs, data, prefix, title, panel):
    axes=[fig.add_subplot(gs[i]) for i in range(3)]
    t=data[prefix+"_time_s"]
    sets=[("Ground truth",data[prefix+"_truth_recruitment"],data[prefix+"_truth_delay_ms"],True),("RAVEL",data[prefix+"_ravel_recruitment"],data[prefix+"_ravel_delay_ms"],False),("Matched dynamic-delay model",data[prefix+"_matched_recruitment"],data[prefix+"_matched_delay_ms"],False)]
    for j,(lab,r,d,truth) in enumerate(sets):
        for m,off in enumerate((-0.025,0.025)): colored_line(axes[j],t,r[:,m]+off,d[:,m],truth=truth)
        axes[j].set_ylim(0.25,1.75); axes[j].set_title(lab,fontsize=9); axes[j].spines[['top','right']].set_visible(False)
        if j<2: axes[j].tick_params(axis='x',labelbottom=False)
        else: axes[j].set_xlabel('event time (s)')
        if j==0: axes[j].text(-0.14,1.35,panel,transform=axes[j].transAxes,fontweight='bold',fontsize=12)
    axes[0].set_ylabel('recruitment'); axes[1].set_ylabel('recruitment'); axes[2].set_ylabel('recruitment')
    axes[0].text(0.5,1.28,title,transform=axes[0].transAxes,ha='center',fontweight='bold')
    return axes


def score_panel(ax,scores,world,panel,title):
    vals=[]
    for m in MODEL_ORDER:
        v=scores[(scores.world==world)&(scores.model==m)].nmse.to_numpy(float); vals.append((np.mean(v),np.std(v,ddof=1)))
    x=np.arange(len(MODEL_ORDER))
    for i,(mean,sd) in enumerate(vals): ax.errorbar(i,mean,yerr=sd,fmt='o',capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER],fontsize=8); ax.set_ylim(0,1.6); ax.set_ylabel('held-out spectral NMSE ↓'); ax.set_title(title,fontsize=10,fontweight='bold'); ax.spines[['top','right']].set_visible(False); ax.text(-0.11,1.05,panel,transform=ax.transAxes,fontweight='bold',fontsize=12)
    for i,(mean,_) in enumerate(vals): ax.text(i,mean+0.08,f'{mean:.3f}',ha='center',fontsize=7)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,default=HERE/'reference_results'); p.add_argument('--out',type=Path,default=HERE/'figure2_reproduced.pdf'); args=p.parse_args()
    data=np.load(args.data_dir/'figure2_curves.npz',allow_pickle=False); scores=pd.read_csv(args.data_dir/'figure2_scores.csv')
    fig=plt.figure(figsize=(9.5,6.5)); outer=fig.add_gridspec(2,2,height_ratios=[2.1,1.0],hspace=0.48,wspace=0.36)
    left=outer[0,0].subgridspec(3,1,hspace=0.22); right=outer[0,1].subgridspec(3,1,hspace=0.22)
    a=mechanism_column(fig,left,data,'stable','Dynamic-recruitment','A'); b=mechanism_column(fig,right,data,'dynamic','Dynamic-delay','B')
    c=fig.add_subplot(outer[1,0]); d=fig.add_subplot(outer[1,1]); score_panel(c,scores,'stable_recruitment','C','Dynamic-recruitment truth'); score_panel(d,scores,'dynamic_delay','D','Dynamic-delay truth')
    # shared colorbar for transport delay
    sm=plt.cm.ScalarMappable(norm=Normalize(5,35),cmap=LinearSegmentedColormap.from_list('delay',DELAY_COLORS)); sm.set_array([])
    cb=fig.colorbar(sm,ax=a+b,orientation='vertical',fraction=0.025,pad=0.02); cb.set_label('transport delay (ms)')
    fig.savefig(args.out,bbox_inches='tight'); png=args.out.with_suffix('.png'); fig.savefig(png,dpi=300,bbox_inches='tight'); print('wrote',args.out,'and',png)

if __name__=='__main__': main()
