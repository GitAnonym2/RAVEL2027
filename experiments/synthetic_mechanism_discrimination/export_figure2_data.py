from __future__ import annotations

"""Export compact Figure 2 plotting data from completed benchmark outputs."""

import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
WORLDS=("stable_recruitment","dynamic_delay")
SEEDS=(271101,271102,271103)


def norm(r):
    r=np.asarray(r,float).copy(); d=np.nanmean(r,axis=0,keepdims=True); d[np.abs(d)<1e-12]=1.; return r/d


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results-root",type=Path,default=HERE/"results")
    p.add_argument("--out",type=Path,default=HERE/"reference_results")
    p.add_argument("--selected-seed",type=int,default=271101)
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for world in WORLDS:
        for seed in SEEDS:
            sd=args.results_root/world/f"seed_{seed}"
            sm=json.loads((sd/"summary.json").read_text())
            rows.extend(sm["scores"])
            ap=sd/"adm_K_sweep"/"adm_K_sweep_summary.json"
            if ap.exists():
                a=json.loads(ap.read_text()); r=dict(a["common_evaluator"]); r["model"]="adm_native_selectedK"; r["selected_K"]=a["selected_K"]; rows.append(r)
    pd.DataFrame(rows).to_csv(args.out/"figure2_scores.csv",index=False)

    arr={}
    for world,prefix in [("stable_recruitment","stable"),("dynamic_delay","dynamic")]:
        sd=args.results_root/world/f"seed_{args.selected_seed}"
        z=np.load(sd/"data"/"dataset.npz",allow_pickle=False)
        sm=json.loads((sd/"summary.json").read_text())
        with open(sd/"ravel_fit.pkl","rb") as f: fit=pickle.load(f)
        model=fit["model"] if isinstance(fit,dict) else fit
        t=np.asarray(z["truth_window_centers_s"],float); W=len(t); arr[prefix+"_time_s"]=t
        if world=="stable_recruitment":
            tr=norm(z["truth_stable_recruitment"]); td=np.tile(z["truth_stable_delays_ms"][None,:],(W,1)); recs=sm["mechanism"]["ravel_stable_atom_matching"]; mrecs=sm["mechanism"]["matched_dynamic_stable_delay_matching"]
        else:
            tr=np.ones((W,2)); td=np.asarray(z["truth_dynamic_delay_ms_wm"],float); recs=sm["mechanism"]["ravel_dynamic_truth_vs_fixed_atoms"]; mrecs=sm["mechanism"]["matched_dynamic_trajectory_matching"]
        mp={int(x["truth_index"]):int(x["fit_index"]) for x in recs}; fr=np.asarray(model.recruitment,float); fd=np.asarray([a.delay_ms for a in model.atoms],float)
        rr=norm(np.column_stack([fr[:,mp[i]] for i in range(2)])); rd=np.column_stack([np.full(W,fd[mp[i]]) for i in range(2)])
        tau=np.asarray(sm["matched_dynamic_baseline"]["tau_ms_wm"],float); mm={int(x["truth_index"]):int(x["fit_index"]) for x in mrecs}; md=np.column_stack([tau[:,mm[i]] for i in range(2)])
        arr[prefix+"_truth_recruitment"]=tr; arr[prefix+"_truth_delay_ms"]=td; arr[prefix+"_ravel_recruitment"]=rr; arr[prefix+"_ravel_delay_ms"]=rd; arr[prefix+"_matched_recruitment"]=np.ones((W,2)); arr[prefix+"_matched_delay_ms"]=md
    np.savez_compressed(args.out/"figure2_curves.npz",**arr)
    print("wrote",args.out)

if __name__=="__main__": main()
