"""Audit saved results; optionally compare a separate execution with identical source."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--results',required=True);ap.add_argument('--compare');args=ap.parse_args()
p=Path(args.results);d=pd.read_csv(p/'runs.csv');cfg=json.loads((p/'config.json').read_text())
expected=len(cfg['scenarios'])*len(cfg['models'])*len(cfg['budgets'])*cfg['repetitions']*9
v={'expected_run_count':expected,'run_count_correct':len(d)==expected,
   'coverage_in_0_1':bool(d.coverage.between(0,1).all()),
   'expected_budget_constraint':bool((d.expected_nominal_c<=d.budget+1e-8).all()),
   'actual_calls_below_assigned':bool((d.mean_actual_calls<=d.mean_nominal_c+1e-8).all()),
   'all_core_metrics_finite':bool(np.isfinite(d[['coverage','mean_lpb','mean_nominal_c','mean_actual_calls','tau']]).all().all())}
a=d[(d.c_source=='known')&d.method.isin(['Trimmed','Optimized'])]
v['trimmed_optimized_max_weight_bound']=bool((a.max_inverse_probability<=cfg['max_weight']+1e-8).all())
idx=['scenario','model','method','budget','repeat']
paired=d[d.c_source=='known'].set_index(idx).join(d[d.c_source=='learned'].set_index(idx),lsuffix='_known',rsuffix='_learned',how='inner')
v['same_observations_for_C_comparison']=bool(np.allclose(paired.mean_nominal_c_known,paired.mean_nominal_c_learned) and np.allclose(paired.mean_actual_calls_known,paired.mean_actual_calls_learned))
pivot=a.pivot(index=['scenario','model','budget','repeat'],columns='method',values='mean_inverse_probability')
v['optimization_reduces_mean_inverse_weight']=bool((pivot.Optimized<=pivot.Trimmed+1e-10).all())
if args.compare:
 b=pd.read_csv(Path(args.compare)/'runs.csv')
 assert d[idx+['c_source']].equals(b[idx+['c_source']]),'Comparison has different row identities/order'
 v['rerun_coverage_matches']=bool(np.allclose(d.coverage,b.coverage,atol=1e-12,rtol=1e-12))
 v['rerun_lpb_matches']=bool(np.allclose(d.mean_lpb,b.mean_lpb,atol=1e-8,rtol=1e-12))
 v['rerun_coverage_max_absolute_difference']=float(abs(d.coverage-b.coverage).max())
 v['rerun_lpb_max_absolute_difference']=float(abs(d.mean_lpb-b.mean_lpb).max())
(p/'validation.json').write_text(json.dumps(v,indent=2))
print(json.dumps(v,indent=2));assert all(value for value in v.values() if isinstance(value,bool))
