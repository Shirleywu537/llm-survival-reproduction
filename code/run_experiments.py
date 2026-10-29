"""Run fixed-split, paired calibration repeats; no expensive response strings needed.
Usage: python run_experiments.py --config config.json --out /path/to/results
"""
import argparse,json,time,platform,sys,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from core import *
from data import generate,auxiliary
from models import ConditionalModel,c_features


def fit_c_models(family,t_model,cal_p,scenario,normalizer,cfg,seed,folder):
    aux=auxiliary(scenario,cfg['c_pilot_train']+cfg['c_pilot_test'],seed,normalizer)
    ntr=cfg['c_pilot_train'];at=aux.subset(slice(0,ntr));av=aux.subset(slice(ntr,None))
    pat=t_model.predict(at.x);pav=t_model.predict(av.x)
    rng=np.random.default_rng(seed+1);models={};diagnostics=[]
    for method in METHODS[1:]:
        xs=[];ys=[];bs=[];vs=[]
        for b in cfg['budgets']:
            if method=='Naive':
                xt=c_features(at.x,b);xv=c_features(av.x,b)
                yt=rng.geometric(1/b,len(at.x));yv=rng.geometric(1/b,len(av.x));truth=np.full(len(av.x),1/b)
            else:
                d=design(method,cal_p,b,cfg['tau_prior'],cfg['max_weight'])
                ct,pt=d.for_predictions(pat,cfg['tau_prior']);cv,pv=d.for_predictions(pav,cfg['tau_prior'])
                xt=c_features(at.x,b,ct);xv=c_features(av.x,b,cv)
                yt=(rng.random(len(at.x))<pt).astype(float);yv=(rng.random(len(av.x))<pv).astype(float);truth=pv
            xs.append(xt);ys.append(yt);bs.append(np.full(len(at.x),b));vs.append((b,xv,yv,truth))
        model=ConditionalModel(family,'C_geom' if method=='Naive' else 'C_ber',cfg,seed+METHODS.index(method))
        model.fit(np.concatenate(xs),np.concatenate(ys),np.concatenate(bs))
        model.save(folder/f'C_{method}');models[method]=model
        for b,xv,yv,truth in vs:
            pred=model.predict(xv,np.full(len(xv),b))
            diag={'scenario':scenario,'model':family,'method':method,'budget':b,
                  'c_parameter_mae':np.mean(np.abs(pred-truth)),
                  'c_relative_parameter_mae':np.mean(np.abs(pred/truth-1)),
                  'c_parameter_bias':np.mean(pred-truth)}
            if method!='Naive':
                diag['brier_score']=np.mean((pred-yv)**2)
                diag['oracle_brier_score']=np.mean((truth-yv)**2)
                diag['fraction_clipped_low']=np.mean(pred<=cfg['c_probability_floor'])
            diagnostics.append(diag)
        print(f'  learned C model: {method}',flush=True)
    return models,diagnostics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='config.json');ap.add_argument('--out',required=True)
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(cfg,indent=2));start=time.time()
    grid=np.r_[0.,np.geomspace(1e-5,cfg['tau_prior'],cfg['grid_size'])]
    allrows=[];t_diag=[];c_diag=[];data_diag=[];timings=[]
    for si,scenario in enumerate(cfg['scenarios']):
        seed=cfg['seed']+10000*si
        n=cfg['n_prompts'];nt=int(n*cfg['train_fraction']);nc=int(n*cfg['cal_fraction'])
        dist,norm=generate(scenario,n,seed)
        train=dist.subset(slice(0,nt));cal=dist.subset(slice(nt,nt+nc));test=dist.subset(slice(nt+nc,None))
        counts=train.counts(np.random.default_rng(seed+1),cfg['training_generations'])
        oracle=test.oracle_lpb(cfg['alpha'])
        ds={'scenario':scenario,'n_train':nt,'n_cal':nc,'n_test':len(test.x),'normalizer':norm,
            'zero_label_fraction':float(np.mean(counts==0)), 'training_unsafe_outputs':int(counts.sum()),
            'training_total_outputs':int(nt*cfg['training_generations']),
            'mean_true_risk':float(test.mean_p().mean()),'oracle_mean_lpb':float(oracle.mean()),
            'oracle_coverage':float(test.coverage(oracle).mean()),'group1_prevalence':float(test.group.mean())}
        data_diag.append(ds)
        for mi,family in enumerate(cfg['models']):
            combo_start=time.time();key=f'{scenario}_{family}';folder=out/'models'/key;folder.mkdir(parents=True,exist_ok=True)
            print(f'\n{key}: fitting T on {nt} prompts',flush=True)
            tm=ConditionalModel(family,'T',cfg,seed+100+mi).fit(train.x,counts/cfg['training_generations'])
            tm.save(folder/'T');pc=tm.predict(cal.x);pt=tm.predict(test.x)
            td={'scenario':scenario,'model':family,'probability_mae':float(np.mean(np.abs(pt-test.mean_p()))),
                'log10_probability_rmse':float(np.sqrt(np.mean((np.log10(pt)-np.log10(test.mean_p()))**2))),
                'fraction_at_probability_floor':float(np.mean(pt<=cfg['probability_floor'])),
                'predicted_p_mean':float(pt.mean()),'predicted_p_min':float(pt.min()),'predicted_p_max':float(pt.max()),
                'uncalibrated_mean_lpb':float(geometric_quantile(pt,cfg['alpha']).mean()),
                'uncalibrated_coverage':float(test.coverage(geometric_quantile(pt,cfg['alpha'])).mean()),
                'final_training_loss':tm.history[-1] if tm.history else None}
            t_diag.append(td)
            print(f'  uncalibrated coverage={td["uncalibrated_coverage"]:.4f}; fitting C pilot models',flush=True)
            cm,cd=fit_c_models(family,tm,pc,scenario,norm,cfg,seed+200+mi,folder);c_diag.extend(cd)
            np.savez_compressed(folder/'evaluation_predictions.npz',cal_p=pc,test_p=pt,test_x=test.x,
                                test_p_low=test.p_low,test_p_high=test.p_high,test_mixing=test.mixing,test_group=test.group,
                                test_oracle=oracle,train_counts=counts,grid=grid)
            q=geometric_quantile(pc[None,:],grid[:,None])
            uncal=geometric_quantile(pt,cfg['alpha']);uncalmet=metric_set(uncal,test)
            for bi,b in enumerate(cfg['budgets']):
                bstart=time.time()
                designs={m:design(m,pc,b,cfg['tau_prior'],cfg['max_weight']) for m in METHODS[2:]}
                learned={m:cm[m].predict(c_features(cal.x,b,designs[m].cost),np.full(nc,b)) for m in METHODS[2:]}
                rhat=cm['Naive'].predict(c_features(cal.x,b),np.full(nc,b))
                wk=naive_weight_matrix(q,b);we=naive_weight_matrix(q,b,rhat)
                for rep in range(cfg['repetitions']):
                    # Common random numbers: identical T and U for all methods/models/budgets.
                    rng=np.random.default_rng(seed+100000+rep)
                    t=cal.sample_t(rng);u=rng.random(nc)
                    base={'scenario':scenario,'model':family,'budget':b,'repeat':rep}
                    allrows.append(dict(base,method='Uncalibrated',c_source='known',tau=cfg['alpha'],
                        mean_nominal_c=0.,mean_actual_calls=0.,expected_nominal_c=0.,selected_fraction=0.,**uncalmet))
                    censor=np.floor(np.log1p(-u)/np.log1p(-1/b))+1;observed=np.minimum(t,censor)
                    for source,weights,rate in [('known',wk,np.full(nc,1/b)),('learned',we,rhat)]:
                        tau=calibrate_naive(q,observed,censor,weights,grid,cfg['alpha'])
                        lower=geometric_quantile(pt,tau);qsel=geometric_quantile(pc,tau)
                        selected=qsel<=censor
                        effective_w=np.exp(np.minimum(-np.maximum(qsel[selected]-1,0)*np.log1p(-rate[selected]),700))
                        # Avoid squaring huge weights; compute effective sample size after normalization.
                        wn=effective_w/max(effective_w.max(initial=1),1)
                        ess=(wn.sum()**2/np.sum(wn**2)) if len(wn) else 0.
                        allrows.append(dict(base,method='Naive',c_source=source,tau=tau,
                            mean_nominal_c=float(censor.mean()),mean_actual_calls=float(observed.mean()),expected_nominal_c=b,
                            selected_fraction=float(selected.mean()),effective_sample_size=float(ess),
                            max_effective_weight=float(effective_w.max(initial=0)),**metric_set(lower,test)))
                    for method,d in designs.items():
                        censor=d.cost*(u<d.pi);observed=np.minimum(t,censor)
                        for source,prob in [('known',d.pi),('learned',learned[method])]:
                            tau=calibrate_adaptive(pc,observed,censor,prob,grid,cfg['alpha'])
                            lower=np.minimum(geometric_quantile(pt,tau),d.cap)
                            selected=censor>0;w=1/prob[selected]
                            allrows.append(dict(base,method=method,c_source=source,tau=tau,
                                mean_nominal_c=float(censor.mean()),mean_actual_calls=float(observed.mean()),
                                expected_nominal_c=float(np.mean(d.pi*d.cost)),selected_fraction=float(selected.mean()),
                                mean_inverse_probability=float(np.mean(1/prob)),max_inverse_probability=float(np.max(1/prob)),
                                max_effective_weight=float(w.max(initial=0)),
                                effective_sample_size=float(w.sum()**2/np.sum(w*w)) if len(w) else 0.,
                                **metric_set(lower,test)))
                del wk,we
                pd.DataFrame(allrows).to_csv(out/'runs.csv',index=False)
                print(f'  b={b:5g}, {cfg["repetitions"]} paired repeats, {time.time()-bstart:.1f}s',flush=True)
            del q
            timings.append({'scenario':scenario,'model':family,'seconds':time.time()-combo_start})
            pd.DataFrame(t_diag).to_csv(out/'t_model_diagnostics.csv',index=False)
            pd.DataFrame(c_diag).to_csv(out/'c_model_diagnostics.csv',index=False)
            pd.DataFrame(data_diag).to_csv(out/'data_diagnostics.csv',index=False)
    df=pd.DataFrame(allrows)
    measures=['coverage','mean_lpb','mean_nominal_c','mean_actual_calls','expected_nominal_c','tau',
              'conditional_coverage_rmse','coverage_group_0','coverage_group_1','mean_lpb_group_0','mean_lpb_group_1',
              'effective_sample_size','max_effective_weight','mean_inverse_probability','max_inverse_probability']
    summary=df.groupby(['scenario','model','method','c_source','budget'])[measures].agg(['mean','std','min','max'])
    summary.columns=['_'.join(a) for a in summary.columns];summary.reset_index().to_csv(out/'summary.csv',index=False)
    metadata={'total_seconds':time.time()-start,'timings':timings,'python':sys.version,'platform':platform.platform(),
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','torch','scikit-learn','matplotlib','pandas']},
        'reference_commit':'fd78d544a62cae88da97c056b537c497d0763848',
        'coverage_convention':'P(T >= L), support T=1,2,...',
        'grid':'{0} union geomspace(1e-5,tau_prior,grid_size)',
        'error_bands':'sample standard deviation over fixed-split calibration repeats; not confidence intervals',
        'training_outputs_are_separate_from_calibration_budget':True}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(f'Finished in {metadata["total_seconds"]:.1f}s',flush=True)

if __name__=='__main__':main()
