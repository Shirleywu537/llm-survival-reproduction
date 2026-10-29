"""Publication/export figures from saved results only; does not rerun experiments."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter,ScalarFormatter
from core import METHODS

COLORS={'Uncalibrated':'#7557A6','Naive':'#D55E00','Basic':'#B18A00','Trimmed':'#0072B2','Optimized':'#009E73'}
LABELS={'paper':'Paper-v2 synthetic distribution','hidden_intent_mixture':'Harder distribution: hidden-intent mixture'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.titleweight':'bold','axes.labelcolor':'#273340','text.color':'#172435','axes.edgecolor':'#AAB3BB',
    'grid.color':'#E0E5EA','grid.alpha':.8,'figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42})

def save(fig,path):
    for ext in ['png','pdf','svg']:fig.savefig(str(path)+'.'+ext,dpi=200,bbox_inches='tight')
    plt.close(fig)

def series(ax,rows,metric,color,label,linestyle='-',band=True):
    a=rows.groupby('budget')[metric].agg(['mean','std']).sort_index();x=a.index.to_numpy();y=a['mean'].to_numpy();s=a['std'].fillna(0).to_numpy()
    ax.plot(x,y,color=color,lw=2,label=label,ls=linestyle,marker='o',ms=3)
    if band:ax.fill_between(x,y-s,y+s,color=color,alpha=.13,linewidth=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    results=Path(a.results);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(results/'runs.csv');known=df[df.c_source=='known']
    for sc in LABELS:
        fig,axs=plt.subplots(2,3,figsize=(15.5,8.5),sharex=True)
        for ri,family in enumerate(['DNN','RF']):
            data=known[(known.scenario==sc)&(known.model==family)]
            for ci,(metric,title) in enumerate([('coverage','Coverage: P(T >= L)'),('mean_nominal_c','Assigned generation limits: mean C'),('mean_lpb','Mean lower prediction bound L')]):
                ax=axs[ri,ci]
                for m in METHODS:
                    if ci==1 and m=='Uncalibrated':continue
                    series(ax,data[data.method==m],metric,COLORS[m],m,linestyle='--' if m=='Uncalibrated' else '-')
                ax.set_title(title if ri==0 else '',pad=10);ax.grid(True);ax.set_xlim(0,10200)
                if ri==1:ax.set_xlabel('Expected calibration budget per prompt b')
                if ci==0:
                    ax.axhline(.9,ls=':',color='#48505A',lw=1.5);ax.yaxis.set_major_formatter(PercentFormatter(1))
                    mincov=data.groupby(['method','budget']).coverage.mean().min()
                    ax.set_ylim(max(0,min(.85,mincov-.07)),1.005);ax.set_ylabel(f'{family}\nCoverage',fontweight='bold')
                elif ci==1:
                    ax.plot([10,10000],[10,10000],color='#777777',ls=':',lw=1,label='Budget');ax.set_ylabel('Outputs / calibration prompt')
                else:
                    ax.set_yscale('log');ax.set_ylabel('Generations (log scale)')
        handles,labels=axs[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=5,loc='lower center',bbox_to_anchor=(.5,.025),frameon=False)
        fig.suptitle(LABELS[sc]+'  |  Figure 1-style reproduction',fontsize=18,fontweight='bold',y=.985)
        fig.text(.5,.93,'Known C distribution • 18,000 train / 18,000 calibration / 4,000 test • 20 paired repeats',ha='center',fontsize=11,color='#5A6774')
        fig.text(.5,.003,'Bands: ±1 sample SD over calibration draws, conditional on one split and fitted model. Middle panel shows assigned C; early-stop calls are reported separately.',ha='center',fontsize=9,color='#5A6774')
        fig.subplots_adjust(top=.87,bottom=.14,wspace=.28,hspace=.20)
        save(fig,out/f'figure1_{sc}')
    # Weight estimation: compare the same observed C/T with different weights.
    fig,axs=plt.subplots(2,2,figsize=(12.5,8),sharex=True)
    for si,sc in enumerate(LABELS):
        for mi,family in enumerate(['DNN','RF']):
            ax=axs[si,mi];data=df[(df.scenario==sc)&(df.model==family)]
            for method in ['Naive','Optimized']:
                for source,ls in [('known','-'),('learned','--')]:
                    series(ax,data[(data.method==method)&(data.c_source==source)],'coverage',COLORS[method],f'{method}, {source} C',ls)
            ax.axhline(.9,ls=':',color='#444444');ax.set_xscale('log');ax.set_ylim(.2,1.01);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(True)
            ax.set_title(f'{family} | '+('Paper-v2' if si==0 else 'Hidden-intent mixture'))
            if si==1:ax.set_xlabel('Expected budget per prompt b (log scale)')
            if mi==0:ax.set_ylabel('Coverage')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',frameon=False,bbox_to_anchor=(.5,.015))
    fig.suptitle('Learning C changes the inverse-censoring weights',fontsize=17,fontweight='bold',y=.99)
    fig.text(.5,.93,'T model and observed calibration data held fixed • independent C-only pilot • no inherited known-weight guarantee',ha='center',color='#5A6774')
    fig.subplots_adjust(top=.86,bottom=.15,hspace=.32,wspace=.20)
    save(fig,out/'learned_c_comparison')
    # Group-level behavior: marginal coverage does not imply subgroup coverage.
    fig,axs=plt.subplots(1,2,figsize=(12,4.9),sharey=True)
    for ax,family in zip(axs,['DNN','RF']):
        data=known[(known.scenario=='hidden_intent_mixture')&(known.model==family)&(known.method=='Optimized')]
        for metric,label,color,ls in [('coverage','Overall','#009E73','-'),('coverage_group_0','Majority language','#0072B2','--'),('coverage_group_1','Minority language','#D55E00','-')]:
            series(ax,data,metric,color,label,ls)
        ax.axhline(.9,ls=':',color='#444444');ax.set_xscale('log');ax.grid(True);ax.set_ylim(.76,1.005)
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_title(family);ax.set_xlabel('Expected budget per prompt b (log scale)')
    axs[0].set_ylabel('Coverage');handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='lower center',frameon=False,bbox_to_anchor=(.5,0))
    fig.suptitle('Harder scenario: overall validity can hide a subgroup gap',fontsize=17,fontweight='bold',y=1.01)
    fig.text(.5,.91,'Optimized method, known C • minority-language flag changes hidden-intent prevalence and unsafe risk',ha='center',fontsize=10,color='#5A6774')
    fig.subplots_adjust(top=.81,bottom=.22,wspace=.16);save(fig,out/'subgroup_coverage')
    # Informative capped methods, expanded LPB panel (unsafe large baselines omitted explicitly).
    fig,axs=plt.subplots(1,2,figsize=(12.5,4.8))
    for ax,sc in zip(axs,LABELS):
        for family,color in [('DNN','#0072B2'),('RF','#D55E00')]:
            data=known[(known.scenario==sc)&(known.model==family)&(known.method=='Optimized')]
            series(ax,data,'mean_lpb',color,family)
        ax.set_title('Paper-v2' if sc=='paper' else 'Hidden-intent mixture');ax.set_xlabel('Expected budget per prompt b');ax.grid(True);ax.set_ylabel('Mean lower prediction bound L');ax.legend(frameon=False)
    fig.suptitle('Optimized bounds become more informative as the cap increases',fontsize=16,fontweight='bold',y=1.01)
    fig.subplots_adjust(top=.83,bottom=.15,wspace=.25);save(fig,out/'optimized_lpb')
    # Audit nominal budget vs actual calls with early stopping.
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for ax,sc in zip(axs,LABELS):
        for family,color in [('DNN','#0072B2'),('RF','#D55E00')]:
            data=known[(known.scenario==sc)&(known.model==family)&(known.method=='Optimized')]
            series(ax,data,'mean_nominal_c',color,f'{family}: assigned C','--',False)
            series(ax,data,'mean_actual_calls',color,f'{family}: min(T,C)','-',False)
        ax.plot([10,10000],[10,10000],ls=':',c='#777777',lw=1);ax.grid(True);ax.set_title('Paper-v2' if sc=='paper' else 'Hidden-intent mixture');ax.set_xlabel('Expected budget per prompt b');ax.set_ylabel('Outputs / calibration prompt');ax.legend(fontsize=9,frameon=False)
    fig.suptitle('Assigned budget and early-stop generation cost are different',fontsize=16,fontweight='bold',y=1.01)
    fig.subplots_adjust(top=.83,bottom=.15,wspace=.25);save(fig,out/'sampling_cost')
    # Compact representative table at b=1000, includes every method and both C sources.
    summary=pd.read_csv(results/'summary.csv')
    keep=['scenario','model','method','c_source','budget','coverage_mean','coverage_std','mean_lpb_mean','mean_nominal_c_mean','mean_actual_calls_mean','coverage_group_1_mean']
    summary.loc[summary.budget==1000,keep].to_csv(out/'results_at_budget_1000.csv',index=False)
    print('Saved figures to',out)

if __name__=='__main__':main()
