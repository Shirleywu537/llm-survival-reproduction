import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_results import COLORS,save
ap=argparse.ArgumentParser();ap.add_argument('--results',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
df=pd.read_csv(Path(a.results)/'runs.csv');idx=['scenario','model','method','budget','repeat']
k=df[df.c_source=='known'].set_index(idx);e=df[df.c_source=='learned'].set_index(idx)
d=(100*(e.coverage-k.coverage)).dropna().rename('delta').reset_index()
fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True)
for si,sc in enumerate(['paper','hidden_intent_mixture']):
 for mi,family in enumerate(['DNN','RF']):
  ax=axs[si,mi]
  for method in ['Naive','Basic','Trimmed','Optimized']:
   t=d[(d.scenario==sc)&(d.model==family)&(d.method==method)].groupby('budget').delta.agg(['mean','std'])
   ax.plot(t.index,t['mean'],c=COLORS[method],label=method,lw=2,marker='o',ms=3)
   ax.fill_between(t.index,t['mean']-t['std'],t['mean']+t['std'],color=COLORS[method],alpha=.10,lw=0)
  ax.axhline(0,c='#444444',ls=':');ax.set_xscale('log');ax.grid(True)
  ax.set_title(f'{family} | '+('Paper-v2' if si==0 else 'Hidden-intent mixture'))
  if mi==0:ax.set_ylabel('Learned minus known coverage (pp)')
  if si==1:ax.set_xlabel('Expected budget per prompt b (log scale)')
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,0),frameon=False)
fig.suptitle('Estimated C weights can shift coverage in either direction',fontsize=17,fontweight='bold',y=.99)
fig.text(.5,.93,'Paired difference on identical calibration observations • bands: ±1 SD • positive values mean higher coverage',ha='center',fontsize=10,color='#5A6774')
fig.subplots_adjust(top=.86,bottom=.15,hspace=.34,wspace=.25);save(fig,Path(a.out)/'learned_c_effect')
