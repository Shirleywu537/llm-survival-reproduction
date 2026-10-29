"""Conditional geometric T models, and learned randomized C distributions."""
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

class Net(nn.Module):
    def __init__(self, d, widths):
        super().__init__(); layers=[]
        for width in widths: layers += [nn.Linear(d,width),nn.ReLU()];d=width
        layers += [nn.Linear(d,1)];self.layers=nn.Sequential(*layers)
    def forward(self,x): return self.layers(x).squeeze(-1)

class ConditionalModel:
    def __init__(self, family, mode, cfg, seed):
        self.family,self.mode,self.cfg,self.seed=family,mode,cfg,seed
        self.scaler=None;self.history=[]
    def fit(self,x,y,budgets=None):
        c=self.cfg
        if self.family=='RF':
            is_c=self.mode!='T'
            self.net=RandomForestRegressor(n_estimators=c['c_rf_trees'] if is_c else c['rf_trees'],
                min_samples_leaf=c['c_rf_min_leaf'] if is_c else c['rf_min_leaf'],
                n_jobs=c['threads'],random_state=self.seed)
            target=(y-1)/(budgets-1) if self.mode=='C_geom' else y
            self.net.fit(x,target);return self
        torch.set_num_threads(c['threads']);torch.manual_seed(self.seed)
        if self.mode!='T':
            self.scaler=StandardScaler().fit(x);x=self.scaler.transform(x)
        self.net=Net(x.shape[1],c['dnn_hidden'])
        if self.mode=='C_geom':
            # Residual log mean relative to a recorded budget, initially no residual.
            nn.init.zeros_(self.net.layers[-1].weight);nn.init.zeros_(self.net.layers[-1].bias)
        elif self.mode=='C_ber':
            p=np.clip(np.mean(y),.001,.999)
            nn.init.constant_(self.net.layers[-1].bias,float(np.log(p/(1-p))))
        is_c=self.mode!='T'
        opt=torch.optim.AdamW(self.net.parameters(),lr=c['c_dnn_lr'] if is_c else c['dnn_lr'],weight_decay=c['dnn_weight_decay'])
        xt=torch.tensor(x,dtype=torch.float32);yt=torch.tensor(y,dtype=torch.float32)
        bt=None if budgets is None else torch.tensor(budgets,dtype=torch.float32)
        epochs=c['c_dnn_epochs'] if is_c else c['dnn_epochs']
        batch=c['c_dnn_batch_size'] if is_c else c['dnn_batch_size']
        self.net.train()
        for epoch in range(epochs):
            perm=torch.randperm(len(x));total=0.
            for idx in perm.split(batch):
                pred=self.net(xt[idx])
                if self.mode=='C_geom':
                    logits=-torch.log(bt[idx]-1)-torch.clamp(pred,-8,8)
                    loss=(nn.functional.softplus(-logits)+(yt[idx]-1)*nn.functional.softplus(logits)).mean()
                else: loss=nn.functional.binary_cross_entropy_with_logits(pred,yt[idx])
                opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=loss.item()*len(idx)
            self.history.append(total/len(x))
        self.net.eval();return self
    def predict(self,x,budgets=None):
        c=self.cfg
        if self.family=='RF':
            v=self.net.predict(x)
            if self.mode=='C_geom': v=1/(1+(budgets-1)*np.maximum(v,1e-3))
        else:
            if self.scaler is not None:x=self.scaler.transform(x)
            with torch.no_grad():
                raw=np.concatenate([self.net(torch.tensor(block,dtype=torch.float32)).numpy() for block in np.array_split(x,max(1,int(np.ceil(len(x)/8192))))])
            if self.mode=='C_geom': v=1/(1+(budgets-1)*np.exp(np.clip(raw,-8,8)))
            else: v=1/(1+np.exp(-np.clip(raw,-80,80)))
        if self.mode=='T':return np.clip(v,c['probability_floor'],1-c['probability_floor'])
        if self.mode=='C_geom':return np.clip(v,1e-8,1-1e-6)
        return np.clip(v,c['c_probability_floor'],1.)
    def save(self,path):
        if self.family=='RF':joblib.dump(self.net,str(path)+'.joblib',compress=3)
        else:torch.save({'weights':self.net.state_dict(),'family':self.family,'mode':self.mode,'cfg':self.cfg,'scaler':self.scaler,'history':self.history},str(path)+'.pt')

def c_features(x,budget,cost=None):
    # The assigned nonzero limit is known before drawing C; it is an allowed design feature.
    n=len(x);b=np.full(n,budget) if np.isscalar(budget) else np.asarray(budget)
    arr=[x,np.log(b)[:,None]]
    if cost is not None:arr.append(np.log(cost)[:,None])
    return np.column_stack(arr)
