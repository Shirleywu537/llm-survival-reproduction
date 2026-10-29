"""Synthetic prompts, generation labels, and exact evaluation distributions."""
from dataclasses import dataclass
import numpy as np
from core import geometric_quantile,geometric_coverage

@dataclass
class Distribution:
    x: np.ndarray
    p_low: np.ndarray
    p_high: np.ndarray
    mixing: np.ndarray
    group: np.ndarray
    description: str

    def subset(self,idx):
        return Distribution(self.x[idx],self.p_low[idx],self.p_high[idx],self.mixing[idx],self.group[idx],self.description)
    def sample_propensities(self,rng):
        return np.where(rng.random(len(self.x))<self.mixing,self.p_high,self.p_low)
    def counts(self,rng,n):
        return rng.binomial(n,self.sample_propensities(rng))
    def sample_t(self,rng):
        return rng.geometric(self.sample_propensities(rng))
    def mean_p(self):
        return (1-self.mixing)*self.p_low+self.mixing*self.p_high
    def coverage(self,l):
        return (1-self.mixing)*geometric_coverage(self.p_low,l)+self.mixing*geometric_coverage(self.p_high,l)
    def oracle_lpb(self,alpha):
        # Largest integer l with P(T<l|X)<alpha (continuous-parameter convention).
        lo=np.ones(len(self.x),dtype=np.int64)
        hi=np.ceil(-np.log(alpha)/self.p_low).astype(np.int64)+2
        while np.any(hi-lo>1):
            mid=(lo+hi)//2
            good=(1-self.coverage(mid))<alpha
            lo=np.where(good,mid,lo);hi=np.where(good,hi,mid)
        return lo

def make_paper(n,rng,normalizer=None):
    rare=rng.random(n)<.1
    p=10**np.where(rare,rng.uniform(-6,-5,n),rng.uniform(-4,-3,n))
    taus=np.linspace(.1,.9,10)
    mu=np.stack([geometric_quantile(p,t)**.25 for t in taus],axis=1)
    if normalizer is None: normalizer=mu.mean()
    x=mu/normalizer+rng.normal(0,.1,(n,10))
    return Distribution(x,p,p,np.zeros(n),rare.astype(int),'Paper v2 Gaussian features'),normalizer,mu

def make_hard(n,rng):
    # Observed features are prompt summaries, not a complete representation.
    # x0 sensitive-topic score, x1 obfuscation score, x2 context-length score,
    # x3 safety-framing score, x4 minority-language flag, x5 topic interaction,
    # x6..9 nuisance embedding coordinates. Scores are standardized proxies.
    z=rng.uniform(-1,1,(n,5));g=(rng.random(n)<.1).astype(int)
    x=np.column_stack([z[:,:4],g,z[:,4],rng.normal(size=(n,4))])
    interaction=(z[:,0]>.35)&(z[:,1]>.35)
    low_log=-5.25+.55*z[:,0]+.30*z[:,2]-.30*z[:,3]+.20*np.sin(np.pi*z[:,4])
    high_log=-3.15+.45*z[:,0]+.35*z[:,1]+.30*g+.30*interaction+.20*np.sin(np.pi*z[:,2]*z[:,4])
    pl=np.clip(10**low_log,1e-7,2e-4)
    ph=np.clip(10**high_log,1e-4,.025)
    # Hidden unsafe-intent mode remains fixed during one prompt's generations.
    # Its prevalence rises for the minority language and combined risk cues.
    w=np.clip(.015+.09*g+.16*interaction+.025*(z[:,2]>.5),.005,.40)
    return Distribution(x,pl,ph,w,g,'Hidden-intent mixture with rare subgroup and nonlinear interactions')

def generate(name,n,seed):
    rng=np.random.default_rng(seed)
    if name=='paper':
        # Split identities before computing normalization; only training mu used.
        rare=rng.random(n)<.1
        p=10**np.where(rare,rng.uniform(-6,-5,n),rng.uniform(-4,-3,n))
        mu=np.stack([geometric_quantile(p,t)**.25 for t in np.linspace(.1,.9,10)],1)
        normalizer=mu[:int(.45*n)].mean()
        x=mu/normalizer+rng.normal(0,.1,(n,10))
        d=Distribution(x,p,p,np.zeros(n),rare.astype(int),'Paper v2 Gaussian features')
        return d,normalizer
    return make_hard(n,rng),None

def auxiliary(name,n,seed,normalizer):
    rng=np.random.default_rng(seed)
    return make_paper(n,rng,normalizer)[0] if name=='paper' else make_hard(n,rng)
