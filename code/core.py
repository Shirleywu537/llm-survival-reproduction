"""Independent implementation of arXiv:2506.13593v2, Sections 4--6.
Integer geometric support is {1,2,...}; coverage is P(T >= L).
Only observed (min(T,C), C) enter calibration. Ground truth is evaluation-only.
"""
from dataclasses import dataclass
import numpy as np

METHODS = ['Uncalibrated', 'Naive', 'Basic', 'Trimmed', 'Optimized']

def geometric_quantile(p, tau):
    p = np.asarray(p, dtype=float)
    z = np.log1p(-np.asarray(tau)) / np.log1p(-p)
    # Avoid an artificial extra generation at exact integer quantile boundaries.
    return np.where(np.asarray(tau)==0, 0., np.maximum(1., np.ceil(z-1e-12*np.maximum(1., np.abs(z)))))

def geometric_coverage(p, lower):
    k = np.maximum(np.ceil(np.asarray(lower))-1., 0.)
    return np.exp(k*np.log1p(-np.asarray(p)))

def optimal_probabilities(cost, budget):
    """Solve Eq. (8) using KKT pi=min(1,a/sqrt(cost)), no CVX dependency."""
    cost=np.asarray(cost,dtype=float)
    if np.any(cost <= 0) or budget <= 0: raise ValueError('Positive costs/budget required')
    if cost.sum() <= budget: return np.ones_like(cost), np.inf
    lo,hi=0.,np.sqrt(cost.max())
    for _ in range(70):
        mid=(lo+hi)/2
        usage=np.sum(cost*np.minimum(1.,mid/np.sqrt(cost)))
        if usage <= budget: lo=mid
        else: hi=mid
    a=lo
    return np.minimum(1.,a/np.sqrt(cost)),a

@dataclass
class Design:
    method: str
    budget: float
    cost: np.ndarray
    pi: np.ndarray
    cap: float
    a: float = 0.

    def for_predictions(self, p, tau_prior):
        c=geometric_quantile(p,tau_prior)
        if self.method in ['Trimmed','Optimized']: c=np.minimum(c,self.cap)
        if self.method=='Optimized': pi=np.minimum(1.,self.a/np.sqrt(c))
        else: pi=np.minimum(1.,self.budget/c)
        return c,pi

def design(method,p,budget,tau_prior,gamma):
    cost=geometric_quantile(p,tau_prior)
    cap=float(np.floor(gamma*budget)) if method in ['Trimmed','Optimized'] else np.inf
    cost=np.minimum(cost,cap)
    if method=='Optimized':
        pi,a=optimal_probabilities(cost,budget*len(cost))
    else:
        pi=np.minimum(1.,budget/cost);a=0.
    return Design(method,budget,cost,pi,cap,a)

def prefix_grid_choice(errors,grid,alpha):
    bad=np.flatnonzero(np.asarray(errors)>alpha)
    return float(grid[max(int(bad[0])-1,0)]) if len(bad) else float(grid[-1])

def calibrate_adaptive(p,observed,censor,selection_probability,grid,alpha):
    """Exact weighted event ordering, projected down to the shared tau grid.
    An event becomes a strict error just above F_hat(T). C=0 never contributes.
    """
    ok=(censor>0)&(observed<censor)
    scores=-np.expm1(observed[ok]*np.log1p(-p[ok]))
    weights=1./selection_probability[ok]
    order=np.argsort(scores,kind='stable')
    scores=scores[order];weights=weights[order]
    cumul=np.cumsum(weights)/len(p)
    hit=np.flatnonzero(cumul>alpha)
    if len(hit)==0: return float(grid[-1])
    threshold=float(scores[hit[0]])
    # At tau=F_hat(T), integer q is T; the strict error has not occurred yet.
    j=np.searchsorted(grid,threshold,side='right')-1
    return float(grid[max(0,j)])

def naive_weight_matrix(q,budget,rate_hat=None):
    rate=np.asarray(1./budget if rate_hat is None else rate_hat)
    log_w=-np.maximum(q-1.,0.)*np.log1p(-rate)
    # exp(700) is only a numerical guard. Impossible selected entries are zeroed
    # by the observation mask. Realized effective weights are audited separately.
    return np.exp(np.minimum(log_w,700.))

def calibrate_naive(q,observed,censor,weights,grid,alpha,return_errors=False):
    mask=(q <= censor[None,:]) & (observed[None,:] < q)
    errors=np.einsum('ij,ij->i',weights,mask,optimize=False)/len(observed)
    tau=prefix_grid_choice(errors,grid,alpha)
    return (tau,errors) if return_errors else tau

def metric_set(lower,dist):
    cov=dist.coverage(lower)
    out={'coverage':float(cov.mean()),'mean_lpb':float(np.mean(lower)),
         'median_lpb':float(np.median(lower)),
         'conditional_coverage_rmse':float(np.sqrt(np.mean((cov-.9)**2)))}
    for g in np.unique(dist.group):
        use=dist.group==g
        out[f'coverage_group_{int(g)}']=float(cov[use].mean())
        out[f'mean_lpb_group_{int(g)}']=float(np.mean(lower[use]))
    return out
