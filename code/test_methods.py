import numpy as np
from scipy.stats import geom
from scipy.optimize import minimize
from core import *
from data import make_hard

def test_integer_conventions():
    p=np.array([.001,.05,.5,.9]);tau=np.array([.001,.1,.5,.9])
    np.testing.assert_allclose(geometric_quantile(p,tau),geom.ppf(tau,p))
    np.testing.assert_allclose(geometric_coverage(p,[1,2,3,4]),geom.sf(np.array([1,2,3,4])-1,p))
    assert geometric_quantile(.5,.5)==1
    assert geometric_quantile(.5,.500001)==2
    assert geometric_coverage(.5,0)==1

def test_optimal_budget_and_solver():
    c=np.array([1.,3.,10.,30.,100.]);b=22.
    pi,a=optimal_probabilities(c,b)
    assert abs(c@pi-b)<1e-9
    ref=minimize(lambda x:np.sum(1/x),np.full(5,b/c.sum()),jac=lambda x:-1/x**2,bounds=[(1e-6,1)]*5,
                 constraints={'type':'ineq','fun':lambda x:b-c@x,'jac':lambda x:-c},method='SLSQP',options={'ftol':1e-8,'maxiter':1000})
    assert ref.success
    np.testing.assert_allclose(np.sum(1/pi),ref.fun,rtol=1e-7)
    assert np.all(optimal_probabilities(c,1e6)[0]==1)

def test_trimmed_weight_bound_and_objective():
    rng=np.random.default_rng(7);p=10**rng.uniform(-7,-2,300)
    for b in [10,1000,10000]:
        tri=design('Trimmed',p,b,.56,10);opt=design('Optimized',p,b,.56,10)
        assert max(1/tri.pi)<=10+1e-9
        assert max(1/opt.pi)<=10+1e-9
        assert opt.cost@opt.pi<=b*len(p)+1e-6
        assert np.mean(1/opt.pi)<=np.mean(1/tri.pi)+1e-9

def test_event_calibration_matches_bruteforce():
    rng=np.random.default_rng(3);n=1000;p=10**rng.uniform(-4,-1,n)
    grid=np.r_[0.,np.geomspace(1e-5,.56,200)];q=geometric_quantile(p[None,:],grid[:,None])
    for method in ['Basic','Trimmed','Optimized']:
        d=design(method,p,100,.56,10);c=d.cost*(rng.random(n)<d.pi)
        y=np.minimum(rng.geometric(p),c);f=np.minimum(q,d.cap)
        errors=np.mean((f<=c[None,:])*(y[None,:]<f)/d.pi[None,:],axis=1)
        np.testing.assert_equal(calibrate_adaptive(p,y,c,d.pi,grid,.1),prefix_grid_choice(errors,grid,.1))

def test_prefix_stops_at_first_crossing():
    assert prefix_grid_choice([0,.05,.2,.01],[0,.1,.2,.3],.1)==.1

def test_ipw_unbiased_at_fixed_quantile():
    # Independent Monte Carlo, checking an estimator rather than its implementation.
    rng=np.random.default_rng(5);n=300000;p=np.full(n,.01);q=np.full((1,n),11.)
    c=rng.geometric(.02,n);t=rng.geometric(p);y=np.minimum(t,c)
    _,err=calibrate_naive(q,y,c,naive_weight_matrix(q,50),np.array([.1]),.1,True)
    assert abs(err[0]-(1-.99**10))<.002

def test_mixture_survival_simulation_and_oracle():
    d=make_hard(3000,np.random.default_rng(7));l=np.full(3000,1000.)
    empirical=np.mean([(d.sample_t(np.random.default_rng(k))>=l).mean() for k in range(100)])
    assert abs(empirical-d.coverage(l).mean())<.002
    oracle=d.oracle_lpb(.1)
    assert np.all(d.coverage(oracle)>=.9-1e-12)
    assert np.all(d.coverage(oracle+1)<=.9+1e-12)
