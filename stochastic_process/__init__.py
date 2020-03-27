# -*- coding: utf-8 -*-
# First version copied from hecmtl/third_party/longstaff_schwartz/stochastic_process.py
import numpy as np
from scipy import stats
from scipy.stats.distributions import norm, lognorm, rv_frozen

def get_dt(t: np.array, n: int, rnd: np.random.RandomState) -> np.array:
    assert t.ndim == 1, 'One dimensional time vector required'
    assert t.size > 0, 'At least one time point is required'
    dt = np.concatenate((t[0:1], np.diff(t)))
    assert (dt >= 0).all(), 'Increasing time vector required'
    return dt

class BrownianMotion:
    '''Brownian Motion (Wiener Process) with optional drift.'''
    def __init__(self, mu: float=0.0, sigma: float=1.0):
        self.mu = mu
        self.sigma = sigma

    def simulate(self, t: np.array, n: int, rnd: np.random.RandomState) -> np.array:
        dt = get_dt(t, n, rnd)

        # transposed simulation for automatic broadcasting
        W = rnd.normal(size=(n, t.size))
        W_drift = (W * np.sqrt(dt) * self.sigma + self.mu * dt).T
        return np.cumsum(W_drift, axis=0)

    def distribution(self, t: float) -> rv_frozen:
        return norm(self.mu * t, self.sigma * np.sqrt(t))


class GeometricBrownianMotion:
    '''Geometric Brownian Motion'''
    def __init__(self, mu: float=0.0, sigma: float=1.0):
        self.mu = mu
        self.sigma = sigma

    def simulate(self, t: np.array, n: int, rnd: np.random.RandomState) -> np.array:
        dt = get_dt(t, n, rnd)

        # transposed simulation for automatic broadcasting
        dW = (rnd.normal(size=(t.size, n)).T * np.sqrt(dt)).T
        W = np.cumsum(dW, axis=0)
        return np.exp(self.sigma * W.T + (self.mu - self.sigma**2 / 2) * t).T

    def distribution(self, t: float) -> rv_frozen:
        mu_t = (self.mu - self.sigma**2/2) * t
        sigma_t = self.sigma * np.sqrt(t)
        return lognorm(scale=np.exp(mu_t), s=sigma_t)


class GARCH:
    '''Geometric Brownian Motion.(with optional drift).'''
    def __init__(self, vol: float=0.20, alpha: float=0.05, beta: float=0.945):
        '''The value for omega yields the target annualized VOL given the other parameters'''
        pi = alpha + beta        
        self.omega = vol**2/252 * (1-pi)
        self.alpha = alpha
        self.beta  = beta

    def simulate(self, t: np.array, n: int, rnd: np.random.RandomState) -> np.array:
        dt = get_dt(t, n, rnd)

        pi = self.alpha+self.beta
        sig2 = self.omega / (1-pi)
        print('Persistence:',pi)
        print('Uncond. Vol:',np.sqrt(252*sig2))
                
        # transposed simulation for automatic broadcasting
        eps = rnd.normal(size=(t.size, n))
        R = np.full((t.size, n), np.nan)
        ht = np.full((t.size+1, n), np.nan)
        ht[0] = sig2 # fills the whole row
        for tn in range(0,t.size):
            R[tn] = np.sqrt(ht[tn])*eps[tn]
            ht[tn+1] = self.omega + self.alpha*ht[tn]*(eps[tn]**2) + self.beta*ht[tn]

        R_T = np.cumsum(R, axis=0)
        print('R_T mean=',R_T.mean(),'and std=',R_T.std())
        return np.exp( R_T )

    def distribution(self, t: float) -> rv_frozen:
        warning.warn("")
        return None    

def sim_garch():
    if False:
        import stochastic_process as sp
        from importlib import reload  
        sp = reload(sp); S = sp.sim_garch()
        
    import matplotlib.pyplot as plt
    plt.ion()

    t = np.linspace(0, 1, 252)
    n = 500000  # number of simulated paths
    rnd = np.random.RandomState() # 60206

    garch = GARCH(alpha=0.06, beta=0.935)
    S = garch.simulate(t,n,rnd)
    R = np.log(S[-1])
    print( stats.describe(R) )
    plt.hist(R, bins=100)
    plt.show()

    return S

    # plt.tight_layout()
