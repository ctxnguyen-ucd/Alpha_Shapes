import numpy as np
import quantecon as qe
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.special import kl_div, logsumexp
from tqdm import tqdm
class log_kde():
    def __init__(self, P, t):
        self.I = np.eye(P.shape[0])
        self.P = P
        self.Pt = expm(-t * (self.I - P))
        self.pi = qe.MarkovChain(P).stationary_distributions
        self.logPt = np.log(self.Pt)

    def calculate_psi_p(self, p) -> NDArray:
        # TODO: multiply steady state i within sum
        log_kde_sum = np.log(sum([pi * np.exp(-kl_div(p, row).sum()) for row, pi in zip(self.Pt, self.pi.flatten())])) # need to sum kl_div, returns array
        return log_kde_sum

    def calculate_psi_c(self, p):
        psi_p = self.calculate_psi_p(p) # log coodinates
        q = self.calculate_q(p) # q = T(p_i)
        psi_c = psi_p + np.sum(kl_div(p, q))
        return psi_c

    def calculate_ti(self, p_i):
        # construct t_i: using KL divergence in a different way
        scores = self.logPt @ p_i + np.log(self.pi)        # (n,)
        t = np.exp(scores - logsumexp(scores))             
        return t.flatten()

    def calculate_r(self, p_i):
        t = self.calculate_ti(p_i)
        r = np.zeros(shape=self.Pt[0].shape)
        # construct r
        for t_i, log_pt_i in zip(t, self.logPt):
            r_i = t_i * log_pt_i             # scalar * log(array) = array
            r += r_i
        return r

    def calculate_q(self, p_i):
        r = self.calculate_r(p_i)
        q_tilde = np.exp(r)
        q = q_tilde / q_tilde.sum()
        return q
    
    def calculate_R_mat(self):
        R = []
        for p_i in tqdm(self.Pt):
            r = self.calculate_r(p_i)
            R.append(r)
        return np.vstack(R)
    
    def calculate_Q_mat(self):
        Q = []
        for p_i in tqdm(self.Pt):
            q = self.calculate_q(p_i)
            Q.append(q)
        return np.vstack(Q)

    