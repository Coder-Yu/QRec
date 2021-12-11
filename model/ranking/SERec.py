from base.socialRecommender import SocialRecommender
from scipy.sparse import *
import numpy as np
from numpy import linalg as LA
from joblib import Parallel, delayed
from math import sqrt
EPS = 1e-8
# this model refers to the following paper:
# #########----  Collaborative Filtering with Social Exposure: A Modular Approach to Social Recommendation   ----#############
# SEREC_boost
class SERec(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=None,fold='[1]'):
        super(SERec, self).__init__(conf,trainingSet,testSet,relation,fold)

    def initModel(self):
        super(SERec, self).initModel()
        self.lam_theta = 1e-5
        self.lam_beta = 1e-5
        self.lam_y = 0.01
        self.init_mu = 0.01
        self.a = 1.0
        self.b = 99.0
        self.s= 2.2
        self.init_std = 0.5
        self.theta = self.init_std * \
            np.random.randn(self.num_users, self.emb_size).astype(np.float32)
        self.beta = self.init_std * \
            np.random.randn(self.num_items, self.emb_size).astype(np.float32)
        self.mu = self.init_mu * np.ones((self.num_users,self.num_items), dtype=np.float32)
        self.n_jobs=4
        self.batch_size=1000
        row,col,val = [],[],[]
        for user in self.data.trainSet_u:
            for item in self.data.trainSet_u[user]:
                u = self.data.user[user]
                i = self.data.item[item]
                row.append(u)
                col.append(i)
                val.append(1)

        self.X = csr_matrix((np.array(val),(np.array(row),np.array(col))),(self.num_users,self.num_items))
        row,col,val = [],[],[]
        for user in self.social.followees:
            for f in self.social.followees[user]:
                u = self.data.user[user]
                i = self.data.user[f]
                row.append(u)
                col.append(i)
                val.append(1)
        self.T = csr_matrix((np.array(val), (np.array(row), np.array(col))), (self.num_users, self.num_users))

    def trainModel(self):
        print('training...')
        self._update(self.X)

    def _update(self, X):
        '''Model training and evaluation on validation set'''
        n_users = X.shape[0]
        XT = X.T.tocsr()  # pre-compute this
        self.vad_ndcg = -np.inf
        for i in range(self.maxEpoch):

            print('epoch #%d' % i)
            self._update_factors(X, XT)
            print(self.mu)
            self._update_expo(X, n_users)

    def _update_factors(self, X, XT):
        '''Update user and item collaborative factors with ALS'''
        self.theta = recompute_factors(self.beta, self.theta, X,
                                       self.lam_theta / self.lam_y,
                                       self.lam_y,
                                       self.mu,
                                       self.n_jobs,
                                       batch_size=self.batch_size)

        self.beta = recompute_factors(self.theta, self.beta, XT,
                                      self.lam_beta / self.lam_y,
                                      self.lam_y,
                                      self.mu,
                                      self.n_jobs,
                                      batch_size=self.batch_size)
    def _update_expo(self, X, n_users):
        '''Update exposure prior'''
        print('\tUpdating exposure prior...')
        start_idx = list(range(0, n_users, self.batch_size))
        end_idx = start_idx[1:] + [n_users]
        A_sum = np.zeros(self.num_items)
        for lo, hi in zip(start_idx, end_idx):
            A_sum += a_row_batch(X[lo:hi], self.theta[lo:hi], self.beta,
                                 self.lam_y, self.mu[lo:hi]).sum(axis=0)
        A_sum=np.tile(A_sum,[self.num_users,1])
        S_sum = self.T.dot(A_sum)
        self.mu = (self.a + A_sum +(self.s-1)*S_sum- 1) / (self.a + self.b + (self.s-1)*S_sum+n_users - 2)

    def predictForRanking(self,u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.beta.dot(self.theta[u])
        else:
            return [self.data.globalMean] * self.num_items

def get_row(Y, i):
    '''Given a scipy.sparse.csr_matrix Y, get the values and indices of the
    non-zero values in i_th row'''
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return Y.data[lo:hi], Y.indices[lo:hi]

def a_row_batch(Y_batch, theta_batch, beta, lam_y, mu):
    '''Compute the posterior of exposure latent variables A by batch'''
    pEX = sqrt(lam_y / 2 / np.pi) * \
          np.exp(-lam_y * theta_batch.dot(beta.T) ** 2 / 2)

    A = (pEX + EPS) / (pEX + EPS + (1 - mu) / mu)
    A[Y_batch.nonzero()] = 1.
    return A

def _solve(k, A_k, X, Y, f, lam, lam_y, mu):
    '''Update one single factor'''
    s_u, i_u = get_row(Y, k)
    a = np.dot(s_u * A_k[i_u], X[i_u])
    B = X.T.dot(A_k[:, np.newaxis] * X) + lam * np.eye(f)
    return LA.solve(B, a)

def _solve_batch(lo, hi, X, X_old_batch, Y, m, f, lam, lam_y, mu):
    '''Update factors by batch, will eventually call _solve() on each factor to
    keep the parallel process busy'''
    assert X_old_batch.shape[0] == hi - lo

    if mu.shape[1] == X.shape[0]:  # update users
        A_batch = a_row_batch(Y[lo:hi], X_old_batch, X, lam_y, mu[lo:hi])
    else:  # update items
        A_batch = a_row_batch(Y[lo:hi], X_old_batch, X, lam_y, mu.T[lo:hi])

    X_batch = np.empty_like(X_old_batch, dtype=X_old_batch.dtype)
    for ib, k in enumerate(range(lo, hi)):
        X_batch[ib] = _solve(k, A_batch[ib], X, Y, f, lam, lam_y, mu)
    return X_batch

def recompute_factors(X, X_old, Y, lam, lam_y, mu, n_jobs, batch_size=1000):
    '''Regress X to Y with exposure matrix (computed on-the-fly with X_old) and
    ridge term lam by embarrassingly parallelization. All the comments below
    are in the view of computing user factors'''
    m, n = Y.shape  # m = number of users, n = number of items
    assert X.shape[0] == n
    assert X_old.shape[0] == m
    f = X.shape[1]  # f = number of factors

    start_idx = list(range(0, m, batch_size))
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs=n_jobs)(delayed(_solve_batch)(
        lo, hi, X, X_old[lo:hi], Y, m, f, lam, lam_y, mu)
                                   for lo, hi in zip(start_idx, end_idx))
    X_new = np.vstack(res)
    return X_new

