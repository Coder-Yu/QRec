from baseclass.IterativeRecommender import IterativeRecommender
from scipy.sparse import *
from scipy import *
import numpy as np
from numpy import linalg as LA
from joblib import Parallel, delayed
from math import sqrt

EPS = 1e-8
# this algorithm refers to the following paper:
# #########----  Collaborative Filtering with Social Exposure: A Modular Approach to Social Recommendation   ----#############
# MERec_boost

class MERec(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(MERec, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(MERec, self).initModel()
        self.lam_theta = 1e-5
        self.lam_beta = 1e-5
        self.lam_y = 1.0
        self.init_mu = 0.05
        self.a = 1.0
        self.b = 20.0
        self.s= 5
        self.init_std = 0.1
        self.theta = self.init_std * \
            np.random.randn(self.m, self.k).astype(np.float32)
        self.beta = self.init_std * \
            np.random.randn(self.n, self.k).astype(np.float32)
        self.mu = self.init_mu * np.ones((self.m,self.n), dtype=np.float32)
        self.n_jobs=4
        self.batch_size=300
        self.loadMetaInfo()
        row,col,val = [],[],[]
        for user in self.dao.trainSet_u:
            for item in self.dao.trainSet_u[user]:
                u = self.dao.user[user]
                i = self.dao.item[item]
                row.append(u)
                col.append(i)
                val.append(1)

        self.X = csr_matrix((np.array(val),(np.array(row),np.array(col))),(self.m,self.n))

        row,col,val = [],[],[]
        for user in self.user2art:
            for a in self.user2art[user]:
                u = self.dao.user[user]
                i = self.id['artist'][a]
                row.append(u)
                col.append(i)
                val.append(1)
        self.U2A = csr_matrix((np.array(val), (np.array(row), np.array(col))), (self.m, len(self.id['artist'])))

        row,col,val = [],[],[]
        for artist in self.art2song:
            for item in self.art2song[artist]:
                a = self.id['artist'][artist]
                i = self.dao.item[item]
                row.append(a)
                col.append(i)
                val.append(1.0/len(self.art2song[artist]))
        self.A2S = csr_matrix((np.array(val), (np.array(row), np.array(col))), (len(self.id['artist']),self.n))

        row,col,val = [],[],[]
        for user in self.user2album:
            for a in self.user2album[user]:
                u = self.dao.user[user]
                i = self.id['album'][a]
                row.append(u)
                col.append(i)
                val.append(1)
        self.U2Al = csr_matrix((np.array(val), (np.array(row), np.array(col))), (self.m, len(self.id['album'])))

        row,col,val = [],[],[]
        for album in self.album2song:
            for item in self.album2song[album]:
                a = self.id['album'][album]
                i = self.dao.item[item]
                row.append(a)
                col.append(i)
                val.append(1.0/len(self.album2song[album]))
        self.Al2S = csr_matrix((np.array(val), (np.array(row), np.array(col))), (len(self.id['album']),self.n))

        self.commuteMatrix_a = np.array(self.U2A.dot(self.A2S).todense())
        self.commuteMatrix_al = np.array(self.U2Al.dot(self.Al2S).todense())

    def loadMetaInfo(self):
        from collections import defaultdict
        self.user2art = defaultdict(dict)
        self.art2song = defaultdict(dict)
        self.user2album = defaultdict(dict)
        self.album2song = defaultdict(dict)
        self.id=defaultdict(dict)
        with open(self.config['ratings']) as f:
            for line in f:
                items = line.strip().split(',')
                if self.dao.trainSet_u.has_key(items[0]) and self.dao.trainSet_u[items[0]].has_key(items[1]):
                    self.art2song[items[2]][items[1]]=1
                    self.user2art[items[0]][items[2]]=1
                    self.album2song[items[3]][items[1]]=1
                    self.user2album[items[0]][items[3]]=1
                    if items[2] not in self.id['artist']:
                        self.id['artist'][items[2]]=len(self.id['artist'])
                    if items[3] not in self.id['album']:
                        self.id['album'][items[3]] = len(self.id['album'])


    def buildModel(self):
        print 'training...'
        iteration = 0

        self._update(self.X)

    def _update(self, X):
        '''Model training and evaluation on validation set'''
        n_users = X.shape[0]
        XT = X.T.tocsr()  # pre-compute this
        self.vad_ndcg = -np.inf
        for i in xrange(self.maxIter):

            print 'ITERATION #%d' % i
            self._update_factors(X, XT)
            print self.mu
            self._update_expo(X, n_users)
            self.ranking_performance()


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
        print '\tUpdating exposure prior...'

        start_idx = range(0, n_users, self.batch_size)
        end_idx = start_idx[1:] + [n_users]

        A_sum = np.zeros(self.n)
        for lo, hi in zip(start_idx, end_idx):
            A_sum += a_row_batch(X[lo:hi], self.theta[lo:hi], self.beta,
                                 self.lam_y, self.mu[lo:hi]).sum(axis=0)

        A_sum=np.tile(A_sum,[self.m,1])
        S_sum = 1*self.commuteMatrix_a*A_sum
        S_sum += 0.2*self.commuteMatrix_al*A_sum
        print  A_sum +self.s*S_sum
        self.mu = (self.a + A_sum +self.s*S_sum- 1) / (self.a + self.b + self.s*S_sum+n_users - 2)


    def predictForRanking(self,u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            u = self.dao.getUserId(u)
            return self.beta.dot(self.theta[u])
        else:
            return [self.dao.globalMean] * len(self.dao.item)

# Utility functions #



def get_row(Y, i):
    '''Given a scipy.sparse.csr_matrix Y, get the values and indices of the
    non-zero values in i_th row'''
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return Y.data[lo:hi], Y.indices[lo:hi]

def a_row_batch(Y_batch, theta_batch, beta, lam_y, mu):
    '''Compute the posterior of exposure latent variables A by batch'''
    pEX = sqrt(lam_y / 2 * np.pi) * \
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
    for ib, k in enumerate(xrange(lo, hi)):
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

    start_idx = range(0, m, batch_size)
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs=n_jobs)(delayed(_solve_batch)(
        lo, hi, X, X_old[lo:hi], Y, m, f, lam, lam_y, mu)
                                   for lo, hi in zip(start_idx, end_idx))
    # res = []
    # for lo, hi in zip(start_idx, end_idx):
    #     res.append(_solve_batch(lo, hi, X, X_old[lo:hi], Y, m, f, lam, lam_y, mu))

    X_new = np.vstack(res)
    return X_new

