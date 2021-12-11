################# Confidence Frequency Matrix Factorization #################
#                   Weighted Rating Matrix  Factorization                   #
# this model refers to the following paper:
# Yifan Hu et al.Collaborative Filtering for Implicit Feedback Datasets
from base.iterativeRecommender import IterativeRecommender
from scipy.sparse import *
from scipy import *
import numpy as np
class WRMF(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(WRMF, self).__init__(conf,trainingSet,testSet,fold)
    def initModel(self):
        super(WRMF, self).initModel()
        self.X=self.P*10
        self.Y=self.Q*10

    def trainModel(self):
        print('training...')
        epoch = 0
        while epoch < self.maxEpoch:
            self.loss = 0
            YtY = self.Y.T.dot(self.Y)
            for user in self.data.user:
                #C_u = np.ones(self.data.getSize(self.recType))
                H = np.ones(self.num_items)
                val = []
                pos = []
                P_u = np.zeros(self.num_items)
                uid = self.data.user[user]
                for item in self.data.trainSet_u[user]:
                    iid = self.data.item[item]
                    r_ui = float(self.data.trainSet_u[user][item])
                    pos.append(iid)
                    val.append(10*r_ui)
                    H[iid]+=10*r_ui
                    P_u[iid]=1
                    error = (P_u[iid]-self.X[uid].dot(self.Y[iid]))
                    self.loss+=pow(error,2)
                #sparse matrix
                C_u = coo_matrix((val,(pos,pos)),shape=(self.num_items,self.num_items))
                A = (YtY + np.dot(self.Y.T,C_u.dot(self.Y)) + self.regU * np.eye(self.emb_size))
                self.X[uid] = np.dot(np.linalg.inv(A),(self.Y.T*H).dot(P_u))

            XtX = self.X.T.dot(self.X)
            for item in self.data.item:
                P_i = np.zeros(self.num_users)
                iid = self.data.item[item]
                H = np.ones(self.num_users)
                val = []
                pos = []
                for user in self.data.trainSet_i[item]:
                    uid = self.data.user[user]
                    r_ui = float(self.data.trainSet_i[item][user])
                    pos.append(uid)
                    val.append(10*r_ui)
                    H[uid] += 10*r_ui
                    P_i[uid] = 1
                # sparse matrix
                C_i = coo_matrix((val, (pos, pos)),shape=(self.num_users,self.num_users))
                A = (XtX + np.dot(self.X.T,C_i.dot(self.X)) + self.regU * np.eye(self.emb_size))
                self.Y[iid]=np.dot(np.linalg.inv(A), (self.X.T*H).dot(P_i))

            #self.loss += (self.X * self.X).sum() + (self.Y * self.Y).sum()
            epoch += 1
            print('epoch:',epoch,'loss:',self.loss)
            if self.isConverged(epoch):
                break

    def predictForRanking(self,u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.Y.dot(self.X[u])
        else:
            return [self.data.globalMean] * self.num_items