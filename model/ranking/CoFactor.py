from base.iterativeRecommender import IterativeRecommender
import numpy as np
from util import config
from collections import defaultdict
from math import log,exp
from scipy.sparse import *
from scipy import *

class CoFactor(IterativeRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]'):
        super(CoFactor, self).__init__(conf, trainingSet, testSet, fold)

    def readConfiguration(self):
        super(CoFactor, self).readConfiguration()
        extraSettings = config.OptionConf(self.config['CoFactor'])
        self.negCount = int(extraSettings['-k']) #the number of negative samples
        if self.negCount < 1:
            self.negCount = 1
        self.regR = float(extraSettings['-gamma'])
        self.filter = int(extraSettings['-filter'])

    def printAlgorConfig(self):
        super(CoFactor, self).printAlgorConfig()
        print('Specified Arguments of', self.config['model.name'] + ':')
        print('k: %d' % self.negCount)
        print('regR: %.5f' %self.regR)
        print('filter: %d' %self.filter)
        print('=' * 80)

    def initModel(self):
        super(CoFactor, self).initModel()
        #constructing SPPMI matrix
        self.SPPMI = defaultdict(dict)
        print('Constructing SPPMI matrix...')
        #for larger data set has many items, the process will be time consuming
        occurrence = defaultdict(dict)
        i=0
        for item1 in self.data.item:
            i += 1
            if i % 100 == 0:
                print(str(i) + '/' + str(self.num_items))
            uList1, rList1 = self.data.itemRated(item1)

            if len(uList1) < self.filter:
                continue
            for item2 in self.data.item:
                if item1 == item2:
                    continue
                if item2 not in occurrence[item1]:
                    uList2, rList2 = self.data.itemRated(item2)
                    if len(uList2) < self.filter:
                        continue
                    count = len(set(uList1).intersection(set(uList2)))
                    if count > self.filter:
                        occurrence[item1][item2] = count
                        occurrence[item2][item1] = count

        maxVal = 0
        frequency = {}
        for item1 in occurrence:
            frequency[item1] = sum(occurrence[item1].values()) * 1.0
        D = sum(frequency.values()) * 1.0
        # maxx = -1
        for item1 in occurrence:
            for item2 in occurrence[item1]:
                try:
                    val = max([log(occurrence[item1][item2] * D / (frequency[item1] * frequency[item2])) - log(
                        self.negCount), 0])
                except ValueError:
                    print(self.SPPMI[item1][item2])
                    print(self.SPPMI[item1][item2] * D / (frequency[item1] * frequency[item2]))
                if val > 0:
                    if maxVal < val:
                        maxVal = val
                    self.SPPMI[item1][item2] = val
                    self.SPPMI[item2][item1] = self.SPPMI[item1][item2]

        #normalize
        for item1 in self.SPPMI:
            for item2 in self.SPPMI[item1]:
                self.SPPMI[item1][item2] = self.SPPMI[item1][item2]/maxVal


    def trainModel(self):
        self.X=self.P*10 #Theta
        self.Y=self.Q*10 #Beta
        self.w = np.random.rand(self.num_items) / 10  # bias value of item
        self.c = np.random.rand(self.num_items) / 10  # bias value of context
        self.G = np.random.rand(self.num_items, self.emb_size) / 10  # context embedding

        print('training...')
        epoch = 0
        while epoch < self.maxEpoch:
            self.loss = 0
            YtY = self.Y.T.dot(self.Y)
            for user in self.data.user:
                # C_u = np.ones(self.data.getSize(self.recType))
                H = np.ones(self.num_items)
                val, pos = [],[]
                P_u = np.zeros(self.num_items)
                uid = self.data.user[user]
                for item in self.data.trainSet_u[user]:
                    iid = self.data.item[item]
                    r_ui = float(self.data.trainSet_u[user][item])
                    pos.append(iid)
                    val.append(10 * r_ui)
                    H[iid] += 10 * r_ui
                    P_u[iid] = 1
                    error = (P_u[iid] - self.X[uid].dot(self.Y[iid]))
                    self.loss += pow(error, 2)
                # sparse matrix
                C_u = coo_matrix((val, (pos, pos)), shape=(self.num_items, self.num_items))
                A = (YtY + np.dot(self.Y.T, C_u.dot(self.Y)) + self.regU * np.eye(self.emb_size))
                self.X[uid] = np.dot(np.linalg.inv(A), (self.Y.T * H).dot(P_u))

            XtX = self.X.T.dot(self.X)
            for item in self.data.item:
                P_i = np.zeros(self.num_users)
                iid = self.data.item[item]
                H = np.ones(self.num_users)
                val,pos = [],[]
                for user in self.data.trainSet_i[item]:
                    uid = self.data.user[user]
                    r_ui = float(self.data.trainSet_i[item][user])
                    pos.append(uid)
                    val.append(10 * r_ui)
                    H[uid] += 10 * r_ui
                    P_i[uid] = 1

                matrix_g1 = np.zeros((self.emb_size, self.emb_size))
                matrix_g2 = np.zeros((self.emb_size, self.emb_size))
                vector_m1 = np.zeros(self.emb_size)
                vector_m2 = np.zeros(self.emb_size)
                update_w = 0
                update_c = 0

                if len(self.SPPMI[item])>0:
                    for context in self.SPPMI[item]:
                        cid = self.data.item[context]
                        gamma = self.G[cid]
                        beta = self.Y[cid]
                        matrix_g1 += gamma.reshape(self.emb_size, 1).dot(gamma.reshape(1, self.emb_size))
                        vector_m1 += (self.SPPMI[item][context]-self.w[iid]-
                                      self.c[cid])*gamma

                        matrix_g2 += beta.reshape(self.emb_size, 1).dot(beta.reshape(1, self.emb_size))
                        vector_m2 += (self.SPPMI[item][context] - self.w[cid]
                                      - self.c[iid]) * beta

                        update_w += self.SPPMI[item][context]-self.Y[iid].dot(gamma)-self.c[cid]
                        update_c += self.SPPMI[item][context]-beta.dot(self.G[iid])-self.w[cid]

                C_i = coo_matrix((val, (pos, pos)), shape=(self.num_users, self.num_users))
                A = (XtX + np.dot(self.X.T, C_i.dot(self.X)) + self.regU * np.eye(self.emb_size) + matrix_g1)
                self.Y[iid] = np.dot(np.linalg.inv(A), (self.X.T * H).dot(P_i)+vector_m1)
                if len(self.SPPMI[item]) > 0:
                    self.G[iid] = np.dot(np.linalg.inv(matrix_g2 + self.regR * np.eye(self.emb_size)), vector_m2)
                    self.w[iid] = update_w/len(self.SPPMI[item])
                    self.c[iid] = update_c/len(self.SPPMI[item])

            epoch += 1
            print('epoch:', epoch, 'loss:', self.loss)

    def predictForRanking(self,u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.Y.dot(self.X[u])
        else:
            return [self.data.globalMean] * self.num_items
