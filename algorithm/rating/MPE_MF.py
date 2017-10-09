from baseclass.SocialRecommender import SocialRecommender
from tool import config
from random import randint
from random import shuffle, choice
from collections import defaultdict
import numpy as np
from tool.qmath import sigmoid, cosine
from math import log
from structure.symmetricMatrix import SymmetricMatrix
from time import localtime,time,strftime
import matplotlib.pyplot as plt



class MPE_MF(SocialRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        super(MPE_MF, self).__init__(conf, trainingSet, testSet, relation, fold)
        self.nonLeafVec = {}
        self.leafVec = {}


    def readConfiguration(self):
        super(MPE_MF, self).readConfiguration()
        options = config.LineConfig(self.config['CUNE-MF'])
        self.walkCount = int(options['-T'])
        self.walkLength = int(options['-L'])
        self.walkDim = int(options['-l'])
        self.winSize = int(options['-w'])
        self.topK = int(options['-k'])
        self.alpha = float(options['-a'])
        self.epoch = int(options['-ep'])
        self.neg = int(options['-neg'])
        self.rate = float(options['-r'])

    def printAlgorConfig(self):
        super(MPE_MF, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['recommender'] + ':'
        print 'Walks count per user', self.walkCount
        print 'Length of each walk', self.walkLength
        print 'Dimension of user embedding', self.walkDim
        print '=' * 80

    def buildModel(self):
        #data clean
        cleanList = []
        cleanPair = []
        for user in self.sao.followees:
            if not self.dao.user.has_key(user):
                cleanList.append(user)
            for u2 in self.sao.followees[user]:
                if not self.dao.user.has_key(u2):
                    cleanPair.append((user, u2))
        for u in cleanList:
            del self.sao.followees[u]

        for pair in cleanPair:
            if self.sao.followees.has_key(pair[0]):
                del self.sao.followees[pair[0]][pair[1]]
        li = self.sao.followees.keys()
        for u in li:
            if len(self.sao.followees[u]) <= 1:
                del self.sao.followees[u]
        print 'Kind Note: This method will probably take much time.'
        # build U-F-NET
        print 'Building weighted user-friend network...'
        # filter isolated nodes and low ratings
        # Definition of Meta-Path
        p1 = 'UIU'
        p2 = 'UFU'
        p3 = 'UFIU'
        p4 = ''
        mPaths = [p1, p2, p3]

        self.G = np.random.rand(self.dao.trainingSize()[1], self.walkDim) / 10
        self.W = np.random.rand(self.dao.trainingSize()[0], self.walkDim) / 10

        self.fItems = {}  # filtered item set
        for item in self.dao.trainSet_i:
            if len(self.dao.trainSet_i[item]) > 1:
                self.fItems[item] = self.dao.trainSet_i[item].keys()

        self.fBuying = {}  # filtered buying set
        for user in self.dao.trainSet_u:
            self.fBuying[user] = []
            for item in self.dao.trainSet_u[user]:
                if self.fItems.has_key(item) and self.dao.trainSet_u[user][item] > 0.75:
                    self.fBuying[user].append(item)
            if self.fBuying[user] == []:
                del self.fBuying[user]
        # self.filteredRatings = defaultdict(list)
        # for item in self.fItems:
        #     for user in self.fItems[item]:
        #         if self.fItems[item][user]>0.75:
        #             self.filteredRatings[user].append(item)

        self.UFNet = defaultdict(list)

        for user1 in self.sao.followees:
            s1 = set(self.sao.followees[user1])
            for user2 in self.sao.followees[user1]:
                if self.sao.followees.has_key(user2):
                    if user1 <> user2:
                        s2 = set(self.sao.followees[user2])
                        weight = len(s1.intersection(s2))
                        self.UFNet[user1] += [user2] * (weight + 1)
        #
        #
        #
        #
        # print 'Generating random meta-path random walks...'
        # self.walks = []
        # # self.visited = defaultdict(dict)
        # for user in self.fBuying:
        #
        #
        #     for mp in mPaths:
        #         if mp==p1:
        #             self.walkCount = 5
        #         if mp==p2:
        #             self.walkCount = 8
        #         if mp==p3:
        #             self.walkCount = 20
        #         for t in range(self.walkCount):
        #
        #             path = [(user, 'U')]
        #             lastNode = user
        #             nextNode = user
        #             lastType = 'U'
        #             for i in range(self.walkLength / len(mp)):
        #
        #                     for tp in mp[1:]:
        #                         try:
        #                             if tp == 'I':
        #                                 # if not self.fBuying.has_key(lastNode):
        #                                 #     path = []
        #                                 #     break
        #                                 nextNode = choice(self.fBuying[lastNode])
        #
        #                             if tp == 'U':
        #                                 # if lastType=='':
        #                                 #     nextNode = user
        #                                 if lastType == 'I':
        #                                     nextNode = choice(self.fItems[lastNode])
        #                                 elif lastType == 'F':
        #                                     nextNode = choice(self.UFNet[lastNode])
        #                                     while not self.dao.user.has_key(nextNode):
        #                                         nextNode = choice(self.UFNet[lastNode])
        #
        #                             if tp == 'F':
        #                                 # if not self.UFNet.has_key(lastNode):
        #                                 #     path = []
        #                                 #     break
        #                                 nextNode = choice(self.UFNet[lastNode])
        #                                 while not self.dao.user.has_key(nextNode):
        #                                     nextNode = choice(self.UFNet[lastNode])
        #
        #                             path.append((nextNode, tp))
        #                             lastNode = nextNode
        #                             lastType = tp
        #
        #                         except (KeyError, IndexError):
        #                             path = []
        #                             break
        #
        #
        #
        #             if path:
        #                 self.walks.append(path)
        #                 # print path
        #                 # if mp == 'UFIU':
        #                 # pass
        # shuffle(self.walks)
        # print 'walks:',len(self.walks)
        # # Training get top-k friends
        # print 'Generating user embedding...'
        # iteration = 1
        # userList = self.dao.user.keys()
        # itemList = self.dao.item.keys()
        # while iteration <= self.epoch:
        #     loss = 0
        #
        #     for walk in self.walks:
        #         for i, node in enumerate(walk):
        #             neighbors = walk[max(0, i - self.winSize / 2):min(len(walk) - 1, i + self.winSize / 2)]
        #             center, ctp = walk[i]
        #             if ctp == 'U' or ctp == 'F':  # user
        #                 centerVec = self.W[self.dao.user[center]]
        #             else:  # Item
        #                 centerVec = self.G[self.dao.item[center]]
        #             for entity, tp in neighbors:
        #                 # negSamples = []
        #                 currentVec = ''
        #                 if tp == 'U' or tp == 'F' and center <> entity:
        #                     currentVec = self.W[self.dao.user[entity]]
        #                     self.W[self.dao.user[entity]] +=   self.rate * (
        #                         1 - sigmoid(currentVec.dot(centerVec))) * centerVec
        #                     if ctp == 'U' or ctp == 'F':
        #                         self.W[self.dao.user[center]] +=   self.rate * (
        #                             1 - sigmoid(currentVec.dot(centerVec))) * currentVec
        #                     else:
        #                         self.G[self.dao.item[center]] +=   self.rate * (
        #                             1 - sigmoid(currentVec.dot(centerVec))) * currentVec
        #                     loss += -  log(sigmoid(currentVec.dot(centerVec)))
        #                     for i in range(self.neg):
        #                         sample = choice(userList)
        #                         while sample == entity:
        #                             sample = choice(userList)
        #                         sampleVec = self.W[self.dao.user[sample]]
        #                         self.W[self.dao.user[sample]] -=   self.rate * (
        #                             1 - sigmoid(-sampleVec.dot(centerVec))) * centerVec
        #                         if ctp == 'U' or ctp == 'F':
        #                             self.W[self.dao.user[center]] -=   self.rate * (
        #                                 1 - sigmoid(-sampleVec.dot(centerVec))) * sampleVec
        #                         else:
        #                             self.G[self.dao.item[center]] -=   self.rate * (
        #                                 1 - sigmoid(-sampleVec.dot(centerVec))) * sampleVec
        #                         #loss += -  log(sigmoid(-sampleVec.dot(centerVec)))
        #                         # negSamples.append(choice)
        #                 elif tp == 'I' and center <> entity:
        #                     currentVec = self.G[self.dao.item[entity]]
        #                     self.G[self.dao.item[entity]] +=   self.rate * (
        #                         1 - sigmoid(currentVec.dot(centerVec))) * centerVec
        #                     if ctp == 'U' or ctp == 'F':
        #                         self.W[self.dao.user[center]] +=   self.rate * (
        #                             1 - sigmoid(currentVec.dot(centerVec))) * currentVec
        #                     else:
        #                         self.G[self.dao.item[center]] +=   self.rate * (
        #                             1 - sigmoid(currentVec.dot(centerVec))) * currentVec
        #                     loss += -  log(sigmoid(currentVec.dot(centerVec)))
        #                     for i in range(self.neg):
        #                         sample = choice(itemList)
        #                         while sample == entity:
        #                             sample = choice(itemList)
        #                         # negSamples.append(choice)
        #                         sampleVec = self.G[self.dao.item[sample]]
        #                         self.G[self.dao.item[sample]] -= self.rate * (
        #                             1 - sigmoid(-currentVec.dot(centerVec))) * centerVec
        #                         if ctp == 'U' or ctp == 'F':
        #                             self.W[self.dao.user[center]] -=   self.rate * (
        #                                 1 - sigmoid(-sampleVec.dot(centerVec))) * sampleVec
        #                         else:
        #                             self.G[self.dao.item[center]] -=   self.rate * (
        #                                 1 - sigmoid(-sampleVec.dot(centerVec))) * sampleVec
        #                         #loss += -self.alpha * log(sigmoid(-sampleVec.dot(centerVec)))
        #     shuffle(self.walks)
        #
        #     print 'iteration:', iteration, 'loss:', loss
        #     iteration += 1
        # print 'User embedding generated.'
        # from mpl_toolkits.mplot3d import Axes3D
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=3)
        # dW = pca.fit_transform(self.W)
        # dG = pca.fit_transform(self.G)
        # ax = plt.figure().add_subplot(111, projection='3d')
        # #print len(self.W[:,0])
        # x = []
        # y = []
        # z = []
        # from random import random
        # for user in self.fBuying:
        #
        #     x.append(dW[self.dao.user[user]][0])
        #     y.append(dW[self.dao.user[user]][1])
        #     z.append(dW[self.dao.user[user]][2])
        # print len(x)
        # ax.scatter(x,y,z,marker='*',color='red')
        # x = []
        # y = []
        # z = []
        # for item in self.fItems:
        #     x.append(dG[self.dao.item[item]][0])
        #     y.append(dG[self.dao.item[item]][1])
        #     z.append(dG[self.dao.item[item]][2])
        # print len(x)
        # ax.scatter(x, y,z, marker='o', color='green')
        # plt.show()
        #
        # print 'Constructing similarity matrix...'
        # i = 0
        # self.topKSim = {}
        # self.Sim = SymmetricMatrix(len(self.UFNet))
        # for user1 in self.fBuying:
        #     uSim = []
        #     i+=1
        #     if i%200==0:
        #         print i,'/',len(self.fBuying)
        #     vec1 = self.W[self.dao.user[user1]]
        #     for user2 in self.fBuying:
        #         if user1 <> user2:
        #             vec2 = self.W[self.dao.user[user2]]
        #             sim = cosine(vec1, vec2)
        #             uSim.append((user2,sim))
        #
        #     self.topKSim[user1] = sorted(uSim, key=lambda d: d[1], reverse=True)[:self.topK]
        #
        #
        # print 'Similarity matrix finished.'
        # #print self.topKSim
        import pickle

        # recordTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # similarity = open('sim'+recordTime+'.pkl', 'wb')
        # vectors = open('vec'+recordTime+'.pkl', 'wb')
        # Pickle dictionary using protocol 0.

        # pickle.dump(self.topKSim, similarity)
        # pickle.dump((self.W,self.G),vectors)
        # similarity.close()
        # vectors.close()

        # matrix decomposition
        pkl_file = open('MPE-Ciao-sim'+self.foldInfo+'.pkl', 'rb')

        self.topKSim = pickle.load(pkl_file)
        print 'Decomposing...'
        self.F = np.random.rand(self.dao.trainingSize()[0], self.k) / 10

        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0

            for user in self.fBuying:
                u = self.dao.user[user]
                friends = self.topKSim[user][:self.topK]
                for friend in friends:
                    uf = self.dao.user[friend[0]]
                    #self.P[u] -= self.lRate * (self.P[u] - self.P[uf]) * self.alpha
                    #self.loss += self.alpha * (self.P[u] - self.P[uf]).dot(self.P[u] - self.P[uf])
                    error = friend[1] - self.P[u].dot(self.F[uf])
                    self.loss += self.alpha*error ** 2
                    p = self.P[u]
                    z = self.F[uf]

                    # update latent vectors
                    self.P[u] += self.alpha* self.lRate * (error * z )  # - self.alpha * (self.P[u]-self.W[u]))
                    self.F[uf] += self.alpha* self.lRate * (error * p -self.regU*z)
                    #self.W[uf] += self.alpha* self.lRate * (error * p - self.regI * q)


            for entry in self.dao.trainingData:
                user, item, rating = entry
                u = self.dao.user[user]  # get user id
                i = self.dao.item[item]  # get item id
                error = rating - self.P[u].dot(self.Q[i])
                self.loss += error ** 2
                p = self.P[u]
                q = self.Q[i]

                # update latent vectors
                self.P[u] += self.lRate * (error * q - self.regU * p)  # - self.alpha * (self.P[u]-self.W[u]))
                self.Q[i] += self.lRate * (error * p - self.regI * q)


            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break
