from baseclass.SocialRecommender import SocialRecommender
from tool import config
from random import randint
from random import shuffle, choice
from collections import defaultdict
import numpy as np
from tool.qmath import sigmoid, cosine, cosine_sp
from math import log
import gensim.models.word2vec as w2v


class HER(SocialRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        super(HER, self).__init__(conf, trainingSet, testSet, relation, fold)

    def readConfiguration(self):
        super(HER, self).readConfiguration()
        options = config.LineConfig(self.config['HER'])
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
        super(HER, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['recommender'] + ':'
        print 'Walks count per user', self.walkCount
        print 'Length of each walk', self.walkLength
        print 'Dimension of user embedding', self.walkDim
        print '=' * 80

    def buildModel(self):
        # tobeCleaned = []
        # for user in self.dao.testSet_u:
        #     for item in self.dao.testSet_u[user]:
        #         if self.dao.testSet_u[user][item]<4.5:
        #             tobeCleaned.append((user,item))
        #
        # for pair in tobeCleaned:
        #     del self.dao.testSet_u[pair[0]][pair[1]]
        #     if len(self.dao.testSet_u[pair[0]])==0:
        #         del self.dao.testSet_u[pair[0]]
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

        cleanList = []
        cleanPair = []
        for user in self.sao.followers:
            if not self.dao.user.has_key(user):
                cleanList.append(user)
            for u2 in self.sao.followers[user]:
                if not self.dao.user.has_key(u2):
                    cleanPair.append((user, u2))
        for u in cleanList:
            del self.sao.followers[u]

        for pair in cleanPair:
            if self.sao.followers.has_key(pair[0]):
                del self.sao.followers[pair[0]][pair[1]]

        # li = self.sao.followees.keys()
        #
        print 'Kind Note: This method will probably take much time.'
        # build U-F-NET
        print 'Building weighted user-friend network...'
        # filter isolated nodes and low ratings
        # Definition of Meta-Path
        p1 = 'UIU'
        p2 = 'UFU'
        p3 = 'UTU'
        p4 = 'UFIU'
        p5 = 'UFUIU'
        mPaths = [p1,p2,p3,p4,p5]

        self.G = np.random.rand(self.dao.trainingSize()[1], self.walkDim) / 10
        self.W = np.random.rand(self.dao.trainingSize()[0], self.walkDim) / 10

        self.fItems = {}  # filtered item set
        for item in self.dao.trainSet_i:
            self.fItems[item] = self.dao.trainSet_i[item].keys()

        self.fBuying = {}  # filtered buying set
        for user in self.dao.trainSet_u:
            self.fBuying[user] = []
            for item in self.dao.trainSet_u[user]:
                if self.fItems.has_key(item) and self.dao.trainSet_u[user][item] > 0:
                    self.fBuying[user].append(item)
            if self.fBuying[user] == []:
                del self.fBuying[user]


        self.UFNet = defaultdict(list)

        for user1 in self.sao.followees:
            s1 = set(self.sao.followees[user1])
            for user2 in self.sao.followees[user1]:
                if self.sao.followees.has_key(user2):
                    if user1 <> user2:
                        s2 = set(self.sao.followees[user2])
                        weight = len(s1.intersection(s2))
                        self.UFNet[user1] += [user2] * (weight + 1)

        self.UTNet = defaultdict(list)

        for user1 in self.sao.followers:
            s1 = set(self.sao.followers[user1])
            for user2 in self.sao.followers[user1]:
                if self.sao.followers.has_key(user2):
                    if user1 <> user2:
                        s2 = set(self.sao.followers[user2])
                        weight = len(s1.intersection(s2))
                        self.UTNet[user1] += [user2] * (weight + 1)
        #
        #
        #
        #
        print 'Generating random meta-path random walks...'
        self.walks = []
        #self.usercovered = {}


        for user in self.dao.user:

            for mp in mPaths:
                if mp == p1:
                    self.walkCount = 10
                if mp == p2:
                    self.walkCount = 8
                if mp == p3:
                    self.walkCount = 8
                if mp == p4:
                    self.walkCount = 5
                if mp == p5:
                    self.walkCount = 5
                for t in range(self.walkCount):

                    path = ['U'+user]
                    lastNode = user
                    nextNode = user
                    lastType = 'U'
                    for i in range(self.walkLength / len(mp[1:])):

                        for tp in mp[1:]:
                            try:
                                if tp == 'I':

                                    nextNode = choice(self.fBuying[lastNode])

                                if tp == 'U':

                                    if lastType == 'I':
                                        nextNode = choice(self.fItems[lastNode])
                                    elif lastType == 'F':
                                        nextNode = choice(self.UFNet[lastNode])
                                        while not self.dao.user.has_key(nextNode):
                                            nextNode = choice(self.UFNet[lastNode])
                                    elif lastType == 'T':
                                        nextNode = choice(self.UTNet[lastNode])
                                        while not self.dao.user.has_key(nextNode):
                                            nextNode = choice(self.UTNet[lastNode])



                                if tp == 'F':

                                    nextNode = choice(self.UFNet[lastNode])
                                    while not self.dao.user.has_key(nextNode):
                                        nextNode = choice(self.UFNet[lastNode])

                                if tp == 'T':

                                    nextNode = choice(self.UFNet[lastNode])
                                    while not self.dao.user.has_key(nextNode):
                                        nextNode = choice(self.UFNet[lastNode])

                                path.append(tp+nextNode)
                                lastNode = nextNode
                                lastType = tp

                            except (KeyError, IndexError):
                                path = []
                                break

                    if path:
                        self.walks.append(path)
                        # for node in path:
                        #     if node[1] == 'U' or node[1] == 'F':
                        #         self.usercovered[node[0]] = 1
                                # print path
                                # if mp == 'UFIU':
                                # pass
        shuffle(self.walks)
        uList = []
        coldCount = 0
        while len(uList)<1000:
            cp = choice(self.walks)
            su = choice(cp)
            if su[0]=='I':
                su = choice(cp)
            if len(self.dao.trainSet_u[su[1:]])<10:
                coldCount+=1
            uList.append(su)
        print 'cold rate:',float(coldCount)/1000


        print 'walks:', len(self.walks)
        # Training get top-k friends
        print 'Generating user embedding...'
        # print 'user covered', len(self.usercovered)
        # print 'user coverage', float(len(self.usercovered)) / len(self.dao.user)
        # sampleWalks = []
        # sampleUser = {}
        # for i in range(1000):
        #     p = choice(self.walks)
        #     u = choice(p)
        #     while u[1]=='I':
        #         u = choice(p)
        #     if len(self.dao.trainSet_u[u[0]])<=10:
        #         sampleUser[u[0]] = 1
        #
        # print 'user coverage:', len(sampleUser)/float(1000)




        # iteration = 1
        # userList = self.dao.user.keys()
        # itemList = self.dao.item.keys()
        self.topKSim = {}
        # while iteration <= self.epoch:
        #     loss = 0
        #
        #     for walk in self.walks:
        #         for i, node in enumerate(walk):
        #             neighbors = walk[max(0, i - self.winSize / 2):min(len(walk) - 1, i + self.winSize / 2)]
        #             center, ctp = walk[i]
        #             if ctp == 'U' or ctp == 'F' or ctp == 'T':  # user
        #                 centerVec = self.W[self.dao.user[center]]
        #             else:  # Item
        #                 centerVec = self.G[self.dao.item[center]]
        #             for entity, tp in neighbors:
        #                 # negSamples = []
        #                 currentVec = ''
        #                 if tp == 'U' or tp == 'F' or tp=='T' and center <> entity:
        #                     currentVec = self.W[self.dao.user[entity]]
        #                     self.W[self.dao.user[entity]] +=   self.rate * (
        #                         1 - sigmoid(currentVec.dot(centerVec))) * centerVec
        #                     if ctp == 'U' or ctp == 'F' or ctp=='T':
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
        #                         if ctp == 'U' or ctp == 'F' or ctp == 'T':
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
        #                     if ctp == 'U' or ctp == 'F' or ctp == 'T':
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
        #                         if ctp == 'U' or ctp == 'F' or ctp == 'T':
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

        model = w2v.Word2Vec(self.walks, size=self.walkDim, window=5, min_count=0, iter=10)

        for user in self.fBuying:
            uid = self.dao.user[user]
            self.W[uid] = model.wv['U'+user]
        print 'User embedding generated.'

        print 'Constructing similarity matrix...'
        i = 0


        for user1 in self.fBuying:
            uSim = []
            i+=1
            if i%200==0:
                print i,'/',len(self.fBuying)
            vec1 = self.W[self.dao.user[user1]]
            for user2 in self.fBuying:
                if user1 <> user2:
                    vec2 = self.W[self.dao.user[user2]]
                    sim = cosine(vec1, vec2)
                    uSim.append((user2,sim))

            self.topKSim[user1] = sorted(uSim, key=lambda d: d[1], reverse=True)[:self.topK]


        from tool import qmath
        # for user1 in self.dao.user:
        #     uSim = []
        #     i += 1
        #     if i % 200 == 0:
        #         print i, '/', len(self.dao.user)
        #     l1 = self.dao.trainSet_u[user1].keys()
        #     for user2 in self.dao.user:
        #         if user1 <> user2:
        #             l2 = self.dao.trainSet_u[user2].keys()
        #             sim = len(set(l1).intersection(l2))
        #             if sim==0:
        #                 continue
        #             uSim.append((user2, sim))
        #
        #     self.topKSim[user1] = sorted(uSim, key=lambda d: d[1], reverse=True)[:self.topK]


        # print 'Similarity matrix finished.'
        # # # #print self.topKSim
        #import pickle
        # # # #
        # # # #recordTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # similarity = open('HER-lastfm-sim'+self.foldInfo+'.pkl', 'wb')
        # vectors = open('HER-lastfm-vec'+self.foldInfo+'.pkl', 'wb')
        # #Pickle dictionary using protocol 0.
        #
        # pickle.dump(self.topKSim, similarity)
        # pickle.dump((self.W,self.G),vectors)
        # similarity.close()
        # vectors.close()

        # matrix decomposition
        #pkl_file = open('HER-lastfm-sim' + self.foldInfo + '.pkl', 'rb')

        #self.topKSim = pickle.load(pkl_file)

        print 'Decomposing...'
        self.F = np.random.rand(self.dao.trainingSize()[0], self.k) / 10
        # prepare Pu set, IPu set, and Nu set
        print 'Preparing item sets...'
        self.PositiveSet = defaultdict(dict)
        self.IPositiveSet = defaultdict(dict)


        for user in self.topKSim:
            for item in self.dao.trainSet_u[user]:
                self.PositiveSet[user][item]=1
                # else:
                #     self.NegativeSet[user].append(item)

            for friend in self.topKSim[user]:
                for item in self.dao.trainSet_u[friend[0]]:
                    if not self.PositiveSet[user].has_key(item):
                        self.IPositiveSet[user][item]=1

        #print self.IPositiveSet
        iteration = 0
        self.s=1
        while iteration < self.maxIter:
            self.loss = 0
            itemList = self.dao.item.keys()

            for user in self.PositiveSet:
                kItems = self.IPositiveSet[user].keys()
                u = self.dao.user[user]

                for item in self.PositiveSet[user]:
                    i = self.dao.item[item]
                    for ind in range(3):
                        if len(kItems) > 0:
                            item_k = choice(kItems)

                            k = self.dao.item[item_k]
                            self.optimization(u,i,k)

                            item_j = choice(itemList)
                            while (self.PositiveSet[user].has_key(item_j) or self.IPositiveSet[user].has_key(item)):
                                item_j = choice(itemList)
                            j = self.dao.item[item_j]
                            self.optimization(u,k,j)
                        else:
                            item_j = choice(itemList)
                            while (self.PositiveSet[user].has_key(item_j)):
                                item_j = choice(itemList)
                            j = self.dao.item[item_j]
                            self.optimization(u, i, j)


            # for user in self.topKSim:
            #     for friend in self.topKSim[user]:
            #         u = self.dao.user[user]
            #         f = self.dao.user[friend[0]]
            #         self.P[u] -= self.alpha*self.lRate*(self.P[u]-self.P[f])

            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break

    def optimization(self, u, i, j):
        s = sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))
        self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
        self.Q[i] += self.lRate * (1 - s) * self.P[u]
        self.Q[j] -= self.lRate * (1 - s) * self.P[u]
        self.loss += -log(s)
        self.P[u] -= self.lRate * self.regU * self.P[u]
        self.Q[i] -= self.lRate * self.regI * self.Q[i]
        self.Q[j] -= self.lRate * self.regI * self.Q[j]

    def predict(self,user,item):

        if self.dao.containsUser(user) and self.dao.containsItem(item):
            u = self.dao.getUserId(user)
            i = self.dao.getItemId(item)
            predictRating = sigmoid(self.Q[i].dot(self.P[u]))
            return predictRating
        else:
            return sigmoid(self.dao.globalMean)

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            u = self.dao.getUserId(u)
            return self.Q.dot(self.P[u])
        else:
            return [self.dao.globalMean] * len(self.dao.item)