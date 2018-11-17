from baseclass.SocialRecommender import SocialRecommender
from tool import config
from random import randint
from random import shuffle, choice
from collections import defaultdict
import numpy as np
from tool.qmath import sigmoid, cosine, cosine_sp
from math import log
import gensim.models.word2vec as w2v
import json

class IF_BPR(SocialRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        super(IF_BPR, self).__init__(conf, trainingSet, testSet, relation, fold)

    def readConfiguration(self):
        super(IF_BPR, self).readConfiguration()
        options = config.LineConfig(self.config['IF_BPR'])
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
        super(IF_BPR, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['recommender'] + ':'
        print 'Walks count per user', self.walkCount
        print 'Length of each walk', self.walkLength
        print 'Dimension of user embedding', self.walkDim
        print '=' * 80

    def readNegativeFeedbacks(self):
        self.negative = defaultdict(list)
        self.nItems = defaultdict(list)
        filename = self.config['ratings'][:-4]+'_n.txt'
        with open(filename) as f:
            for line in f:
                items = line.strip().split()
                self.negative[items[0]].append(items[1])
                self.nItems[items[1]].append(items[0])
                if items[0] not in self.dao.user:
                    self.dao.user[items[0]]=len(self.dao.user)



    def initModel(self):
        super(IF_BPR, self).initModel()
        self.positive = defaultdict(list)
        self.pItems = defaultdict(list)
        for user in self.dao.trainSet_u:
            for item in self.dao.trainSet_u[user]:
                self.positive[user].append(item)
                self.pItems[item].append(user)
        self.readNegativeFeedbacks()
        self.P = np.ones((len(self.dao.user), self.k))*0.1  # latent user matrix
        #self.Q = np.ones((len(self.dao.item), self.k)) / 10  # latent item matrix
        self.threshold = {}
        self.avg_sim = {}
        self.thres_d = dict.fromkeys(self.dao.user.keys(),0)
        self.thres_count = dict.fromkeys(self.dao.user.keys(),0)

        print 'Preparing item sets...'
        self.PositiveSet = defaultdict(dict)
        self.NegSets = defaultdict(dict)

        for user in self.dao.user:
            for item in self.dao.trainSet_u[user]:
                self.PositiveSet[user][item] = 1

        for user in self.dao.user:
            for item in self.negative[user]:
                if self.dao.item.has_key(item):
                    self.NegSets[user][item] = 1

    def randomWalks(self):
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
        mPaths = [p1, p2, p3, p4, p5]

        self.G = np.random.rand(self.dao.trainingSize()[0], self.walkDim) * 0.1
        self.W = np.random.rand(self.dao.trainingSize()[0], self.walkDim) * 0.1

        self.UFNet = defaultdict(list)

        for u in self.sao.followees:
            s1 = set(self.sao.followees[u])
            for v in self.sao.followees[u]:
                if v in self.sao.followees:  # make sure that v has out links
                    if u <> v:
                        s2 = set(self.sao.followees[v])
                        weight = len(s1.intersection(s2))
                        self.UFNet[u] += [v] * (weight + 1)

        self.UTNet = defaultdict(list)

        for u in self.sao.followers:
            s1 = set(self.sao.followers[u])
            for v in self.sao.followers[u]:
                if self.sao.followers.has_key(v):  # make sure that v has out links
                    if u <> v:
                        s2 = set(self.sao.followers[v])
                        weight = len(s1.intersection(s2))
                        self.UTNet[u] += [v] * (weight + 1)

        print 'Generating random meta-path random walks... (Positive)'
        self.pWalks = []
        # self.usercovered = {}

        # positive
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

                    path = ['U' + user]
                    lastNode = user
                    nextNode = user
                    lastType = 'U'
                    for i in range(self.walkLength / len(mp[1:])):

                        for tp in mp[1:]:
                            try:
                                if tp == 'I':
                                    nextNode = choice(self.positive[lastNode])

                                if tp == 'U':
                                    if lastType == 'I':
                                        nextNode = choice(self.pItems[lastNode])
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

                                path.append(tp + nextNode)
                                lastNode = nextNode
                                lastType = tp

                            except (KeyError, IndexError):
                                path = []
                                break

                    if path:
                        self.pWalks.append(path)

        self.nWalks = []
        # self.usercovered = {}

        # negative
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

                    path = ['U' + user]
                    lastNode = user
                    nextNode = user
                    lastType = 'U'
                    for i in range(self.walkLength / len(mp[1:])):

                        for tp in mp[1:]:
                            try:
                                if tp == 'I':
                                    nextNode = choice(self.negative[lastNode])

                                if tp == 'U':
                                    if lastType == 'I':
                                        nextNode = choice(self.nItems[lastNode])
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

                                path.append(tp + nextNode)
                                lastNode = nextNode
                                lastType = tp

                            except (KeyError, IndexError):
                                path = []
                                break

                    if path:
                        self.nWalks.append(path)

        shuffle(self.pWalks)
        print 'pwalks:', len(self.pWalks)
        print 'nwalks:', len(self.nWalks)

    def computeSimilarity(self):
        # Training get top-k friends
        print 'Generating user embedding...'
        self.pTopKSim = {}
        self.nTopKSim = {}
        self.pSimilarity = defaultdict(dict)
        self.nSimilarity = defaultdict(dict)
        model = w2v.Word2Vec(self.pWalks, size=self.walkDim, window=5, min_count=0, iter=10)
        model2 = w2v.Word2Vec(self.nWalks, size=self.walkDim, window=5, min_count=0, iter=10)

        for user in self.positive:
            uid = self.dao.user[user]
            try:
                self.W[uid] = model.wv['U' + user]
            except KeyError:
                continue

        for user in self.negative:
            uid = self.dao.user[user]
            try:
                self.G[uid] = model2.wv['U' + user]
            except KeyError:
                continue
        print 'User embedding generated.'

        print 'Constructing similarity matrix...'
        i = 0
        for user1 in self.positive:
            uSim = []
            i += 1
            if i % 200 == 0:
                print i, '/', len(self.positive)
            vec1 = self.W[self.dao.user[user1]]
            for user2 in self.positive:
                if user1 <> user2:
                    vec2 = self.W[self.dao.user[user2]]
                    sim = cosine(vec1, vec2)
                    uSim.append((user2, sim))
            fList = sorted(uSim, key=lambda d: d[1], reverse=True)[:self.topK]
            self.threshold[user1] = fList[self.topK / 2][1]
            for pair in fList:
                self.pSimilarity[user1][pair[0]] = pair[1]
            self.pTopKSim[user1] = [item[0] for item in fList]
            self.avg_sim[user1] = sum([item[1] for item in fList][:self.topK / 2]) / (self.topK / 2)

        i = 0
        for user1 in self.negative:
            uSim = []
            i += 1
            if i % 200 == 0:
                print i, '/', len(self.negative)
            vec1 = self.G[self.dao.user[user1]]
            for user2 in self.negative:
                if user1 <> user2:
                    vec2 = self.G[self.dao.user[user2]]
                    sim = cosine(vec1, vec2)
                    uSim.append((user2, sim))
            fList = sorted(uSim, key=lambda d: d[1], reverse=True)[:self.topK]
            for pair in fList:
                self.nSimilarity[user1][pair[0]] = pair[1]
            self.nTopKSim[user1] = [item[0] for item in fList]

    def updateSets(self):
        self.JointSet = defaultdict(dict)
        self.PS_Set = defaultdict(dict)
        for user in self.dao.user:
            if user in self.trueTopKFriends:
                for friend in self.trueTopKFriends[user]:
                    if friend in self.dao.user and self.pSimilarity[user][friend] >= self.threshold[user]:
                        for item in self.positive[friend]:
                            if item not in self.PositiveSet[user] and item not in self.NegSets[user]:
                                self.JointSet[user][item] = friend

            if self.pTopKSim.has_key(user):
                for friend in self.pTopKSim[user][:self.topK]:
                    if friend in self.dao.user and self.pSimilarity[user][friend] >= self.threshold[user]:
                        for item in self.positive[friend]:
                            if item not in self.PositiveSet[user] and item not in self.JointSet[user] \
                                    and item not in self.NegSets[user]:
                                self.PS_Set[user][item] = friend

            if self.nTopKSim.has_key(user):
                for friend in self.nTopKSim[user][:self.topK]:
                    if friend in self.dao.user:  # and self.nSimilarity[user][friend]>=self.threshold[user]:
                        for item in self.negative[friend]:
                            if item in self.dao.item:
                                if item not in self.PositiveSet[user] and item not in self.JointSet[user] \
                                        and item not in self.PS_Set[user]:
                                    self.NegSets[user][item] = 1

    def buildModel(self):
        # self.P = np.ones((self.dao.trainingSize()[0], self.k))/10  # latent user matrix
        # self.Q = np.ones((self.dao.trainingSize()[1], self.k))/10  # latent item matrix


        # li = self.sao.followees.keys()
        #
        # import pickle
        #
        # self.trueTopKFriends = defaultdict(list)
        # pkl_file = open(self.config['ratings'] + self.foldInfo + 'p.pkl', 'rb')
        # self.pTopKSim = pickle.load(pkl_file)
        # pkl_file = open(self.config['ratings'] + self.foldInfo + 'n.pkl', 'rb')
        # self.nTopKSim = pickle.load(pkl_file)
        # self.trueTopKFriends = defaultdict(list)
        # for user in self.pTopKSim:
        #     trueFriends = list(
        #         set(self.pTopKSim[user][:self.topK]).intersection(set(self.nTopKSim[user][:self.topK])))
        #     self.trueTopKFriends[user] = trueFriends
        #
        # ps = open(self.config['ratings'] + self.foldInfo + 'psim.pkl', 'rb')
        # self.pSimilarity=pickle.load(ps)
        # ns = open(self.config['ratings'] + self.foldInfo + 'nsim.pkl', 'rb')
        # self.nSimilarity=pickle.load(ns)
        # av = open(self.config['ratings'] + self.foldInfo + 'av.pkl', 'rb')
        # self.avg_sim=pickle.load(av)
        # th = open(self.config['ratings'] + self.foldInfo + 'th.pkl', 'rb')
        # self.threshold=pickle.load(th)


        # import pickle
        # ps = open(self.config['ratings'] + self.foldInfo + 'ps.pkl', 'wb')
        #
        # pickle.dump(self.pSimilarity, ps)
        # av = open(self.config['ratings'] + self.foldInfo + 'av.pkl', 'wb')
        #
        # pickle.dump(self.avg_sim, av)
        #
        # th = open(self.config['ratings'] + self.foldInfo + 'th.pkl', 'wb')
        # pickle.dump(self.threshold, th)


        # print 'Similarity matrix finished.'
        # # # #print self.topKSim

        # # # #
        # # # #recordTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # psimilarity = open(self.config['ratings']+self.foldInfo+'p.pkl', 'wb')
        # nsimilarity = open(self.config['ratings'] + self.foldInfo + 'n.pkl', 'wb')
        # vectors = open('HERP-lastfm-vec'+self.foldInfo+'.pkl', 'wb')
        # #Pickle dictionary using protocol 0.
        #
        # pickle.dump(self.pTopKSim, psimilarity)
        # pickle.dump(self.nTopKSim, nsimilarity)
        #
        # psimilarity = open(self.config['ratings'] + self.foldInfo + 'psim.pkl', 'wb')
        # nsimilarity = open(self.config['ratings'] + self.foldInfo + 'nsim.pkl', 'wb')
        # vectors = open('HERP-lastfm-vec'+self.foldInfo+'.pkl', 'wb')
        # #Pickle dictionary using protocol 0.
        #


        # pickle.dump(self.pSimilarity, psimilarity)
        # pickle.dump(self.nSimilarity, nsimilarity)


        #pickle.dump((self.W,self.G),vectors)
        # similarity.close()
        # vectors.close()

        # matrix decomposition
        #pkl_file = open('IF_BPR-lastfm-sim' + self.foldInfo + '.pkl', 'rb')

        #self.topKSim = pickle.load(pkl_file)

        self.randomWalks()
        self.computeSimilarity()

        self.trueTopKFriends=defaultdict(list)
        for user in self.pTopKSim:
            trueFriends = list(set(self.pTopKSim[user]).intersection(set(self.nTopKSim[user])))
            self.trueTopKFriends[user] = trueFriends
            self.pTopKSim[user] = list(set(self.pTopKSim[user]).difference(set(trueFriends)))

        print 'Decomposing...'

        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            self.updateSets()
            itemList = self.dao.item.keys()
            for user in self.PositiveSet:
                #itemList = self.NegSets[user].keys()
                kItems = self.JointSet[user].keys()
                okItems = self.PS_Set[user].keys()
                nItems = self.NegSets[user].keys()

                u = self.dao.user[user]

                for item in self.PositiveSet[user]:
                    i = self.dao.item[item]

                    for ind in range(1):
                        if len(kItems) > 0 and len(okItems) > 0:

                            item_k = choice(kItems)
                            uf = self.JointSet[user][item_k]
                            k = self.dao.item[item_k]
                            self.optimization_thres(u,i,k,user,uf)

                            item_ok = choice(okItems)
                            ok = self.dao.item[item_ok]

                            self.optimization(u,k,ok)

                            item_j = choice(itemList)
                            while (self.PositiveSet[user].has_key(item_j) or self.JointSet[user].has_key(item_j) or self.PS_Set[user].has_key(item_j)):
                                item_j = choice(itemList)
                            j = self.dao.item[item_j]
                            self.optimization(u,ok,j)

                        elif len(kItems)==0 and len(okItems)>0:
                            item_ok = choice(okItems)
                            ok = self.dao.item[item_ok]

                            uf = self.PS_Set[user][item_ok]
                            self.optimization_thres(u, i, ok, user, uf)

                            item_j = choice(itemList)
                            while (self.PositiveSet[user].has_key(item_j) or self.JointSet[user].has_key(item_j) or self.PS_Set[user].has_key(item_j)):
                                item_j = choice(itemList)
                            j = self.dao.item[item_j]
                            self.optimization(u,ok,j)

                        elif len(kItems)>0 and len(okItems)==0:
                            item_k = choice(kItems)
                            uf = self.JointSet[user][item_k]
                            k = self.dao.item[item_k]
                            self.optimization_thres(u,i,k,user,uf)

                            item_j = choice(itemList)
                            while (self.PositiveSet[user].has_key(item_j) or self.JointSet[user].has_key(item_j) or self.PS_Set[user].has_key(item_j)):
                                item_j = choice(itemList)
                            j = self.dao.item[item_j]
                            self.optimization(u,k,j)

                        else:
                            item_j = choice(itemList)
                            while (self.PositiveSet[user].has_key(item_j) or self.JointSet[user].has_key(item_j) or
                                   self.PS_Set[user].has_key(item_j)):
                                item_j = choice(itemList)
                            j = self.dao.item[item_j]
                            self.optimization(u, i, j)
                        if len(nItems)>0:
                            item_n = choice(nItems)
                            n = self.dao.item[item_n]
                            self.optimization(u,j,n)

                if self.thres_count[user]>0:
                    self.threshold[user] -= self.lRate * self.thres_d[user] / self.thres_count[user]
                    self.thres_d[user]=0
                    self.thres_count[user]=0
                    li = [sim for sim in self.pSimilarity[user].values() if sim>=self.threshold[user]]
                    if len(li)==0:
                        self.avg_sim[user] = self.threshold[user]
                    else:
                        self.avg_sim[user]= sum(li)/(len(li)+0.0)


                for friend in self.trueTopKFriends[user]:
                    if self.pSimilarity[user][friend]>self.threshold[user]:
                        u = self.dao.user[user]
                        f = self.dao.user[friend]
                        self.P[u] -= self.alpha*self.lRate*(self.P[u]-self.P[f])

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

    def optimization_thres(self, u, i, j,user,friend):
        #print 'inner', (self.pSimilarity[user][friend]-self.threshold[user])/(self.avg_sim[user]-self.threshold[user])
        try:
            g_theta = sigmoid((self.pSimilarity[user][friend]-self.threshold[user])/(self.avg_sim[user]-self.threshold[user]))
        except OverflowError:
            print 'threshold',self.threshold[user],'smilarity',self.pSimilarity[user][friend],'avg',self.avg_sim[user]
            print (self.pSimilarity[user][friend]-self.threshold[user]),(self.avg_sim[user]-self.threshold[user])
            print (self.pSimilarity[user][friend]-self.threshold[user])/(self.avg_sim[user]-self.threshold[user])
            exit(-1)
        #print 'g_theta',g_theta

        s = sigmoid((self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))/(1+g_theta))
        self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
        self.Q[i] += self.lRate * (1 - s) * self.P[u]
        self.Q[j] -= self.lRate * (1 - s) * self.P[u]
        self.loss += -log(s)
        self.P[u] -= self.lRate * self.regU * self.P[u]
        self.Q[i] -= self.lRate * self.regI * self.Q[i]
        self.Q[j] -= self.lRate * self.regI * self.Q[j]
        t_derivative = -g_theta*(1-g_theta)*(1-s)*(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))\
                       *(self.pSimilarity[user][friend]-self.avg_sim[user])/(self.avg_sim[user]-self.threshold[user])**2/(1+g_theta)**2 + 0.01*self.threshold[user]
        #print 'derivative', t_derivative
        self.thres_d[user] += t_derivative
        self.thres_count[user] += 1
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