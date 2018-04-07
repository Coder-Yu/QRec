from baseclass.SocialRecommender import SocialRecommender
from tool import config
from random import randint
from random import shuffle, choice
from collections import defaultdict
import numpy as np
from tool.qmath import sigmoid, cosine, cosine_sp
from math import log
import gensim.models.word2vec as w2v


class HERP(SocialRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        super(HERP, self).__init__(conf, trainingSet, testSet, relation, fold)

    def readConfiguration(self):
        super(HERP, self).readConfiguration()
        options = config.LineConfig(self.config['HERP'])
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
        super(HERP, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['recommender'] + ':'
        print 'Walks count per user', self.walkCount
        print 'Length of each walk', self.walkLength
        print 'Dimension of user embedding', self.walkDim
        print '=' * 80

    def initModel(self):
        super(HERP, self).initModel()
        tobeCleaned = []
        for user in self.dao.testSet_u:
            for item in self.dao.testSet_u[user]:
                if self.dao.testSet_u[user][item]<4.5:
                    tobeCleaned.append((user,item))

        for pair in tobeCleaned:
            del self.dao.testSet_u[pair[0]][pair[1]]
            if len(self.dao.testSet_u[pair[0]])==0:
                del self.dao.testSet_u[pair[0]]

        self.positive = defaultdict(list)
        self.negative = defaultdict(list)
        self.pItems = defaultdict(list)
        self.nItems = defaultdict(list)
        for user in self.dao.trainSet_u:
            for item in self.dao.trainSet_u[user]:
                if self.dao.trainSet_u[user][item]>0.85:
                    self.positive[user].append(item)
                    self.pItems[item].append(user)
                elif self.dao.trainSet_u[user][item]<0.5:
                    self.negative[user].append(item)
                    self.nItems[item].append(user)



    def buildModel(self):
        self.P = np.ones((self.dao.trainingSize()[0], self.k))/10  # latent user matrix
        self.Q = np.ones((self.dao.trainingSize()[1], self.k))/10  # latent item matrix
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

        self.G = np.random.rand(self.dao.trainingSize()[0], self.walkDim) / 10
        self.W = np.random.rand(self.dao.trainingSize()[0], self.walkDim) / 10





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
        print 'Generating random meta-path random walks... (Positive)'
        self.pWalks = []
        #self.usercovered = {}

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

                    path = ['U'+user]
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

                                path.append(tp+nextNode)
                                lastNode = nextNode
                                lastType = tp

                            except (KeyError, IndexError):
                                path = []
                                break

                    if path:
                        self.pWalks.append(path)
                        # for node in path:
                        #     if node[1] == 'U' or node[1] == 'F':
                        #         self.usercovered[node[0]] = 1
                                # print path
                                # if mp == 'UFIU':
                                # pass
        self.nWalks = []
        # self.usercovered = {}

        #negative
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
                        # for node in path:
                        #     if node[1] == 'U' or node[1] == 'F':
                        #         self.usercovered[node[0]] = 1
                        # print path
                        # if mp == 'UFIU':
                        # pass
        shuffle(self.pWalks)
        print 'pwalks:', len(self.pWalks)
        print 'nwalks:', len(self.nWalks)
        # Training get top-k friends
        print 'Generating user embedding...'

        self.pTopKSim = {}
        self.nTopKSim = {}

        model = w2v.Word2Vec(self.pWalks, size=self.walkDim, window=5, min_count=0, iter=10)
        model2 = w2v.Word2Vec(self.nWalks, size=self.walkDim, window=5, min_count=0, iter=10)

        for user in self.positive:
            uid = self.dao.user[user]
            try:
                self.W[uid] = model.wv['U'+user]
            except KeyError:
                continue

        for user in self.negative:
            uid = self.dao.user[user]
            try:
                self.G[uid] = model2.wv['U'+user]
            except KeyError:
                continue
        print 'User embedding generated.'

        print 'Constructing similarity matrix...'
        i = 0


        for user1 in self.positive:
            uSim = []
            i+=1
            if i%200==0:
                print i,'/',len(self.positive)
            vec1 = self.W[self.dao.user[user1]]
            for user2 in self.positive:
                if user1 <> user2:
                    vec2 = self.W[self.dao.user[user2]]
                    sim = cosine(vec1, vec2)
                    uSim.append((user2,sim))
            fList = sorted(uSim, key=lambda d: d[1], reverse=True)[:self.topK]
            self.pTopKSim[user1] = [item[0] for item in fList]

        i=0
        for user1 in self.negative:
            uSim = []
            i+=1
            if i%200==0:
                print i,'/',len(self.negative)
            vec1 = self.G[self.dao.user[user1]]
            for user2 in self.negative:
                if user1 <> user2:
                    vec2 = self.G[self.dao.user[user2]]
                    sim = cosine(vec1, vec2)
                    uSim.append((user2,sim))
            fList = sorted(uSim, key=lambda d: d[1], reverse=True)[:self.topK]
            self.nTopKSim[user1] = [item[0] for item in fList]


        self.trueTopKFriends=defaultdict(list)
        for user in self.pTopKSim:
            trueFriends = list(set(self.pTopKSim[user]).intersection(set(self.nTopKSim[user])))
            self.trueTopKFriends[user] = trueFriends
            if len(trueFriends)>0:
                print trueFriends



        # print 'Similarity matrix finished.'
        # # # #print self.topKSim
        #import pickle
        # # # #
        # # # #recordTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # similarity = open('HERP-lastfm-sim'+self.foldInfo+'.pkl', 'wb')
        # vectors = open('HERP-lastfm-vec'+self.foldInfo+'.pkl', 'wb')
        # #Pickle dictionary using protocol 0.
        #
        # pickle.dump(self.topKSim, similarity)
        # pickle.dump((self.W,self.G),vectors)
        # similarity.close()
        # vectors.close()

        # matrix decomposition
        #pkl_file = open('HERP-lastfm-sim' + self.foldInfo + '.pkl', 'rb')

        #self.topKSim = pickle.load(pkl_file)

        print 'Decomposing...'
        self.F = np.random.rand(self.dao.trainingSize()[0], self.k) / 10
        # prepare Pu set, IPu set, and Nu set
        print 'Preparing item sets...'
        self.PositiveSet = defaultdict(dict)
        self.IPositiveSet = defaultdict(dict)
        self.OKSet = defaultdict(dict)


        for user in self.dao.user:
            for item in self.dao.trainSet_u[user]:
                if self.dao.trainSet_u[user][item] >= 1:
                    self.PositiveSet[user][item] = 1
                    # else:
                    #     self.NegativeSet[user].append(item)
            if self.trueTopKFriends.has_key(user):
                for friend in self.trueTopKFriends[user][:self.topK]:
                    if self.dao.user.has_key(friend):
                        for item in self.dao.trainSet_u[friend]:
                            if not self.PositiveSet[user].has_key(item):
                                if not self.IPositiveSet[user].has_key(item):
                                    self.IPositiveSet[user][item] = 1
                                else:
                                    self.IPositiveSet[user][item] += 1

            if self.pTopKSim.has_key(user):
                for friend in self.pTopKSim[user][:self.topK]:
                    if self.dao.user.has_key(friend):
                        for item in self.dao.trainSet_u[friend]:
                            if not self.PositiveSet[user].has_key(item) and not self.IPositiveSet[user].has_key(item):
                                if not self.OKSet[user].has_key(item):
                                    self.OKSet[user][item] = 1
                                else:
                                    self.OKSet[user][item] += 1
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            itemList = self.dao.item.keys()

            for user in self.PositiveSet:
                kItems = self.IPositiveSet[user].keys()
                okItems = self.OKSet[user].keys()
                u = self.dao.user[user]
                for item in self.PositiveSet[user]:
                    i = self.dao.item[item]
                    if len(self.IPositiveSet[user]) > 0:
                        item_k = choice(kItems)
                        k = self.dao.item[item_k]
                        Suk = self.IPositiveSet[user][item_k]
                        self.P[u] += (1 / (Suk + 1)) * self.lRate * (1 - sigmoid(
                            (self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[k])) / (Suk + 1))) \
                                     * (self.Q[i] - self.Q[k])
                        self.Q[i] += (1 / (Suk + 1)) * self.lRate * (1 - sigmoid(
                            (self.P[u].dot(self.Q[i])- self.P[u].dot(self.Q[k])) / (
                                Suk + 1))) * \
                                     self.P[u]
                        self.Q[k] -= (1 / (Suk + 1)) * self.lRate * (1 - sigmoid(
                            (self.P[u].dot(self.Q[i])- self.P[u].dot(self.Q[k])) / (
                                Suk + 1))) * self.P[u]
                        item_j = ''
                        # if len(self.NegativeSet[user])>0:
                        #     item_j = choice(self.NegativeSet[user])
                        # else:
                        if len(okItems)==0:
                            continue
                        item_ok = choice(okItems)
                        ok = self.dao.item[item_ok]
                        Sok = self.OKSet[user][item_ok]
                        self.P[u] += self.lRate * (1 - sigmoid(
                            (self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[ok])))) \
                                     * (self.Q[i] - self.Q[ok])
                        self.Q[k] += self.lRate * (1 - sigmoid(
                            (self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[ok])))) * self.P[u]
                        self.Q[ok] -= self.lRate * (1 - sigmoid(
                            (self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[ok])))) * self.P[u]





                        item_j = choice(itemList)
                        while (self.PositiveSet[user].has_key(item_j) or self.IPositiveSet[user].has_key(item_j) or self.OKSet[user].has_key(item_j)):
                            item_j = choice(itemList)
                        j = self.dao.item[item_j]
                        self.P[u] += self.lRate * (
                            1 - sigmoid(self.P[u].dot(self.Q[ok]) - self.P[u].dot(self.Q[j]))) * (
                                         self.Q[ok] - self.Q[j])
                        self.Q[ok] += self.lRate * (
                            1 - sigmoid(self.P[u].dot(self.Q[ok])- self.P[u].dot(self.Q[j]))) * \
                                     self.P[u]
                        self.Q[j] -= self.lRate * (
                            1 - sigmoid(self.P[u].dot(self.Q[ok])  - self.P[u].dot(self.Q[j]))) * \
                                     self.P[u]

                        self.P[u] -= self.lRate * self.regU * self.P[u]
                        self.Q[i] -= self.lRate * self.regI * self.Q[i]
                        self.Q[j] -= self.lRate * self.regI * self.Q[j]
                        self.Q[k] -= self.lRate * self.regI * self.Q[k]
                        self.Q[ok] -= self.lRate * self.regI * self.Q[ok]
                        self.loss += -log(sigmoid(
                            (self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[k]) ) / (Suk + 1))) \
                                     - log(sigmoid(self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[ok])))\
                                     - log(sigmoid(self.P[u].dot(self.Q[ok]) - self.P[u].dot(self.Q[j])))
                    else:
                        item_j = choice(itemList)
                        while (self.PositiveSet[user].has_key(item_j)):
                            item_j = choice(itemList)
                        j = self.dao.item[item_j]
                        self.P[u] += self.lRate * (
                            1 - sigmoid(self.P[u].dot(self.Q[i])  - self.P[u].dot(self.Q[j]) )) * (
                                         self.Q[i] - self.Q[j])
                        self.Q[i] += self.lRate * (
                            1 - sigmoid(self.P[u].dot(self.Q[i])  - self.P[u].dot(self.Q[j]) )) * \
                                     self.P[u]
                        self.Q[j] -= self.lRate * (
                            1 - sigmoid(self.P[u].dot(self.Q[i])  - self.P[u].dot(self.Q[j]) )) * \
                                     self.P[u]


                        self.loss += -log(
                            sigmoid(self.P[u].dot(self.Q[i])  - self.P[u].dot(self.Q[j])))

                        self.loss += -log(sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j])))

            for user in self.trueTopKFriends:
                for friend in self.trueTopKFriends[user]:
                    u = self.dao.user[user]
                    f = self.dao.user[friend[0]]
                    self.P[u] -= self.alpha*self.lRate*(self.P[u]-self.P[f])

            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break

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