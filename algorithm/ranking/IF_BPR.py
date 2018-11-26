from baseclass.SocialRecommender import SocialRecommender
from baseclass.DeepRecommender import DeepRecommender
from tool import config
from random import shuffle, choice
from collections import defaultdict
import numpy as np
from tool.qmath import sigmoid, cosine
from math import log
import gensim.models.word2vec as w2v
from random import random
try:
    import tensorflow as tf
except ImportError:
    print 'TensorFlow is undetected on your computer!'


class IF_BPR(SocialRecommender,DeepRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        DeepRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

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
                if items[0] not in self.data.user:
                    self.data.user[items[0]]=len(self.data.user)
                    self.data.id2user[self.data.user[items[0]]] = items[0]

    def initModel(self):
        super(IF_BPR, self).initModel()
        super(DeepRecommender,self).initModel()
        super(SocialRecommender,self).initModel()
        self.positive = defaultdict(list)
        self.pItems = defaultdict(list)
        for user in self.data.trainSet_u:
            for item in self.data.trainSet_u[user]:
                self.positive[user].append(item)
                self.pItems[item].append(user)
        self.readNegativeFeedbacks()
        self.P = np.ones((len(self.data.user), self.k))*0.1  # latent user matrix
        self.threshold = {}
        self.avg_sim = {}
        self.thres_d = dict.fromkeys(self.data.user.keys(),0) #derivatives for learning thresholds
        self.thres_count = dict.fromkeys(self.data.user.keys(),0)

        print 'Preparing item sets...'
        self.PositiveSet = defaultdict(dict)
        self.NegSets = defaultdict(dict)

        for user in self.data.user:
            for item in self.data.trainSet_u[user]:
                self.PositiveSet[user][item] = 1

        for user in self.data.user:
            for item in self.negative[user]:
                if self.data.item.has_key(item):
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

        self.G = np.random.rand(self.data.trainingSize()[0], self.walkDim) * 0.1
        self.W = np.random.rand(self.data.trainingSize()[0], self.walkDim) * 0.1

        self.UFNet = defaultdict(list) # a -> b #a trusts b
        for u in self.social.followees:
            s1 = set(self.social.followees[u])
            for v in self.social.followees[u]:
                if v in self.social.followees:  # make sure that v has out links
                    if u <> v:
                        s2 = set(self.social.followees[v])
                        weight = len(s1.intersection(s2))
                        self.UFNet[u] += [v] * (weight + 1)

        self.UTNet = defaultdict(list) # a <- b #a is trusted by b
        for u in self.social.followers:
            s1 = set(self.social.followers[u])
            for v in self.social.followers[u]:
                if self.social.followers.has_key(v):  # make sure that v has out links
                    if u <> v:
                        s2 = set(self.social.followers[v])
                        weight = len(s1.intersection(s2))
                        self.UTNet[u] += [v] * (weight + 1)

        print 'Generating random meta-path random walks... (Positive)'
        self.pWalks = []
        # self.usercovered = {}

        # positive
        for user in self.data.user:
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
                                        while not self.data.user.has_key(nextNode):
                                            nextNode = choice(self.UFNet[lastNode])
                                    elif lastType == 'T':
                                        nextNode = choice(self.UTNet[lastNode])
                                        while not self.data.user.has_key(nextNode):
                                            nextNode = choice(self.UTNet[lastNode])

                                if tp == 'F':
                                    nextNode = choice(self.UFNet[lastNode])
                                    while not self.data.user.has_key(nextNode):
                                        nextNode = choice(self.UFNet[lastNode])

                                if tp == 'T':
                                    nextNode = choice(self.UFNet[lastNode])
                                    while not self.data.user.has_key(nextNode):
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
        for user in self.data.user:
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
                                        while not self.data.user.has_key(nextNode):
                                            nextNode = choice(self.UFNet[lastNode])
                                    elif lastType == 'T':
                                        nextNode = choice(self.UTNet[lastNode])
                                        while not self.data.user.has_key(nextNode):
                                            nextNode = choice(self.UTNet[lastNode])

                                if tp == 'F':
                                    nextNode = choice(self.UFNet[lastNode])
                                    while not self.data.user.has_key(nextNode):
                                        nextNode = choice(self.UFNet[lastNode])

                                if tp == 'T':
                                    nextNode = choice(self.UFNet[lastNode])
                                    while not self.data.user.has_key(nextNode):
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
        pos_model = w2v.Word2Vec(self.pWalks, size=self.walkDim, window=5, min_count=0, iter=10)
        neg_model = w2v.Word2Vec(self.nWalks, size=self.walkDim, window=5, min_count=0, iter=10)
        for user in self.positive:
            uid = self.data.user[user]
            try:
                self.W[uid] = pos_model.wv['U' + user]
            except KeyError:
                continue
        for user in self.negative:
            uid = self.data.user[user]
            try:
                self.G[uid] = neg_model.wv['U' + user]
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
            vec1 = self.W[self.data.user[user1]]
            for user2 in self.positive:
                if user1 <> user2:
                    vec2 = self.W[self.data.user[user2]]
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
            vec1 = self.G[self.data.user[user1]]
            for user2 in self.negative:
                if user1 <> user2:
                    vec2 = self.G[self.data.user[user2]]
                    sim = cosine(vec1, vec2)
                    uSim.append((user2, sim))
            fList = sorted(uSim, key=lambda d: d[1], reverse=True)[:self.topK]
            for pair in fList:
                self.nSimilarity[user1][pair[0]] = pair[1]
            self.nTopKSim[user1] = [item[0] for item in fList]

        self.trueTopKFriends = defaultdict(list)
        for user in self.pTopKSim:
            trueFriends = list(set(self.pTopKSim[user]).intersection(set(self.nTopKSim[user])))
            self.trueTopKFriends[user] = trueFriends
            self.pTopKSim[user] = list(set(self.pTopKSim[user]).difference(set(trueFriends)))

    def updateSets(self):
        self.JointSet = defaultdict(dict)
        self.PS_Set = defaultdict(dict)
        for user in self.data.user:
            if user in self.trueTopKFriends:
                for friend in self.trueTopKFriends[user]:
                    if friend in self.data.user and self.pSimilarity[user][friend] >= self.threshold[user]:
                        for item in self.positive[friend]:
                            if item not in self.PositiveSet[user] and item not in self.NegSets[user]:
                                self.JointSet[user][item] = friend

            if self.pTopKSim.has_key(user):
                for friend in self.pTopKSim[user][:self.topK]:
                    if friend in self.data.user and self.pSimilarity[user][friend] >= self.threshold[user]:
                        for item in self.positive[friend]:
                            if item not in self.PositiveSet[user] and item not in self.JointSet[user] \
                                    and item not in self.NegSets[user]:
                                self.PS_Set[user][item] = friend

            if self.nTopKSim.has_key(user):
                for friend in self.nTopKSim[user][:self.topK]:
                    if friend in self.data.user:  # and self.nSimilarity[user][friend]>=self.threshold[user]:
                        for item in self.negative[friend]:
                            if item in self.data.item:
                                if item not in self.PositiveSet[user] and item not in self.JointSet[user] \
                                        and item not in self.PS_Set[user]:
                                    self.NegSets[user][item] = 1

    def buildModel(self):

        self.randomWalks()
        self.computeSimilarity()

        print 'Decomposing...'
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            self.updateSets()
            itemList = self.data.item.keys()
            for user in self.PositiveSet:
                #itemList = self.NegSets[user].keys()
                kItems = self.JointSet[user].keys()
                pItems = self.PS_Set[user].keys()
                nItems = self.NegSets[user].keys()

                u = self.data.user[user]

                for item in self.PositiveSet[user]:
                    i = self.data.item[item]
                    selectedItems = [i]
                    #select items from different sets
                    if len(kItems) > 0:
                        item_k = choice(kItems)                        
                        uf = self.JointSet[user][item_k]
                        k = self.data.item[item_k]
                        selectedItems.append(k)
                        self.optimization_thres(u,i,k,user,uf)
                    if len(pItems)>0:
                        item_p = choice(pItems)
                        p = self.data.item[item_p]
                        selectedItems.append(p)

                    item_r = choice(itemList)
                    while item_r in self.PositiveSet[user] or item_r in self.JointSet[user]\
                        or item_r in self.PS_Set[user] or item_r in self.NegSets[user]:
                        item_r = choice(itemList)
                    r = self.data.item[item_r]
                    selectedItems.append(r)

                    if len(nItems)>0:
                        item_n = choice(nItems)
                        n = self.data.item[item_n]
                        selectedItems.append(n)

                    #optimization
                    for ind,item in enumerate(selectedItems[:-1]):
                        self.optimization(u,item,selectedItems[ind+1])


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
                        u = self.data.user[user]
                        f = self.data.user[friend]
                        self.P[u] -= self.alpha*self.lRate*(self.P[u]-self.P[f])

            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break

        self.advTraining()

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

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.Q.dot(self.P[u])
        else:
            return [self.data.globalMean] * len(self.data.item)


    ###############################################################################################

    #Implementation Based on TensorFlow

    ###############################################################################################
    # def _create_variables(self):
    #     self.neg_idx = tf.placeholder(tf.int32, [None], name="n_idx")
    #     self.pSim =  tf.placeholder(tf.float32, [None], name="pSim")
    #     self.avgSim = tf.placeholder(tf.float32, [None], name="avgSim")
    #
    #     self.V_neg_embed = tf.nn.embedding_lookup(self.V, self.neg_idx)
    #     simVec = np.random.rand(len(self.data.user))
    #     for i,user in enumerate(self.data.user):
    #         simVec[self.data.user[user]]=self.threshold[user]
    #     self.sim_thres = tf.Variable(simVec,dtype=tf.float32,trainable=True)
    #     self.U_sim = tf.nn.embedding_lookup(self.sim_thres, self.u_idx)
    #
    # def _create_inference(self):
    #     result = tf.subtract(tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), 1),
    #                               tf.reduce_sum(tf.multiply(self.U_embed, self.V_neg_embed), 1))
    #     return result
    #
    # def _create_adaptive_inference(self):
    #     g = tf.nn.sigmoid(tf.divide (tf.clip_by_value(tf.subtract(self.pSim,self.U_sim),1e-8, 1.0),
    #                                  tf.clip_by_value(tf.subtract(self.avgSim,self.U_sim),1e-8, 1.0)))
    #     self.adapEfficients = tf.clip_by_value(tf.add(tf.constant(1.0,dtype=tf.float32),g),1.0,2.0)
    #     result = tf.subtract(tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed),1),
    #                         tf.reduce_sum(tf.multiply(self.U_embed, self.V_neg_embed), 1))
    #     result = tf.divide(result, self.adapEfficients)
    #     return result
    #
    # def _create_loss(self):
    #     self.reg_lambda = tf.constant(self.regU, dtype=tf.float32)
    #     self.reg_T = tf.constant(0.02, dtype=tf.float32)
    #     self.loss = tf.reduce_sum(tf.nn.softplus(-self._create_inference()))
    #     self.reg_loss = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U_embed)),
    #                            tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V_embed)))
    #     self.total_loss = tf.add(self.loss, self.reg_loss)
    #     self.adapLoss = tf.add(tf.reduce_sum(tf.nn.softplus(-self._create_adaptive_inference())),self.reg_loss)
    #     self.adapLoss = tf.add(self.adapLoss,tf.multiply(self.reg_T,tf.nn.l2_loss(self.U_sim)))
    #
    #
    # def _create_optimizer(self):
    #     self.optimizer = tf.train.AdamOptimizer(self.lRate)
    #     self.train = self.optimizer.minimize(self.total_loss)
    #     self.adapOptimizer = tf.train.AdamOptimizer(self.lRate)
    #     self.adapTrain = self.optimizer.minimize(self.adapLoss)
    #
    # def next_adap_batch(self):
    #     batch_idx = np.random.randint(self.train_size, size=self.batch_size)
    #     users = [self.data.trainingData[idx][0] for idx in batch_idx]
    #     items = [self.data.trainingData[idx][1] for idx in batch_idx]
    #     user_idx, item_idx, neg_item_idx,friend_idx = [], [], [],[]
    #
    #     for i, user in enumerate(users):
    #         kItems = self.JointSet[user].keys()
    #         uid = self.data.user[user]
    #         pid = self.data.item[items[i]]
    #
    #         if len(kItems)>0:
    #             item_k = choice(kItems)
    #             user_idx.append(uid)
    #             item_idx.append(pid)
    #             neg_item_idx.append(self.data.item[item_k])
    #             friend_idx.append(self.data.user[self.JointSet[user][item_k]])
    #
    #     return user_idx,item_idx,neg_item_idx,friend_idx
    #
    #
    # def next_batch(self):
    #     batch_idx = np.random.randint(self.train_size, size=self.batch_size)
    #
    #     users = [self.data.trainingData[idx][0] for idx in batch_idx]
    #     items = [self.data.trainingData[idx][1] for idx in batch_idx]
    #
    #     user_idx,item_idx, neg_item_idx = [], [], []
    #
    #     itemList = self.data.item.keys()
    #
    #     for i,user in enumerate(users):
    #         kItems = self.JointSet[user].keys()
    #         pItems = self.PS_Set[user].keys()
    #         nItems = self.NegSets[user].keys()
    #         uid = self.data.user[user]
    #         iid = self.data.item[items[i]]
    #         group = []
    #         #group.append(iid)
    #         if len(kItems)>0:
    #             item_k = choice(kItems)
    #             group.append(self.data.item[item_k])
    #         if len(pItems)>0:
    #             item_p = choice(pItems)
    #             group.append(self.data.item[item_p])
    #         item_r = choice(itemList)
    #         while item_r in self.PositiveSet[user] or item_r in self.JointSet[user] \
    #                 or item_r in self.PS_Set[user] or item_r in self.NegSets[user]:
    #             item_r = choice(itemList)
    #         group.append(self.data.item[item_r])
    #         if len(nItems)>0:
    #             item_n = choice(nItems)
    #             group.append(self.data.item[item_n])
    #
    #         for ind, item in enumerate(group[:-1]):
    #             user_idx.append(uid)
    #             item_idx.append(group[ind])
    #             neg_item_idx.append(group[ind+1])
    #         # for ind, item in enumerate(group[1:]):
    #         #     user_idx.append(uid)
    #         #     item_idx.append(iid)
    #         #     neg_item_idx.append(group[ind])
    #
    #     return user_idx,item_idx,neg_item_idx


    # def buildModel_tf(self):
    #
    #     super(IF_BPR, self).buildModel_tf()
    #     self.randomWalks()
    #     self.computeSimilarity()

        # print 'Decomposing...'
        # self._create_variables()
        # self._create_loss()
        # self._create_optimizer()
        # # train the model until converged
        # init = tf.global_variables_initializer()
        # self.sess.run(init)
        #
        # for epoch in range(self.maxIter):
        #     if epoch % 10 == 0:
        #         simArray = self.sess.run(self.sim_thres)
        #         for i,sim in enumerate(simArray):
        #             self.threshold[self.data.id2user[i]]=sim
        #
        #         for user in self.data.user:
        #             li = [sim for sim in self.pSimilarity[user].values() if sim >= self.threshold[user]]
        #             if len(li) == 0:
        #                 self.avg_sim[user] = self.threshold[user]
        #             else:
        #                 self.avg_sim[user] = sum(li) / (len(li) + 0.0)
        #         self.updateSets()
        #
        #     #print self.foldInfo, self.threshold.values()[:10]
        #     # if epoch% 30==0:
        #     #     self.ranking_performance()
        #     user_idx, item_idx, neg_item_idx,friend_idx = self.next_adap_batch()
        #     Suv = []
        #     avgSim = []
        #     for u,f in zip(user_idx,friend_idx):
        #         Suv.append(self.pSimilarity[self.data.id2user[u]][self.data.id2user[f]])
        #         avgSim.append(self.avg_sim[self.data.id2user[u]])
        #
        #     _, loss,effi = self.sess.run([self.adapTrain, self.adapLoss,self.adapEfficients],
        #                        feed_dict={self.u_idx: user_idx, self.v_idx: item_idx,
        #                                   self.neg_idx: neg_item_idx,self.avgSim:avgSim, self.pSim:Suv})
        #     print self.foldInfo,'iteration:', epoch, 'adaptive loss:', loss
        #     print self.foldInfo,'iteration:',epoch, effi[:20]
        #
        #     user_idx, item_idx, neg_item_idx = self.next_batch()
        #     _, loss = self.sess.run([self.train, self.total_loss],
        #                        feed_dict={self.u_idx: user_idx, self.v_idx: item_idx, self.neg_idx: neg_item_idx})
        #     print self.foldInfo,'iteration:', epoch, 'loss:', loss
        #
        #     self.P = self.sess.run(self.U)
        #     self.Q = self.sess.run(self.V)

###############################################################################################

# Adversarial Training

###############################################################################################
    # def readParameters(self):
    #     args = config.LineConfig(self.config['AT'])
    #     self.eps = float(args['-eps'])
    #     self.regAdv = float(args['-regA'])
    #     self.advEpoch = int(args['-advEpoch'])
    #
    #
    #
    # def _create_variables(self):
    #     #perturbation vectors
    #     self.adv_U = tf.Variable(tf.zeros(shape=[self.m, self.k]),dtype=tf.float32, trainable=False)
    #     self.adv_V = tf.Variable(tf.zeros(shape=[self.n, self.k]),dtype=tf.float32, trainable=False)
    #
    #     self.neg_idx = tf.placeholder(tf.int32, [None], name="n_idx")
    #     self.V_neg_embed = tf.nn.embedding_lookup(self.V, self.neg_idx)
    #     #parameters
    #     self.eps = tf.constant(self.eps,dtype=tf.float32)
    #     self.regAdv = tf.constant(self.regAdv,dtype=tf.float32)
    #
    # def _create_inference(self):
    #     result = tf.subtract(tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), 1),
    #                               tf.reduce_sum(tf.multiply(self.U_embed, self.V_neg_embed), 1))
    #     return result
    #
    # def _create_adv_inference(self):
    #     self.U_plus_delta = tf.add(self.U_embed, tf.nn.embedding_lookup(self.adv_U, self.u_idx))
    #     self.V_plus_delta = tf.add(self.V_embed, tf.nn.embedding_lookup(self.adv_V, self.v_idx))
    #     self.V_neg_plus_delta = tf.add(self.V_neg_embed, tf.nn.embedding_lookup(self.adv_V, self.neg_idx))
    #     result = tf.subtract(tf.reduce_sum(tf.multiply(self.U_plus_delta, self.V_plus_delta), 1),
    #                          tf.reduce_sum(tf.multiply(self.U_plus_delta, self.V_neg_plus_delta), 1))
    #     return result
    #
    # def _create_adversarial(self):
    #     #get gradients of Delta
    #     self.grad_U, self.grad_V = tf.gradients(self.loss_adv, [self.adv_U,self.adv_V])
    #
    #     # convert the IndexedSlice Data to Dense Tensor
    #     self.grad_U_dense = tf.stop_gradient(self.grad_U)
    #     self.grad_V_dense = tf.stop_gradient(self.grad_V)
    #
    #     # normalization: new_grad = (grad / |grad|) * eps
    #     self.update_U = self.adv_U.assign(tf.nn.l2_normalize(self.grad_U_dense, 1) * self.eps)
    #     self.update_V = self.adv_V.assign(tf.nn.l2_normalize(self.grad_V_dense, 1) * self.eps)
    #
    #
    # def _create_loss(self):
    #     self.reg_lambda = tf.constant(self.regU, dtype=tf.float32)
    #     self.loss = tf.reduce_sum(tf.nn.softplus(-self._create_inference()))
    #     self.reg_loss = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U_embed)),
    #                            tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V_embed)))
    #
    #     self.total_loss = tf.add(self.loss, self.reg_loss)
    #     #loss of adversarial training
    #     self.loss_adv = tf.multiply(self.regAdv,tf.reduce_sum(tf.nn.softplus(-self._create_adv_inference())))
    #     self.loss_adv = tf.add(self.loss,self.loss_adv)
    #
    # def _create_optimizer(self):
    #     self.optimizer = tf.train.AdamOptimizer(0.002)
    #     self.train = self.optimizer.minimize(self.total_loss)
    #
    #     self.optimizer_adv = tf.train.AdamOptimizer(0.002)
    #     self.train_adv = self.optimizer.minimize(self.loss_adv)
    #
    #
    #
    # def next_batch(self):
    #     batch_idx = np.random.randint(self.train_size, size=self.batch_size)
    #     itemList = self.data.item.keys()
    #     users = [self.data.trainingData[idx][0] for idx in batch_idx]
    #     items = [self.data.trainingData[idx][1] for idx in batch_idx]
    #     user_idx,item_idx=[],[]
    #     neg_item_idx = []
    #     for i, user in enumerate(users):
    #         kItems = self.JointSet[user].keys()
    #         pItems = self.PS_Set[user].keys()
    #         nItems = self.NegSets[user].keys()
    #         uid = self.data.user[user]
    #         iid = self.data.item[items[i]]
    #         group = []
    #         group.append(iid)
    #         if len(kItems)>0:
    #             item_k = choice(kItems)
    #             group.append(self.data.item[item_k])
    #         if len(pItems)>0:
    #             item_p = choice(pItems)
    #             group.append(self.data.item[item_p])
    #         item_r = choice(itemList)
    #         while item_r in self.PositiveSet[user] or item_r in self.JointSet[user] \
    #                 or item_r in self.PS_Set[user] or item_r in self.NegSets[user]:
    #             item_r = choice(itemList)
    #         group.append(self.data.item[item_r])
    #         if len(nItems)>0:
    #             item_n = choice(nItems)
    #             group.append(self.data.item[item_n])
    #
    #         for ind, item in enumerate(group[:-1]):
    #             user_idx.append(uid)
    #             item_idx.append(group[ind])
    #             neg_item_idx.append(group[ind+1])
    #         # for ind, item in enumerate(group[1:]):
    #         #     user_idx.append(uid)
    #         #     item_idx.append(iid)
    #         #     neg_item_idx.append(group[ind])
    #
    #     return user_idx,item_idx,neg_item_idx
    #
    #
    # def advTraining(self):
    #     self.readParameters()
    #     self._create_variables()
    #     self._create_loss()
    #     self._create_adversarial()
    #     self._create_optimizer()
    #     print 'adversarial training...'
    #     self.U = tf.Variable(self.P)
    #     self.V = tf.Variable(self.Q)
    #     with tf.Session() as sess:
    #         init = tf.global_variables_initializer()
    #         sess.run(init)
    #
    #         # # train the model until converged
    #         # for epoch in range(self.maxIter):
    #         #
    #         #     user_idx,item_idx,neg_item_idx = self.next_batch()
    #         #     _,loss = sess.run([self.train,self.total_loss],feed_dict={self.u_idx: user_idx, self.v_idx: item_idx, self.neg_idx:neg_item_idx})
    #         #     print 'iteration:', epoch, 'loss:',loss
    #         #
    #         #
    #         #     self.P = sess.run(self.U)
    #         #     self.Q = sess.run(self.V)
    #         #     #self.ranking_performance()
    #
    #         # start adversarial training
    #         for epoch in range(self.advEpoch):
    #
    #             user_idx,item_idx,neg_item_idx = self.next_batch()
    #             sess.run([self.update_U, self.update_V],
    #                      feed_dict={self.u_idx: user_idx, self.v_idx: item_idx, self.neg_idx: neg_item_idx})
    #             _,loss = sess.run([self.train_adv,self.loss_adv],feed_dict={self.u_idx: user_idx, self.v_idx: item_idx, self.neg_idx:neg_item_idx})
    #
    #             print 'iteration:', epoch, 'loss:',loss
    #
    #             self.P = sess.run(self.U)
    #             self.Q = sess.run(self.V)
    #             #self.ranking_performance()
       