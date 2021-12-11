# coding:utf8
from base.deepRecommender import DeepRecommender
from base.socialRecommender import SocialRecommender
import numpy as np
from random import choice,shuffle
from collections import defaultdict
import tensorflow as tf
import gensim.models.word2vec as w2v
from util.qmath import cosine

def gumbel_softmax(logits, temperature=0.2):
    eps = 1e-20
    u = tf.random_uniform(tf.shape(logits), minval=0, maxval=1)
    gumbel_noise = -tf.log(-tf.log(u + eps) + eps)
    y = logits + gumbel_noise
    return tf.nn.softmax(y / temperature)

class RSGAN(SocialRecommender,DeepRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        DeepRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)
    def readNegativeFeedbacks(self):
        self.negative = defaultdict(list)
        self.nItems = defaultdict(list)
        filename = self.config['ratings'][:-4]+'_n.txt'
        with open(filename) as f:
            for line in f:
                items = line.strip().split()
                if items[0] not in self.data.user:
                    continue
                self.negative[items[0]].append(items[1])
                self.nItems[items[1]].append(items[0])

    def randomWalks(self):
        self.positive = defaultdict(list)
        self.pItems = defaultdict(list)
        for user in self.data.trainSet_u:
            for item in self.data.trainSet_u[user]:
                self.positive[user].append(item)
                self.pItems[item].append(user)

        # build U-F-NET
        print('Building weighted user-friend network...')
        # Definition of Meta-Path
        p1 = 'UIU'
        p2 = 'UFU'
        p3 = 'UTU'
        p4 = 'UFIU'
        p5 = 'UFUIU'
        mPaths = [p1, p2, p3, p4, p5]
        mPathCnt = [10,8,8,5,5]
        mPathSetting = list(zip(mPaths,mPathCnt))
        walkLen=20

        self.G = np.random.rand(self.num_users, 50) * 0.1
        self.W = np.random.rand(self.num_users, 50) * 0.1

        self.UFNet = defaultdict(list) # a -> b #a trusts b
        for u in self.social.followees:
            s1 = set(self.social.followees[u])
            for v in self.social.followees[u]:
                if v in self.social.followees:  # make sure that v has out links
                    if u != v:
                        s2 = set(self.social.followees[v])
                        weight = len(s1.intersection(s2))
                        self.UFNet[u] += [v] * (weight + 1)

        self.UTNet = defaultdict(list) # a <- b #a is trusted by b
        for u in self.social.followers:
            s1 = set(self.social.followers[u])
            for v in self.social.followers[u]:
                if v in self.social.followers:  # make sure that v has out links
                    if u != v:
                        s2 = set(self.social.followers[v])
                        weight = len(s1.intersection(s2))
                        self.UTNet[u] += [v] * (weight + 1)
        
        # positive
        print('Generating random meta-path random walks... (Positive)')
        self.pWalks = []
        for user in self.data.user:
            for mps in mPathSetting:
                mp,walkCnt = mps
                for t in range(walkCnt):
                    path = ['U' + user]
                    lastNode = user
                    nextNode = user
                    lastType = 'U'
                    for i in range(walkLen//len(mp[1:])):
                        for tp in mp[1:]:
                            try:
                                if tp == 'I':
                                    nextNode = choice(self.positive[lastNode])

                                if tp == 'U':
                                    if lastType == 'I':
                                        nextNode = choice(self.pItems[lastNode])
                                    elif lastType == 'F':
                                        nextNode = choice(self.UFNet[lastNode])
                                        while nextNode not in self.data.user:
                                            nextNode = choice(self.UFNet[lastNode])
                                    elif lastType == 'T':
                                        nextNode = choice(self.UTNet[lastNode])
                                        while nextNode not in self.data.user:
                                            nextNode = choice(self.UTNet[lastNode])
                                if tp == 'F':
                                    nextNode = choice(self.UFNet[lastNode])
                                    while nextNode not in self.data.user:
                                        nextNode = choice(self.UFNet[lastNode])
                                if tp == 'T':
                                    nextNode = choice(self.UFNet[lastNode])
                                    while nextNode not in self.data.user:
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
        # negative
        for user in self.data.user:
            for mps in mPathSetting:
                mp,walkCnt = mps
                for t in range(walkCnt):
                    path = ['U' + user]
                    lastNode = user
                    nextNode = user
                    lastType = 'U'
                    for i in range(walkLen // len(mp[1:])):
                        for tp in mp[1:]:
                            try:
                                if tp == 'I':
                                    nextNode = choice(self.negative[lastNode])
                                if tp == 'U':
                                    if lastType == 'I':
                                        nextNode = choice(self.nItems[lastNode])
                                    elif lastType == 'F':
                                        nextNode = choice(self.UFNet[lastNode])
                                        while nextNode not in self.data.user:
                                            nextNode = choice(self.UFNet[lastNode])
                                    elif lastType == 'T':
                                        nextNode = choice(self.UTNet[lastNode])
                                        while nextNode not in self.data.user:
                                            nextNode = choice(self.UTNet[lastNode])
                                if tp == 'F':
                                    nextNode = choice(self.UFNet[lastNode])
                                    while nextNode not in self.data.user:
                                        nextNode = choice(self.UFNet[lastNode])
                                if tp == 'T':
                                    nextNode = choice(self.UFNet[lastNode])
                                    while nextNode not in self.data.user:
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
        shuffle(self.nWalks)
        print('pwalks:', len(self.pWalks))
        print('nwalks:', len(self.nWalks))

    def computeSimilarity(self):
        # Training get top-k friends
        print('Generating user embedding...')
        topK = 100
        self.pTopKSim = {}
        self.nTopKSim = {}

        pos_model = w2v.Word2Vec(self.pWalks, size=50, window=5, min_count=0, iter=10)
        neg_model = w2v.Word2Vec(self.nWalks, size=50, window=5, min_count=0, iter=10)
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
        print('User embedding generated.')

        print('Constructing similarity matrix...')

        for i,user1 in enumerate(self.positive):
            uSim = []
            if i % 200 == 0:
                print(i, '/', len(self.positive))
            vec1 = self.W[self.data.user[user1]]
            for user2 in self.positive:
                if user1 != user2:
                    vec2 = self.W[self.data.user[user2]]
                    sim = cosine(vec1, vec2)
                    uSim.append((user2, sim))
            fList = sorted(uSim, key=lambda d: d[1], reverse=True)[:topK]
            self.pTopKSim[user1] = [item[0] for item in fList]

        for i,user1 in enumerate(self.negative):
            uSim = []
            if i % 200 == 0:
                print(i, '/', len(self.negative))
            vec1 = self.G[self.data.user[user1]]
            for user2 in self.negative:
                if user1 != user2:
                    vec2 = self.G[self.data.user[user2]]
                    sim = cosine(vec1, vec2)
                    uSim.append((user2, sim))
            fList = sorted(uSim, key=lambda d: d[1], reverse=True)[:topK]
            self.nTopKSim[user1] = [item[0] for item in fList]

        self.seededFriends = defaultdict(list)
        for user in self.pTopKSim:
            trueFriends = list(set(self.pTopKSim[user]).intersection(set(self.nTopKSim[user])))
            self.seededFriends[user] = trueFriends+self.pTopKSim[user][:30]

    def sampling(self,vec):
        vec = tf.nn.softmax(vec)
        logits = gumbel_softmax(vec, 0.1)
        return logits

    def build_graph(self):
        indices = [[self.data.item[item[1]], self.data.user[item[0]]] for item in self.data.trainingData]
        values = [item[2] for item in self.data.trainingData]
        self.i_u_matrix = tf.SparseTensor(indices=indices, values=values, dense_shape=[self.num_items, self.num_users])
        self.pos = tf.placeholder(tf.int32, name="positive_item")
        self.fnd = tf.placeholder(tf.int32, name="friend_item")
        self.neg = tf.placeholder(tf.int32, name="neg_holder")
        self.i = tf.placeholder(tf.int32, name="item_holder")

        with tf.name_scope("generator"):
            #CDAE
            initializer = tf.contrib.layers.xavier_initializer()
            self.X = tf.placeholder(tf.float32, [None, self.num_users])
            self.V = tf.Variable(initializer([self.num_users, 200]))
            chosen_user_embeddings = tf.nn.embedding_lookup(self.V,self.u_idx)

            self.weights = {
                'encoder': tf.Variable(initializer([self.num_users, 200])),
                'decoder': tf.Variable(initializer([200, self.num_users])),
            }
            self.biases = {
                'encoder': tf.Variable(initializer([200])),
                'decoder': tf.Variable(initializer([self.num_users])),
            }

            self.g_params = [self.weights, self.biases,self.V]

            layer = tf.nn.sigmoid(tf.matmul(self.X, self.weights['encoder']) + self.biases['encoder']+chosen_user_embeddings)
            self.g_output = tf.nn.sigmoid(tf.matmul(layer, self.weights['decoder']) + self.biases['decoder'])

            self.y_pred = tf.multiply(self.X, self.g_output)
            self.y_pred = tf.maximum(1e-6, self.y_pred)

            cross_entropy = -tf.multiply(self.X, tf.log(self.y_pred)) - tf.multiply((1 - self.X),
                                                                                    tf.log(1 - self.y_pred))
            self.reconstruction = tf.reduce_sum(cross_entropy) + self.regU * (
                    tf.nn.l2_loss(self.weights['encoder']) + tf.nn.l2_loss(self.weights['decoder']) +
                    tf.nn.l2_loss(self.biases['encoder']) + tf.nn.l2_loss(self.biases['decoder']))
            g_pre = tf.train.AdamOptimizer(self.lRate)
            self.g_pretrain = g_pre.minimize(self.reconstruction, var_list=self.g_params)

        with tf.variable_scope('discriminator'):
            self.item_selection = tf.get_variable('item_selection',initializer=tf.constant_initializer(0.01),shape=[self.num_users, self.num_items])
            self.g_params.append(self.item_selection)
            self.d_params = [self.user_embeddings, self.item_embeddings]
            # placeholder definition
            self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u_idx,name='u_e')
            self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.pos,name='i_e')
            self.j_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.neg,name='j_e')
            #generate virtual friends by gumbel-softmax
            self.virtualFriends = self.sampling(self.g_output) #one-hot
            #get candidate list (items)
            self.candidateItems = tf.transpose(tf.sparse_tensor_dense_matmul(self.i_u_matrix,tf.transpose(self.virtualFriends)))
            self.embedding_selection = tf.nn.embedding_lookup(self.item_selection, self.u_idx,name='e_s')
            self.virtual_items = self.sampling(tf.multiply(self.candidateItems,self.embedding_selection))
            self.v_i_embedding = tf.matmul(self.virtual_items,self.item_embeddings,transpose_a=False,transpose_b=False)
            y_us = tf.reduce_sum(tf.multiply(self.u_embedding,self.i_embedding),1)\
                                 -tf.reduce_sum(tf.multiply(self.u_embedding,self.j_embedding),1)
            self.d_pretrain_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y_us)))+self.regU*(tf.nn.l2_loss(self.u_embedding)+
                                                                                       tf.nn.l2_loss(self.j_embedding)+
                                                                                       tf.nn.l2_loss(self.i_embedding))
            y_uf = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) - \
                 tf.reduce_sum(tf.multiply(self.u_embedding, self.v_i_embedding), 1)
            y_fs = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_i_embedding), 1)-\
                 tf.reduce_sum(tf.multiply(self.u_embedding, self.j_embedding), 1)
            self.d_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y_uf)))-tf.reduce_sum(tf.log(tf.sigmoid(y_fs)))+\
                          self.regU*(tf.nn.l2_loss(self.u_embedding)+tf.nn.l2_loss(self.i_embedding)+tf.nn.l2_loss(self.j_embedding))            #
            self.g_loss = 30*tf.reduce_sum(y_uf) #better performance
            d_pre = tf.train.AdamOptimizer(self.lRate)
            self.d_pretrain = d_pre.minimize(self.d_pretrain_loss, var_list=self.d_params)
            self.d_output = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings),1)
        d_opt = tf.train.AdamOptimizer(self.lRate)
        self.d_update = d_opt.minimize(self.d_loss,var_list=self.d_params)
        g_opt = tf.train.AdamOptimizer(self.lRate)
        self.g_update = g_opt.minimize(self.g_loss,var_list=self.g_params)

    def next_batch_g(self):
        userList = list(self.data.user.keys())
        batch_id=0
        while batch_id<self.num_users:
            if batch_id + self.batch_size <= self.num_users:
                sampled_users = []
                profiles = np.zeros((self.batch_size, self.num_users))
                for i,user in enumerate(userList[batch_id:self.batch_size+batch_id]):
                    ind = [self.data.user[friend] for friend in self.seededFriends[user]]
                    profiles[i][ind]=1
                    sampled_users.append(self.data.user[user])
                batch_id+=self.batch_size
            else:
                profiles = []
                sampled_users = []
                for i, user in enumerate(userList[self.num_users-batch_id:]):
                    vals = np.zeros(self.num_users)
                    ind = [self.data.user[friend] for friend in self.seededFriends[user]]
                    vals[ind]=1
                    profiles.append(vals)
                    sampled_users.append(self.data.user[user])
                batch_id=self.num_users
            yield profiles,sampled_users

    def initModel(self):
        super(RSGAN, self).initModel()
        self.readNegativeFeedbacks()
        self.randomWalks()
        self.computeSimilarity()
        self.build_graph()

    def trainModel(self):
        # minimax training
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # pretraining
        print('pretraining for generator...')
        for i in range(30):
            for num,batch in enumerate(self.next_batch_g()):
                profiles,uid = batch
                _,loss = self.sess.run([self.g_pretrain,self.reconstruction],feed_dict={self.X:profiles,self.u_idx:uid})
                print('pretraining:', i + 1, 'batch',num,'generator loss:', loss)

        print('Training GAN...')
        for i in range(self.maxEpoch):
            for num,batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                profiles = np.zeros((len(user_idx),self.num_users))
                for n,u in enumerate(user_idx):
                    u_name = self.data.id2user[u]
                    idx = [self.data.user[friend] for friend in self.seededFriends[u_name]]
                    profiles[n][idx]=1
                #generator
                _,loss = self.sess.run([self.g_update,self.g_loss],feed_dict={self.u_idx: user_idx,self.neg:j_idx,
                                                   self.pos: i_idx,self.X:profiles})
                #discriminator
                _, loss = self.sess.run([self.d_update, self.d_loss],
                                        feed_dict={self.u_idx: user_idx,self.neg:j_idx,self.pos: i_idx,self.X:profiles})

                print('training:', i + 1, 'batch_id', num, 'discriminator loss:', loss)

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.user[u]
            res = self.sess.run(self.d_output, {self.u_idx:[u]})
            return res
        else:
            return [self.data.globalMean] * self.num_items



