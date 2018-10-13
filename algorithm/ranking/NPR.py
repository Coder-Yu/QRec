#coding:utf8
from baseclass.DeepRecommender import DeepRecommender
from baseclass.SocialRecommender import SocialRecommender
import numpy as np
import random
from tool import config
from collections import defaultdict
try:
    import tensorflow as tf
except ImportError:
    print 'This method can only be run tensorflow!'
    exit(-1)
from tensorflow import set_random_seed
set_random_seed(2)

class NPR(SocialRecommender,DeepRecommender):

    # NPR：Adversarial Personalized Ranking for Recommendation

    def __init__(self,conf,trainingSet=None,testSet=None,relation=None,fold='[1]'):
        DeepRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self,conf=conf,trainingSet=trainingSet,testSet=testSet,relation=relation,fold=fold)


    # def readConfiguration(self):
    #     super(NPR, self).readConfiguration()
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

        #build friend-item set
        self.FSet_dict = defaultdict(dict)
        self.FSet_list = defaultdict(list)

        for user in self.sao.followees:
            for friend in self.sao.followees[user]:
                for item in self.dao.userRated(friend)[0]:
                    if not self.dao.trainSet_u[user].has_key(item):
                        self.FSet_dict[user][item]=1
                        self.FSet_list[user].append(item)

        self.users_without_relations = []
        for user in self.dao.user:
            if not self.FSet_dict.has_key(user):
                self.users_without_relations.append(user)

    def readConfiguration(self):
        super(NPR, self).readConfiguration()
        args = config.LineConfig(self.config['NPR'])
        self.eps = float(args['-eps'])
        self.regAdv = float(args['-regA'])
        self.advEpoch = int(args['-advEpoch'])
        self.negativeCount = 2


    def _create_variables(self):
        #perturbation vectors
        self.adv_U = tf.Variable(tf.zeros(shape=[self.m, self.k]),dtype=tf.float32, trainable=False)
        self.adv_V = tf.Variable(tf.zeros(shape=[self.n, self.k]),dtype=tf.float32, trainable=False)

        self.neg_idx = tf.placeholder(tf.int32, [None], name="n_idx")
        self.V_neg_embed = tf.nn.embedding_lookup(self.V, self.neg_idx)

        self.friend_item_idx = tf.placeholder(tf.int32, [None], name="f_idx")
        self.V_friend_embed = tf.nn.embedding_lookup(self.V, self.friend_item_idx)

        #parameters
        self.eps = tf.constant(self.eps,dtype=tf.float32)
        self.regAdv = tf.constant(self.regAdv,dtype=tf.float32)

    def _create_inference(self):
        result = tf.subtract(tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), 1),
                                  tf.reduce_sum(tf.multiply(self.U_embed, self.V_neg_embed), 1))
        return result

    def _create_friend_inference(self):
        result = tf.subtract(tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), 1),
                             tf.reduce_sum(tf.multiply(self.U_embed, self.V_friend_embed), 1))
        result = tf.reduce_sum(tf.nn.softplus(-result))
        result2 = tf.subtract(tf.reduce_sum(tf.multiply(self.U_embed, self.V_friend_embed), 1),
                             tf.reduce_sum(tf.multiply(self.U_embed, self.V_neg_embed), 1))
        result = tf.add(result,tf.reduce_sum(tf.nn.softplus(-result2)))

        return result

    def _create_adv_friend_inference(self):
        self.U_plus_delta = tf.add(self.U_embed, tf.nn.embedding_lookup(self.adv_U, self.u_idx))
        self.V_friend_plus_delta = tf.add(self.V_friend_embed, tf.nn.embedding_lookup(self.adv_V, self.friend_item_idx))
        self.V_neg_plus_delta = tf.add(self.V_embed, tf.nn.embedding_lookup(self.adv_V, self.neg_idx))


        result = tf.subtract(tf.reduce_sum(tf.multiply(self.U_plus_delta, self.V_friend_plus_delta), 1),
                             tf.reduce_sum(tf.multiply(self.U_plus_delta, self.V_neg_plus_delta), 1))
        return result

    def _create_adv_inference(self):
        self.U_plus_delta = tf.add(self.U_embed, tf.nn.embedding_lookup(self.adv_U, self.u_idx))
        self.V_plus_delta = tf.add(self.V_embed, tf.nn.embedding_lookup(self.adv_V, self.v_idx))
        self.V_neg_plus_delta = tf.add(self.V_neg_embed, tf.nn.embedding_lookup(self.adv_V, self.neg_idx))
        result = tf.subtract(tf.reduce_sum(tf.multiply(self.U_plus_delta, self.V_plus_delta), 1),
                             tf.reduce_sum(tf.multiply(self.U_plus_delta, self.V_neg_plus_delta), 1))
        return result

    def _create_adversarial(self):
        #get gradients of Delta
        self.grad_U, self.grad_V = tf.gradients(self.loss_adv, [self.adv_U,self.adv_V])

        # convert the IndexedSlice Data to Dense Tensor
        self.grad_U_dense = tf.stop_gradient(self.grad_U)
        self.grad_V_dense = tf.stop_gradient(self.grad_V)

        # normalization: new_grad = (grad / |grad|) * eps
        self.update_U = self.adv_U.assign(tf.nn.l2_normalize(self.grad_U_dense, 1) * self.eps)
        self.update_V = self.adv_V.assign(tf.nn.l2_normalize(self.grad_V_dense, 1) * self.eps)


    def _create_loss(self):
        self.reg_lambda = tf.constant(self.regU, dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.nn.softplus(-self._create_inference()))
        self.reg_loss = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U_embed)),
                               tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V_embed)))

        self.reg_loss = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U_embed)), self.reg_loss)
        self.total_loss = tf.add(self.loss, self.reg_loss)

        self.loss_friend = self._create_friend_inference()
        self.total_friend_loss = tf.add(self.loss_friend, self.reg_loss)

        #loss of adversarial training
        self.loss_adv = tf.multiply(self.regAdv,tf.reduce_sum(tf.nn.softplus(-self._create_adv_friend_inference())))
        self.loss_adv = tf.add(self.loss_friend,self.loss_adv)

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self.lRate)
        self.train = self.optimizer.minimize(self.total_loss)

        self.optimizer_friend = tf.train.AdamOptimizer(self.lRate)
        self.train_friend = self.optimizer.minimize(self.total_friend_loss)

        self.optimizer_adv = tf.train.AdamOptimizer(self.lRate)
        self.train_adv = self.optimizer.minimize(self.loss_adv)


    def initModel(self):
        super(NPR, self).initModel()
        self._create_variables()
        self._create_loss()
        self._create_adversarial()
        self._create_optimizer()


    def next_batch(self):
        batch_idx = np.random.randint(self.train_size, size=self.batch_size)

        users = [self.dao.trainingData[idx][0] for idx in batch_idx]
        items = [self.dao.trainingData[idx][1] for idx in batch_idx]
        user_idx,item_idx=[],[]
        friend_item_idx,neg_item_idx = [],[]
        for i,user in enumerate(users):
            if not self.FSet_dict.has_key(user):
                continue
            f_item = random.choice(self.FSet_list[user])
            while self.dao.trainSet_u[user].has_key(f_item):
                f_item = random.choice(self.FSet_list[user])
            for j in range(self.negativeCount): #negative sampling
                item_j = random.randint(0,self.n-1)
                while self.dao.trainSet_u[user].has_key(self.dao.id2item[item_j]) and self.FSet_dict[user].has_key(item_j):
                    item_j = random.randint(0, self.n - 1)
                user_idx.append(self.dao.user[user])
                item_idx.append(self.dao.item[items[i]])
                friend_item_idx.append(self.dao.item[f_item])
                neg_item_idx.append(item_j)

        return user_idx,item_idx,friend_item_idx,neg_item_idx


    def next_batch_v2(self):
        batch_idx = np.random.randint(self.train_size, size=self.batch_size)

        users = [self.dao.trainingData[idx][0] for idx in batch_idx]
        items = [self.dao.trainingData[idx][1] for idx in batch_idx]
        user_idx,item_idx=[],[]
        neg_item_idx = []
        for i,user in enumerate(users):
            for j in range(self.negativeCount): #negative sampling
                item_j = random.randint(0,self.n-1)

                while self.dao.trainSet_u[user].has_key(self.dao.id2item[item_j]):
                    item_j = random.randint(0, self.n - 1)

                user_idx.append(self.dao.user[user])
                item_idx.append(self.dao.item[items[i]])
                neg_item_idx.append(item_j)

        return user_idx,item_idx,neg_item_idx

    def buildModel(self):
        print 'training...'
        iteration = 0
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            # train the model until converged
            for epoch in range(self.maxIter):
                #user_idx, item_idx, neg_item_idx = self.next_batch_v2()
                #_,loss = sess.run([self.train,self.total_loss],feed_dict={self.u_idx: user_idx, self.v_idx: item_idx, self.neg_idx:neg_item_idx})

                user_idx, item_idx, friend_item_idx, neg_item_idx = self.next_batch()
                _,loss = sess.run([self.train_friend, self.total_friend_loss],feed_dict={self.u_idx: user_idx, self.v_idx: item_idx, self.friend_item_idx:friend_item_idx,self.neg_idx: neg_item_idx})

                print 'iteration:', epoch, 'loss:',loss

                self.P = sess.run(self.U)
                self.Q = sess.run(self.V)
                self.ranking_performance()

            # start adversarial training
            for epoch in range(self.advEpoch):

                user_idx, item_idx, neg_item_idx = self.next_batch_v2()
                _, loss = sess.run([self.train, self.total_loss],
                                   feed_dict={self.u_idx: user_idx, self.v_idx: item_idx, self.neg_idx: neg_item_idx})

                user_idx,item_idx,friend_item_idx,neg_item_idx = self.next_batch()
                sess.run([self.update_U, self.update_V],
                         feed_dict={self.u_idx: user_idx, self.v_idx: item_idx, self.friend_item_idx:friend_item_idx,self.neg_idx: neg_item_idx})
                _,loss = sess.run([self.train_adv,self.loss_adv],feed_dict={self.u_idx: user_idx, self.v_idx: item_idx,self.friend_item_idx:friend_item_idx, self.neg_idx:neg_item_idx})

                print 'iteration:', epoch, 'loss:',loss

                self.P = sess.run(self.U)
                self.Q = sess.run(self.V)
                self.ranking_performance()


    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            u = self.dao.getUserId(u)
            return self.Q.dot(self.P[u])
        else:
            return [self.dao.globalMean] * len(self.dao.item)


