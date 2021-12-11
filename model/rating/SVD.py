from base.iterativeRecommender import IterativeRecommender
import numpy as np

class SVD(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(SVD, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(SVD, self).initModel()
        self.Bu = np.random.rand(self.data.trainingSize()[0])/5  # bias value of user
        self.Bi = np.random.rand(self.data.trainingSize()[1])/5  # bias value of item

    def trainModel(self):
        epoch = 0
        while epoch < self.maxEpoch:
            self.loss = 0
            for entry in self.data.trainingData:
                user, item, rating = entry
                u = self.data.user[user]
                i = self.data.item[item]
                error = rating-self.predictForRating(user, item)
                self.loss+=error**2
                p = self.P[u]
                q = self.Q[i]
                bu = self.Bu[u]
                bi = self.Bi[i]
                #update latent vectors
                self.P[u] += self.lRate*(error*q-self.regU*p)
                self.Q[i] += self.lRate*(error*p-self.regI*q)
                self.Bu[u] += self.lRate*(error-self.regB*bu)
                self.Bi[i] += self.lRate*(error-self.regB*bi)
            self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()\
               +self.regB*((self.Bu*self.Bu).sum()+(self.Bi*self.Bi).sum())
            epoch += 1
            self.isConverged(epoch)

    def trainModel_tf(self):
        super(SVD, self).trainModel_tf()
        import tensorflow as tf
        global_mean = tf.placeholder(tf.float32, [None], name="mean")
        self.U_bias = tf.Variable(tf.truncated_normal(shape=[self.num_users], stddev=0.005), name='U_bias')
        self.V_bias = tf.Variable(tf.truncated_normal(shape=[self.num_items], stddev=0.005), name='V_bias')
        self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
        self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.v_idx)
        self.r_hat = tf.reduce_sum(tf.multiply(self.user_embedding, self.item_embedding), axis=1)
        self.r_hat = self.r_hat + self.U_bias_embed
        self.r_hat = self.r_hat + self.V_bias_embed
        self.r_hat = self.r_hat + global_mean
        self.loss = tf.nn.l2_loss(self.r-self.r_hat)
        reg_loss = self.regU * tf.nn.l2_loss(self.user_embedding) + self.regI * tf.nn.l2_loss(self.item_embedding)
        reg_loss += self.regB*self.U_bias_embed+ self.regB*self.U_bias_embed
        self.total_loss = self.loss + reg_loss
        optimizer = tf.train.AdamOptimizer(self.lRate)
        train_U = optimizer.minimize(self.total_loss, var_list=[self.U, self.U_bias])
        train_V = optimizer.minimize(self.total_loss, var_list=[self.V, self.V_bias])
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for step in range(self.maxEpoch):
                batch_size = self.batch_size
                batch_idx = np.random.randint(self.train_size, size=batch_size)
                user_idx = [self.data.user[self.data.trainingData[idx][0]] for idx in batch_idx]
                item_idx = [self.data.item[self.data.trainingData[idx][1]] for idx in batch_idx]
                g_mean = [self.data.globalMean]*batch_size
                rating = [self.data.trainingData[idx][2] for idx in batch_idx]
                sess.run(train_U, feed_dict={self.r: rating, self.u_idx: user_idx, self.v_idx: item_idx,global_mean:g_mean})
                sess.run(train_V, feed_dict={self.r: rating, self.u_idx: user_idx, self.v_idx: item_idx, global_mean: g_mean})

                print('epoch:', step, 'loss:', sess.run(self.total_loss,
                                                            feed_dict={self.r: rating, self.u_idx: user_idx, self.v_idx: item_idx,global_mean:g_mean}))
            self.P = sess.run(self.U)
            self.Q = sess.run(self.V)
            self.Bu = sess.run(self.U_bias)
            self.Bi = sess.run(self.V_bias)

    def predictForRating(self, u, i):
        if self.data.containsUser(u) and self.data.containsItem(i):
            u = self.data.user[u]
            i = self.data.item[i]
            return self.P[u].dot(self.Q[i])+self.data.globalMean+self.Bi[i]+self.Bu[u]
        else:
            return self.data.globalMean

    def predictForRanking(self,u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.Q.dot(self.P[u])+self.data.globalMean + self.Bi + self.Bu[u]
        else:
            return [self.data.globalMean] * self.num_items

