from baseclass.IterativeRecommender import IterativeRecommender
import numpy as np

class SVD(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(SVD, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(SVD, self).initModel()
        self.Bu = np.random.rand(self.dao.trainingSize()[0])/5  # bias value of user
        self.Bi = np.random.rand(self.dao.trainingSize()[1])/5  # bias value of item

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                user, item, rating = entry
                u = self.dao.user[user]
                i = self.dao.item[item]
                error = rating-self.predict(user,item)
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
            iteration += 1
            self.isConverged(iteration)


    def buildModel_tf(self):
        super(SVD, self).buildModel_tf()

        import tensorflow as tf

        global_mean = tf.placeholder(tf.float32, [None], name="mean")
        reg_lambda = tf.constant(self.regU, dtype=tf.float32)
        reg_biase = tf.constant(self.regB, dtype=tf.float32)



        self.U_bias = tf.Variable(tf.truncated_normal(shape=[self.m], stddev=0.005,mean=0.02), name='U_bias')
        self.V_bias = tf.Variable(tf.truncated_normal(shape=[self.n], stddev=0.005,mean=0.02), name='V_bias')

        self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
        self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.v_idx)

        self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), reduction_indices=1)
        self.r_hat = tf.add(self.r_hat, self.U_bias_embed)
        self.r_hat = tf.add(self.r_hat, self.V_bias_embed)
        self.r_hat = tf.add(self.r_hat, global_mean)

        self.loss = tf.nn.l2_loss(tf.subtract(self.r, self.r_hat))
        reg_loss = tf.add(tf.multiply(reg_lambda, tf.nn.l2_loss(self.U)),
                          tf.multiply(reg_lambda, tf.nn.l2_loss(self.V)))
        reg_loss = tf.add(reg_loss,tf.multiply(reg_biase, tf.nn.l2_loss(self.U_bias)))
        reg_loss = tf.add(reg_loss, tf.multiply(reg_biase, tf.nn.l2_loss(self.V_bias)))
        self.total_loss = tf.add(self.loss, reg_loss)
        optimizer = tf.train.AdamOptimizer(self.lRate)
        train_U = optimizer.minimize(self.total_loss, var_list=[self.U, self.U_bias])
        train_V = optimizer.minimize(self.total_loss, var_list=[self.V, self.V_bias])

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)


            for step in range(self.maxIter):

                batch_size = self.batch_size

                batch_idx = np.random.randint(self.train_size, size=batch_size)

                user_idx = [self.dao.user[self.dao.trainingData[idx][0]] for idx in batch_idx]
                item_idx = [self.dao.item[self.dao.trainingData[idx][1]] for idx in batch_idx]
                g_mean = [self.dao.globalMean]*batch_size
                rating = [self.dao.trainingData[idx][2] for idx in batch_idx]

                sess.run(train_U, feed_dict={self.r: rating, self.u_idx: user_idx, self.v_idx: item_idx,global_mean:g_mean})
                sess.run(train_V, feed_dict={self.r: rating, self.u_idx: user_idx, self.v_idx: item_idx, global_mean: g_mean})

                print 'iteration:', step, 'loss:', sess.run(self.total_loss,
                                                            feed_dict={self.r: rating, self.u_idx: user_idx, self.v_idx: item_idx,global_mean:g_mean})


            self.P = sess.run(self.U)
            self.Q = sess.run(self.V)
            self.Bu = sess.run(self.U_bias)
            self.Bi = sess.run(self.V_bias)

    def predict(self,u,i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            u = self.dao.user[u]
            i = self.dao.item[i]
            return self.P[u].dot(self.Q[i])+self.dao.globalMean+self.Bi[i]+self.Bu[u]
        else:
            return self.dao.globalMean

    def predictForRanking(self,u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            u = self.dao.getUserId(u)
            return self.Q.dot(self.P[u])+self.dao.globalMean + self.Bi + self.Bu[u]
        else:
            return [self.dao.globalMean] * len(self.dao.item)

