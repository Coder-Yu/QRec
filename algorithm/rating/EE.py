from baseclass.IterativeRecommender import IterativeRecommender
import numpy as np
from tool import config


class EE(IterativeRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]'):
        super(EE, self).__init__(conf, trainingSet, testSet, fold)

    # def readConfiguration(self):
    #     super(EE, self).readConfiguration()
    #     Dim = config.LineConfig(self.config['EE'])
    #     self.Dim = int(Dim['-d'])

    def initModel(self):
        super(EE, self).initModel()
        self.Bu = np.random.rand(self.dao.trainingSize()[0])/10  # bias value of user
        self.Bi = np.random.rand(self.dao.trainingSize()[1])/10  # bias value of item
        # self.X = np.random.rand(self.dao.trainingSize()[0], self.Dim)/10
        # self.Y = np.random.rand(self.dao.trainingSize()[1], self.Dim)/10

    def buildModel(self):
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                user, item, rating = entry
                error = rating - self.predict(user,item)
                u = self.dao.user[user]
                i = self.dao.item[item]
                self.loss += error ** 2
                self.loss += self.regU * (self.P[u] - self.Q[i]).dot(self.P[u] - self.Q[i])
                bu = self.Bu[u]
                bi = self.Bi[i]
                #self.loss += self.regB * bu ** 2 + self.regB * bi ** 2
                # update latent vectors
                self.P[u] -= self.lRate * (error + self.regU) * (self.P[u] - self.Q[i])
                self.Q[i] += self.lRate * (error + self.regI) * (self.P[u] - self.Q[i])
                self.Bu[u] += self.lRate * (error - self.regB * bu)
                self.Bi[i] += self.lRate * (error - self.regB * bi)
            self.loss+=self.regB*(self.Bu*self.Bu).sum()+self.regB*(self.Bi*self.Bi).sum()
            iteration += 1
            self.isConverged(iteration)

    def buildModel_tf(self):

        import tensorflow as tf

        u_idx = tf.placeholder(tf.int32, [None], name="u_idx")
        v_idx = tf.placeholder(tf.int32, [None], name="v_idx")
        r = tf.placeholder(tf.float32, [None], name="rating")
        global_mean = tf.placeholder(tf.float32, [None], name="mean")
        self.reg_lambda = tf.constant(self.regU, dtype=tf.float32)
        self.reg_biase = tf.constant(self.regB, dtype=tf.float32)

        m, n, train_size = self.dao.trainingSize()
        self.U = tf.Variable(tf.truncated_normal(shape=[m, self.k], stddev=0.005), name='U')
        self.V = tf.Variable(tf.truncated_normal(shape=[n, self.k], stddev=0.005), name='V')

        U_bias = tf.Variable(tf.truncated_normal(shape=[m], stddev=0.005,mean=0.02), name='U_bias')
        V_bias = tf.Variable(tf.truncated_normal(shape=[n], stddev=0.005,mean=0.02), name='V_bias')


        U_embed = tf.nn.embedding_lookup(self.U, u_idx)
        V_embed = tf.nn.embedding_lookup(self.V, v_idx)

        U_bias_embed = tf.nn.embedding_lookup(U_bias, u_idx)
        V_bias_embed = tf.nn.embedding_lookup(V_bias, v_idx)

        difference = tf.subtract(U_embed,V_embed)
        r_hat = tf.reduce_sum(tf.multiply(difference, difference), reduction_indices=1)
        r_hat = tf.subtract(U_bias_embed,r_hat)
        r_hat = tf.add(r_hat, V_bias_embed)
        r_hat = tf.add(r_hat, global_mean)

        loss = tf.nn.l2_loss(tf.subtract(r, r_hat))
        reg_loss = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(difference)),
                          tf.multiply(self.reg_lambda, tf.nn.l2_loss(difference)))
        reg_loss = tf.add(reg_loss,tf.multiply(self.reg_biase, tf.nn.l2_loss(U_bias)))
        reg_loss = tf.add(reg_loss, tf.multiply(self.reg_biase, tf.nn.l2_loss(V_bias)))

        total_loss = tf.add(loss, reg_loss)
        optimizer = tf.train.AdamOptimizer(self.lRate)
        train_U = optimizer.minimize(total_loss, var_list=[self.U, U_bias])
        train_V = optimizer.minimize(total_loss, var_list=[self.V, V_bias])

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)


            for step in range(self.maxIter):

                batch_size = self.batch_size

                batch_idx = np.random.randint(train_size, size=batch_size)

                user_idx = [self.dao.user[self.dao.trainingData[idx][0]] for idx in batch_idx]
                item_idx = [self.dao.item[self.dao.trainingData[idx][1]] for idx in batch_idx]
                g_mean = [self.dao.globalMean]*batch_size
                rating = [self.dao.trainingData[idx][2] for idx in batch_idx]

                sess.run(train_U, feed_dict={r: rating, u_idx: user_idx, v_idx: item_idx,global_mean:g_mean})
                sess.run(train_V, feed_dict={r: rating, u_idx: user_idx, v_idx: item_idx, global_mean: g_mean})

                print 'iteration:', step, 'loss:', sess.run(loss,
                                                            feed_dict={r: rating, u_idx: user_idx, v_idx: item_idx,global_mean:g_mean})


            self.P = sess.run(self.U)
            self.Q = sess.run(self.V)
            self.Bu = sess.run(U_bias)
            self.Bi = sess.run(V_bias)



    def predict(self, u, i):
        if self.dao.containsUser(u) and self.dao.containsItem(i):
            u = self.dao.user[u]
            i = self.dao.item[i]
            return self.dao.globalMean + self.Bi[i] + self.Bu[u] - (self.P[u] - self.Q[i]).dot(self.P[u] - self.Q[i])
        else:
            return self.dao.globalMean

    def predictForRanking(self,u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            u = self.dao.user[u]
            res = ((self.Q-self.P[u])*(self.Q-self.P[u])).sum(axis=1)+self.Bi+self.Bu[u]+self.dao.globalMean
            return res
        else:
            return [self.dao.globalMean]*len(self.dao.item)

