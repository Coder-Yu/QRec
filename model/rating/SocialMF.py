from base.socialRecommender import SocialRecommender
import numpy as np
import tensorflow as tf
class SocialMF(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=None,fold='[1]'):
        super(SocialMF, self).__init__(conf,trainingSet,testSet,relation,fold)

    def readConfiguration(self):
        super(SocialMF, self).readConfiguration()

    def trainModel(self):
        epoch = 0
        while epoch < self.maxEpoch:
            self.loss = 0
            for entry in self.data.trainingData:
                user, item, rating = entry
                u = self.data.user[user]
                i = self.data.item[item]
                error = rating - self.P[u].dot(self.Q[i])
                self.loss += error**2
                p = self.P[u].copy()
                q = self.Q[i].copy()
                self.P[u] += self.lRate * (error * q - self.regU * p)
                self.Q[i] += self.lRate * (error * p - self.regI * q)

            for user in self.social.user:
                if self.data.containsUser(user):
                    fPred = 0
                    denom = 0
                    u = self.data.user[user]
                    relationLoss = np.zeros(self.emb_size)
                    followees = self.social.getFollowees(user)
                    for followee in followees:
                        weight= followees[followee]
                        if self.data.containsUser(followee):
                            uf = self.data.user[followee]
                            fPred += weight * self.P[uf]
                            denom += weight
                    if denom != 0:
                        relationLoss = self.P[u] - fPred / denom
                    self.loss +=  self.regS *  relationLoss.dot(relationLoss)
                    # update latent vectors
                    self.P[u] -= self.lRate * self.regS * relationLoss
            self.loss+=self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()
            epoch += 1
            if self.isConverged(epoch):
                break

    def next_batch(self):
        batch_id=0
        while batch_id<self.train_size:
            if batch_id+self.batch_size<=self.train_size:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id,self.batch_size+batch_id)]
                items = [self.data.trainingData[idx][1] for idx in range(batch_id,self.batch_size+batch_id)]
                r = [self.data.trainingData[idx][2] for idx in range(batch_id,self.batch_size+batch_id)]
                batch_id+=self.batch_size
            else:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                items = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                r =  [self.data.trainingData[idx][2] for idx in range(batch_id, self.train_size)]
                batch_id=self.train_size

            u_idx,i_idx = [],[]
            for i,user in enumerate(users):
                i_idx.append(self.data.item[items[i]])
                u_idx.append(self.data.user[user])
            yield u_idx,i_idx,r

    def trainModel_tf(self):
        super(SocialMF, self).trainModel_tf()
        indices = [[self.data.user[entry[0]], self.data.user[entry[1]]] for entry in self.social.relation]
        values = [float(entry[2]) / len(self.social.followees[entry[0]]) for entry in self.social.relation]
        social_adj = tf.SparseTensor(indices=indices, values=values, dense_shape=[self.num_users, self.num_users])
        self.r = tf.placeholder(tf.float32)
        y = tf.reduce_sum(tf.multiply(self.user_embedding, self.item_embedding), 1)
        loss = tf.nn.l2_loss(tf.convert_to_tensor(self.r)-y) + self.regU * (tf.nn.l2_loss(self.user_embedding) + tf.nn.l2_loss(self.item_embedding))
        sr = 1*tf.nn.l2_loss(self.user_embedding-tf.gather(tf.sparse_tensor_dense_matmul(social_adj,self.U),self.u_idx))
        loss+=sr
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(self.maxEpoch):
                for n, batch in enumerate(self.next_batch()):
                    user_idx, i_idx,ratings = batch
                    _, l = sess.run([train, loss],
                                    feed_dict={self.u_idx: user_idx, self.v_idx: i_idx,self.r:ratings})
                    print('epoch:', epoch, 'loss:', l)
            self.P, self.Q = sess.run([self.U, self.V])
            import pickle
            f = open('user_embeddings', 'wb')
            pickle.dump(self.P, f)
            f = open('user_idx', 'wb')
            pickle.dump(self.data.user, f)