from baseclass.IterativeRecommender import IterativeRecommender
import numpy as np

################# Confidence Frequency Matrix Factorization #################
#                   Weighted Rating Matrix  Factorization                   #
# this algorithm refers to the following paper:
# Yifan Hu et al.Collaborative Filtering for Implicit Feedback Datasets,2008 IEEE
class My(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(My, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(My, self).initModel()
        self.alpha = float(self.config['alpha'])
        self.lamba = float(self.config['lamba'])

    def buildModel(self):
        iteration = 0
        #user_item_matrix with contains operation frequency(rating here) of m users for n items
        user_item_matrix = np.zeros([len(self.P), len(self.Q)])
        while iteration < self.maxIter:
            self.loss = 0
            ################# handle the user-factors #################
            #                                                        #
            # Y is item-factors matrix with n X f
            Y = self.Q
            Y_Transpose_Y = Y.T.dot(Y)
            for i in range(len(self.P)):
                user = self.dao.id2user[i]

                #Ru is the operation frequency of user 'u' for items
                Ru = np.zeros([1, len(self.dao.item)])
                itemsDictOfCurrentUser = self.dao.trainSet_u.get(user)
                for key in itemsDictOfCurrentUser:
                    Ru[0, self.dao.item[key]] = itemsDictOfCurrentUser[key]
                user_item_matrix[i]=Ru

                #Cu is confidence of user 'u' for its preference for items
                Cu=1+self.alpha*Ru

                #Pu is preference of user 'u' for items, and '1' indicates favor
                Pu = np.zeros([1, len(self.dao.item)])
                Pu[Ru>0]=1

                #Xu is user-factors of user 'u'
                Xu=( np.matrix(Y_Transpose_Y+(Y.T*(Cu-1)).dot(Y) + self.lamba*np.eye(self.k) ).I ).dot((Y.T*Cu).dot(Pu.T))
                self.P[i]=Xu.getA1()


            ################# handle the item-factors #################
            #                                                        #
            # X is user-factors matrix with m X f
            X = self.P
            X_Transpose_X = X.T.dot(X)
            for i in range(len(self.Q)):
                item =self.dao.id2item[i]

                #Ri is the operation frequency of item 'i' by users
                Ri = np.zeros([1, len(self.dao.user)])
                usersDictOfCurrentItem = self.dao.trainSet_i.get(item)
                for key in usersDictOfCurrentItem:
                    Ri[0, self.dao.user[key]] = usersDictOfCurrentItem[key]

                # Ci is confidence of item 'i' favored by users
                Ci = 1 + self.alpha * Ri

                # Pi is preference of item 'i' favored by users, and '1' indicates favor
                Pi = np.zeros([1, len(self.dao.user)])
                Pi[Ri > 0] = 1

                # Yi is item-factors of item 'i'
                Yi = (np.matrix(X_Transpose_X + (X.T * (Ci - 1)).dot(X) + self.lamba*np.eye(self.k) ).I).dot((X.T * Ci).dot(Pi.T))
                self.Q[i] = Yi.getA1()

            ################# cost error #################
            #                                            #
            error = user_item_matrix-self.P.dot(self.Q.T)
            self.loss+=(error*error).sum()
            self.loss += self.lamba*(self.P*self.P).sum() + self.lamba*(self.Q*self.Q).sum()

            iteration += 1
            self.isConverged(iteration)

    # def predict(self,user,item):
    #     if self.dao.containsUser(user) and self.dao.containsItem(item):
    #         userLocationIndex = self.dao.user[user]
    #         itemLocationIndex = self.dao.item[item]
    #         return self.P[userLocationIndex].dot(self.Q[itemLocationIndex])
    #     else:
    #         return self.dao.globalMean

    def predictForRanking(self,u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            u = self.dao.getUserId(u)
            return self.Q.dot(self.P[u])+self.dao.globalMean
        else:
            return [self.dao.globalMean] * len(self.dao.item)