from baseclass.IterativeRecommender import IterativeRecommender

class HTreeNode(object):
    def __init__(self,left,right,freq,id):
        self.left = left
        self.right = right
        self.frequency = freq
        self.id = id

class HuffmanTree(object):
    def __init__(self,root=None):
        self.root = root

    def buildTree(self,nodeList):
        while(len(nodeList)):
            for

    def traverse(self):


class CUNE_MF(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(CUNE_MF, self).__init__(conf,trainingSet,testSet,fold)
        self.nonLeafVec = {}
        self.leafVec = {}
        self.topKSim = {}

    def buildModel(self):
        #build Huffman Tree First
        #get frequency
        print 'Building Huffman tree...'
        print 'Computing frequency...'
        #To accelerate the method, the frequency is estimated roughly
        self.frequency = {}
        for user in self.dao.trainSet_u:
            self.frequency[user] = len(self.dao.trainSet_u[user])


        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for entry in self.dao.trainingData:
                user, item, rating = entry
                u = self.dao.user[user] #get user id
                i = self.dao.item[item] #get item id
                error = rating - self.P[u].dot(self.Q[i])
                self.loss += error**2
                p = self.P[u]
                q = self.Q[i]

                #update latent vectors
                self.P[u] += self.lRate*(error*q-self.regU*p)
                self.Q[i] += self.lRate*(error*p-self.regI*q)

            self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break

