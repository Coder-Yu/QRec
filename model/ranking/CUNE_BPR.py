
from base.iterativeRecommender import IterativeRecommender
from util import config
from random import shuffle,choice
from collections import defaultdict
import numpy as np
from util.qmath import sigmoid,cosine
from math import log
import gensim.models.word2vec as w2v


# class Node(object):
#     def __init__(self):
#         self.val = 0
#         self.last = None
#         self.next = None
#
# class OrderedLinkList(object):
#     def __init__(self):
#         self.head=None
#         self.tail=None
#         self.length = 0
#
#     def __len__(self):
#         return self.length
#
#     def insert(self,node):
#         self.length+=1
#         if self.head:
#             tmp = self.head
#             while tmp.val < node.val:
#                 if tmp==self.tail:
#                     break
#                 tmp = tmp.next
#
#             if tmp is self.head:
#
#                 if self.head.val < node.val:
#                     node.next = self.head
#                     self.head.last = node
#                     self.head = node
#                 else:
#                     node.next = self.head
#                     self.head.last = node
#                     self.head = node
#                 return
#
#             node.next = tmp.next
#             tmp.next = node
#             node.last = tmp
#             if not node.next:
#                 self.tail = node
#
#         else:
#             self.head = node
#             self.tail = node
#
#     def removeHead(self):
#         if self.head:
#             self.head = self.head.next
#             self.length -= 1
#
#     def removeNode(self,node):
#         if self.head:
#             tmp = self.head
#             while tmp is not node and tmp.next:
#                 tmp = tmp.next
#             if tmp.next:
#                 tmp.last.next = tmp.next
#                 tmp.next.last = tmp.last
#             self.length-=1
#
#
# class HTreeNode(object):
#     def __init__(self,left,right,freq,id,code=None):
#         self.left = left
#         self.right = right
#         self.weight = freq
#         self.id = id
#         self.code = code
#
#     def __lt__(self, other):
#         if self.weight < other.weight:
#             return True
#         else:
#             return False
#
# class HuffmanTree(object):
#     def __init__(self,root=None,vecLength=10):
#         self.root = root
#         self.weight = 0
#         self.code = {}
#         self.vecLength = vecLength
#         self.vector = {}
#
#     def buildFromTrees(self,left,right):
#         root = HTreeNode(left.val,right.val,left.val.weight+right.val.weight,None)
#         return root
#
#     def buildTree(self,nodeList):
#         if len(nodeList)<2:
#             self.root = nodeList.head
#             return
#
#         while(len(nodeList)>1):
#             left = nodeList.head
#             right = nodeList.head.next
#             nodeList.removeHead()
#             nodeList.removeHead()
#             tree = self.buildFromTrees(left,right)
#             node = Node()
#             node.val = tree
#             nodeList.insert(node)
#
#         self.root = nodeList.head.val
#
#     def coding(self,root,prefix,hierarchy):
#         if root:
#             root.code = prefix
#             self.vector[prefix] = np.random.random(self.vecLength)
#             if root.id:
#                 self.code[root.id] = prefix
#
#             # if root.id:
#             #     print 'level', hierarchy
#             #     print root.id,prefix,root.weight
#
#             self.coding(root.left,prefix+'0',hierarchy+1)
#             self.coding(root.right,prefix+'1',hierarchy+1)

class CUNE_BPR(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(CUNE_BPR, self).__init__(conf,trainingSet,testSet,fold)
        self.nonLeafVec = {}
        self.leafVec = {}

    def readConfiguration(self):
        super(CUNE_BPR, self).readConfiguration()
        options = config.OptionConf(self.config['CUNE-BPR'])
        self.walkCount = int(options['-T'])
        self.walkLength = int(options['-L'])
        self.walkDim = int(options['-l'])
        self.winSize = int(options['-w'])
        self.topK = int(options['-k'])
        self.s = float(options['-s'])
        self.epoch = int(options['-ep'])

    def printAlgorConfig(self):
        super(CUNE_BPR, self).printAlgorConfig()
        print('Specified Arguments of', self.config['model.name'] + ':')
        print('Walks count per user', self.walkCount)
        print('Length of each walk', self.walkLength)
        print('Dimension of user embedding', self.walkDim)
        print('='*80)

    def trainModel(self):
        print('Kind Note: This method will probably take much time.')
        #build C-U-NET
        print('Building collaborative user network...')
        #filter isolated nodes
        self.itemNet = {}
        for item in self.data.trainSet_i:
            if len(self.data.trainSet_i[item])>1:
                self.itemNet[item] = self.data.trainSet_i[item]

        self.filteredRatings = defaultdict(list)
        for item in self.itemNet:
            for user in self.itemNet[item]:
                if self.itemNet[item][user]>=1:
                    self.filteredRatings[user].append(item)

        self.CUNet = defaultdict(list)

        for user1 in self.filteredRatings:
            s1 = set(self.filteredRatings[user1])
            for user2 in self.filteredRatings:
                if user1 != user2:
                    s2 = set(self.filteredRatings[user2])
                    weight = len(s1.intersection(s2))
                    if weight > 0:
                        self.CUNet[user1]+=[user2]*weight


        #build Huffman Tree First
        #get weight
        # print 'Building Huffman tree...'
        # #To accelerate the method, the weight is estimated roughly
        # nodes = {}
        # for user in self.CUNet:
        #     nodes[user] = len(self.CUNet[user])
        # nodes = sorted(nodes.iteritems(),key=lambda d:d[1])
        # nodes = [HTreeNode(None,None,user[1],user[0]) for user in nodes]
        # nodeList = OrderedLinkList()
        # for node in nodes:
        #     listNode = Node()
        #     listNode.val = node
        #     try:
        #         nodeList.insert(listNode)
        #     except AttributeError:
        #         pass
        # self.HTree = HuffmanTree(vecLength=self.walkDim)
        # self.HTree.buildTree(nodeList)
        # print 'Coding for all users...'
        # self.HTree.coding(self.HTree.root,'',0)


        print('Generating random deep walks...')
        self.walks = []
        self.visited = defaultdict(dict)
        for user in self.CUNet:
            for t in range(self.walkCount):
                path = [user]
                lastNode = user
                for i in range(1,self.walkLength):
                    nextNode = choice(self.CUNet[lastNode])
                    count=0
                    while nextNode in self.visited[lastNode]:
                        nextNode = choice(self.CUNet[lastNode])
                        #break infinite loop
                        count+=1
                        if count==10:
                            break
                    path.append(nextNode)
                    self.visited[user][nextNode] = 1
                    lastNode = nextNode
                self.walks.append(path)
                #print path
        shuffle(self.walks)

        #Training get top-k friends
        print('Generating user embedding...')
        # epoch = 1
        # while epoch <= self.epoch:
        #     loss = 0
        #     #slide windows randomly
        #
        #     for n in range(self.walkLength/self.winSize):
        #
        #         for walk in self.walks:
        #             center = randint(0, len(walk)-1)
        #             s = max(0,center-self.winSize/2)
        #             e = min(center+self.winSize/2,len(walk)-1)
        #             for user in walk[s:e]:
        #                 centerUser = walk[center]
        #                 if user <> centerUser:
        #                     code = self.HTree.code[user]
        #                     centerCode = self.HTree.code[centerUser]
        #                     x = self.HTree.vector[centerCode]
        #                     for i in range(1,len(code)):
        #                         prefix = code[0:i]
        #                         w = self.HTree.vector[prefix]
        #                         self.HTree.vector[prefix] += self.lRate*(1-sigmoid(w.dot(x)))*x
        #                         self.HTree.vector[centerCode] += self.lRate*(1-sigmoid(w.dot(x)))*w
        #                         loss += -log(sigmoid(w.dot(x)),2)
        #     print 'epoch:', epoch, 'loss:', loss
        #     epoch+=1
        model = w2v.Word2Vec(self.walks, size=self.walkDim, window=5, min_count=0, iter=3)
        print('User embedding generated.')

        print('Constructing similarity matrix...')
        self.W = np.random.rand(self.data.trainingSize()[0], self.walkDim) / 10
        self.topKSim = {}
        i = 0
        for user1 in self.CUNet:
            # prefix1 = self.HTree.code[user1]
            # vec1 = self.HTree.vector[prefix1]
            sims = []
            u1 = self.data.user[user1]
            self.W[u1] = model.wv[user1]
            for user2 in self.CUNet:
                if user1 != user2:
                    u2 = self.data.user[user2]
                    self.W[u2] = model.wv[user2]
                    sims.append((user2,cosine(self.W[u1],self.W[u2])))
            self.topKSim[user1] = sorted(sims, key=lambda d: d[1], reverse=True)[:self.topK]
            i += 1
            if i % 200 == 0:
                print('progress:', i, '/', len(self.CUNet))
        print('Similarity matrix finished.')
        #prepare Pu set, IPu set, and Nu set
        print('Preparing item sets...')
        self.PositiveSet = defaultdict(dict)
        self.IPositiveSet = defaultdict(dict)

        for user in self.topKSim:
            for item in self.data.trainSet_u[user]:
                 self.PositiveSet[user][item]=1
            for friend in self.topKSim[user]:
                for item in self.data.trainSet_u[friend[0]]:
                    if item not in self.PositiveSet[user]:
                        self.IPositiveSet[user][item]=1


        print('Training...')
        epoch = 0
        while epoch < self.maxEpoch:
            self.loss = 0
            itemList = list(self.data.item.keys())
            for user in self.PositiveSet:
                u = self.data.user[user]
                kItems = list(self.IPositiveSet[user].keys())
                for item in self.PositiveSet[user]:
                    i = self.data.item[item]
                    for n in range(3): #negative sampling for 3 times
                        if len(self.IPositiveSet[user]) > 0:
                            item_k = choice(kItems)
                            k = self.data.item[item_k]
                            self.P[u] += self.lRate * (1 - sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[k]))) * (
                            self.Q[i] - self.Q[k])
                            self.Q[i] += self.lRate * (1 - sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[k]))) * \
                                         self.P[u]
                            self.Q[k] -= self.lRate * (1 - sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[k]))) * \
                                         self.P[u]
                            item_j = choice(itemList)
                            while item_j in self.PositiveSet[user] or item_j in self.IPositiveSet:
                                item_j = choice(itemList)
                            j = self.data.item[item_j]
                            self.P[u] += (1 / self.s) * self.lRate * (
                            1 - sigmoid((1 / self.s) * (self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[j])))) * (
                                         self.Q[k] - self.Q[j])
                            self.Q[k] += (1 / self.s) * self.lRate * (
                            1 - sigmoid((1 / self.s) * (self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[j])))) * self.P[u]
                            self.Q[j] -= (1 / self.s) * self.lRate * (
                            1 - sigmoid((1 / self.s) * (self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[j])))) * self.P[u]

                            self.P[u] -= self.lRate * self.regU * self.P[u]
                            self.Q[i] -= self.lRate * self.regI * self.Q[i]
                            self.Q[j] -= self.lRate * self.regI * self.Q[j]
                            self.Q[k] -= self.lRate * self.regI * self.Q[k]

                            self.loss += -log(sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[k]))) - \
                                         log(sigmoid((1 / self.s) * (self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[j]))))
                        else:
                            item_j = choice(itemList)
                            while item_j in self.PositiveSet[user]:
                                item_j = choice(itemList)
                            j = self.data.item[item_j]
                            self.P[u] += self.lRate * (1 - sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))) * (self.Q[i] - self.Q[j])
                            self.Q[i] += self.lRate * (1 - sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))) * self.P[u]
                            self.Q[j] -= self.lRate * (1 - sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))) * self.P[u]

                            self.loss += -log(sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j])))

                self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()
            epoch += 1
            if self.isConverged(epoch):
                break

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.Q.dot(self.P[u])
        else:
            return [self.data.globalMean] * self.num_items