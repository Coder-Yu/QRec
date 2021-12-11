from base.iterativeRecommender import IterativeRecommender
from util import config
from random import shuffle,choice
from collections import defaultdict
import numpy as np
from util.qmath import cosine
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

class CUNE_MF(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(CUNE_MF, self).__init__(conf,trainingSet,testSet,fold)
        self.nonLeafVec = {}
        self.leafVec = {}


    def readConfiguration(self):
        super(CUNE_MF, self).readConfiguration()
        options = config.OptionConf(self.config['CUNE-MF'])
        self.walkCount = int(options['-T'])
        self.walkLength = int(options['-L'])
        self.walkDim = int(options['-l'])
        self.winSize = int(options['-w'])
        self.topK = int(options['-k'])
        self.epoch = int(options['-ep'])
        self.alpha = float(options['-a'])

    def printAlgorConfig(self):
        super(CUNE_MF, self).printAlgorConfig()
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
        
        #print self.topKSim

        #matrix decomposition
        print('Decomposing...')

        epoch = 0
        while epoch < self.maxEpoch:
            self.loss = 0
            for entry in self.data.trainingData:
                user, item, rating = entry
                u = self.data.user[user] #get user id
                i = self.data.item[item] #get item id
                error = rating - self.P[u].dot(self.Q[i])
                self.loss += error**2
                p = self.P[u]
                q = self.Q[i]
                #update latent vectors
                self.P[u] += self.lRate*(error*q-self.regU*p)
                self.Q[i] += self.lRate*(error*p-self.regI*q)
            for user in self.CUNet:
                u = self.data.user[user]
                friends = self.topKSim[user]
                for friend in friends:
                    uf = self.data.user[friend[0]]
                    self.P[u] -= self.lRate*(self.P[u]-self.P[uf])*self.alpha
                    self.loss += self.alpha * (self.P[u]-self.P[uf]).dot(self.P[u]-self.P[uf])
            self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()
            epoch += 1
            if self.isConverged(epoch):
                break

