from baseclass.SocialRecommender import SocialRecommender
from tool import config
from random import randint
from random import shuffle,choice
from collections import defaultdict
import numpy as np
from tool.qmath import sigmoid,cosine
from math import log
from structure.symmetricMatrix import SymmetricMatrix
class Node(object):
    def __init__(self):
        self.val = 0
        self.last = None
        self.next = None

class OrderedLinkList(object):
    def __init__(self):
        self.head=None
        self.tail=None
        self.length = 0

    def __len__(self):
        return self.length

    def insert(self,node):
        self.length+=1
        if self.head:
            tmp = self.head
            while tmp.val < node.val:
                if tmp==self.tail:
                    break
                tmp = tmp.next

            if tmp is self.head:

                if self.head.val < node.val:
                    node.next = self.head
                    self.head.last = node
                    self.head = node
                else:
                    node.next = self.head
                    self.head.last = node
                    self.head = node
                return

            node.next = tmp.next
            tmp.next = node
            node.last = tmp
            if not node.next:
                self.tail = node

        else:
            self.head = node
            self.tail = node

    def removeHead(self):
        if self.head:
            self.head = self.head.next
            self.length -= 1

    def removeNode(self,node):
        if self.head:
            tmp = self.head
            while tmp is not node and tmp.next:
                tmp = tmp.next
            if tmp.next:
                tmp.last.next = tmp.next
                tmp.next.last = tmp.last
            self.length-=1


class HTreeNode(object):
    def __init__(self,left,right,freq,id,code=None):
        self.left = left
        self.right = right
        self.weight = freq
        self.id = id
        self.code = code

    def __lt__(self, other):
        if self.weight < other.weight:
            return True
        else:
            return False

class HuffmanTree(object):
    def __init__(self,root=None,vecLength=10):
        self.root = root
        self.weight = 0
        self.code = {}
        self.vecLength = vecLength
        self.vector = {}

    def buildFromTrees(self,left,right):
        root = HTreeNode(left.val,right.val,left.val.weight+right.val.weight,None)
        return root

    def buildTree(self,nodeList):
        if len(nodeList)<2:
            self.root = nodeList.head
            return

        while(len(nodeList)>1):
            left = nodeList.head
            right = nodeList.head.next
            nodeList.removeHead()
            nodeList.removeHead()
            tree = self.buildFromTrees(left,right)
            node = Node()
            node.val = tree
            nodeList.insert(node)

        self.root = nodeList.head.val

    def coding(self,root,prefix,hierarchy):
        if root:
            root.code = prefix
            self.vector[prefix] = np.random.random(self.vecLength)
            if root.id:
                self.code[root.id] = prefix

            # if root.id:
            #     print 'level', hierarchy
            #     print root.id,prefix,root.weight

            self.coding(root.left,prefix+'0',hierarchy+1)
            self.coding(root.right,prefix+'1',hierarchy+1)

class MPE_MF(SocialRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation= None,fold='[1]'):
        super(MPE_MF, self).__init__(conf,trainingSet,testSet,relation,fold)
        self.nonLeafVec = {}
        self.leafVec = {}


    def readConfiguration(self):
        super(MPE_MF, self).readConfiguration()
        options = config.LineConfig(self.config['CUNE-MF'])
        self.walkCount = int(options['-T'])
        self.walkLength = int(options['-L'])
        self.walkDim = int(options['-l'])
        self.winSize = int(options['-w'])
        self.topK = int(options['-k'])
        self.alpha = float(options['-a'])
        self.epoch = int(options['-ep'])

    def printAlgorConfig(self):
        super(MPE_MF, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['recommender'] + ':'
        print 'Walks count per user', self.walkCount
        print 'Length of each walk', self.walkLength
        print 'Dimension of user embedding', self.walkDim
        print '='*80

    def buildModel(self):
        print 'Kind Note: This method will probably take much time.'
        #build U-F-NET
        print 'Building weighted user-friend network...'
        #filter isolated nodes and low ratings
        #Definition of Meta-Path
        p1 = 'UIU'
        p2 = 'UFU'
        p3 = 'UFIU'
        #p4 = ''
        mPaths = [p1,p2,p3]

        self.G = np.random.rand(self.dao.trainingSize()[1], self.k)/10

        self.fItems = {} #filtered item set
        for item in self.dao.trainSet_i:
            if len(self.dao.trainSet_i[item])>1:
                self.fItems[item] = self.dao.trainSet_i[item]

        self.fBuying = defaultdict(list) #filtered buying set
        for user in self.dao.trainSet_u:
            for item in self.dao.trainSet_u[user]:
                if self.fItems.has_key(item):
                    self.fBuying[user].append(item)
        # self.filteredRatings = defaultdict(list)
        # for item in self.fItems:
        #     for user in self.fItems[item]:
        #         if self.fItems[item][user]>0.75:
        #             self.filteredRatings[user].append(item)

        self.UFNet = defaultdict(list)

        for user1 in self.sao.user:
            s1 = set(self.sao.followees[user1])
            for user2 in self.sao.followees[user1]:
                if user1 <> user2:                    
                    s2 = set(self.sao.followees[user2])
                    weight = len(s1.intersection(s2))
                    self.UFNet[user1]+=[user2]*(weight+1)
                    


        #build Huffman Tree First
        #get weight
        # print 'Building Huffman tree...'
        # #To accelerate the method, the weight is estimated roughly
        # nodes = {}
        # for user in self.UFNet:
        #     nodes[user] = len(self.UFNet[user])
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


        print 'Generating random meta-path random walks...'
        self.walks = []
        self.visited = defaultdict(dict)
        for user in self.dao.user:
            for t in range(self.walkCount):
                path = [user]
                lastNode = user
                for mp in mPaths:
                    for i in range(1,self.walkLength/len(mp)):
                        for tp in mp[1:]:
                            if tp == 'I':
                                nextNode = choice(self.fBuying[lastNode])
                                path.append(nextNode)


                        nextNode = self.UFNet[user][randint(0,len(self.UFNet[user]))-1]
                        count=0
                        while(self.visited[user].has_key(nextNode)):
                            nextNode = self.UFNet[randint(0, len(self.UFNet[user]))-1]
                            #break infinite loop
                            count+=1
                            if count==10:
                                break
                        path.append(nextNode)
                    self.walks.append(path)
                    #print path
        shuffle(self.walks)

        #Training get top-k friends
        print 'Generating user embedding...'
        iteration = 1
        while iteration <= self.maxIter:
            loss = 0
            for walk in self.walks:
                for user in walk:
                    centerUser = walk[len(walk)/2]
                    if user <> centerUser:
                        code = self.HTree.code[user]
                        centerCode = self.HTree.code[centerUser]
                        x = self.HTree.vector[centerCode]
                        for i in range(1,len(code)):
                            prefix = code[0:i]
                            w = self.HTree.vector[prefix]
                            self.HTree.vector[prefix] += self.lRate*(1-sigmoid(w.dot(x)))*x
                            self.HTree.vector[centerCode] += self.lRate*(1-sigmoid(w.dot(x)))*w
                            loss += -log(sigmoid(w.dot(x)))
            print 'iteration:', iteration, 'loss:', loss
            iteration+=1
        print 'User embedding generated.'

        # print 'Constructing similarity matrix...'
        # self.Sim = SymmetricMatrix(len(self.UFNet))
        # for user1 in self.UFNet:
        #     for user2 in self.UFNet:
        #         if user1 <> user2:
        #             prefix1 = self.HTree.code[user1]
        #             vec1 = self.HTree.vector[prefix1]
        #             prefix2 = self.HTree.code[user2]
        #             vec2 = self.HTree.vector[prefix2]
        #             if self.Sim.contains(user1, user2):
        #                 continue
        #             sim = cosine(vec1,vec2)
        #             self.Sim.set(user1, user2, sim)
        # self.topKSim = {}
        # for user in self.UFNet:
        #     self.topKSim[user]= sorted(self.Sim[user].iteritems(),key=lambda d:d[1],reverse=True)[:self.topK]
        # print 'Similarity matrix finished.'
        #print self.topKSim

        #matrix decomposition
        print 'Decomposing...'

        iteration = 0
        while iteration < self.epoch:
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

            for user in self.UFNet:

                u = self.dao.user[user]
                friends = self.topKSim[user]
                for friend in friends:
                    uf = self.dao.user[friend[0]]
                    self.P[u] -= self.lRate*(self.P[u]-self.P[uf])*self.alpha
                    self.loss += self.alpha * (self.P[u]-self.P[uf]).dot(self.P[u]-self.P[uf])

            self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break

