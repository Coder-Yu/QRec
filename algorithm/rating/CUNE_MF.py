
from baseclass.IterativeRecommender import IterativeRecommender

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
    def __init__(self,root=None):
        self.root = root
        self.weight = 0
        self.code = {}

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


    def traverse(self,root,prefix,hierarchy):
        if root:
            root.code=prefix
            if root.id:
                self.code[root.id] = prefix

            if root.id:
                print 'level', hierarchy
                print root.id,prefix,root.weight

            self.traverse(root.left,prefix+'0',hierarchy+1)
            self.traverse(root.right,prefix+'1',hierarchy+1)




class CUNE_MF(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(CUNE_MF, self).__init__(conf,trainingSet,testSet,fold)
        self.nonLeafVec = {}
        self.leafVec = {}
        self.topKSim = {}

    def buildModel(self):
        #build Huffman Tree First
        #get weight
        print 'Building Huffman tree...'
        #To accelerate the method, the weight is estimated roughly
        nodes = {}
        for user in self.dao.trainSet_u:
            nodes[user] = len(self.dao.trainSet_u[user])
        nodes = sorted(nodes.iteritems(),key=lambda d:d[1])
        nodes = [HTreeNode(None,None,user[1],user[0]) for user in nodes]
        nodeList = OrderedLinkList()
        for node in nodes:
            listNode = Node()
            listNode.val = node
            try:
                nodeList.insert(listNode)
            except AttributeError:
                pass
        self.HTree = HuffmanTree()
        self.HTree.buildTree(nodeList)
        print 'Coding for all users...'
        self.HTree.traverse(self.HTree.root,'',0)


        print 'Generating random deep walks...'


        # iteration = 0
        # while iteration < self.maxIter:
        #     self.loss = 0
        #     for entry in self.dao.trainingData:
        #         user, item, rating = entry
        #         u = self.dao.user[user] #get user id
        #         i = self.dao.item[item] #get item id
        #         error = rating - self.P[u].dot(self.Q[i])
        #         self.loss += error**2
        #         p = self.P[u]
        #         q = self.Q[i]
        #
        #         #update latent vectors
        #         self.P[u] += self.lRate*(error*q-self.regU*p)
        #         self.Q[i] += self.lRate*(error*p-self.regI*q)
        #
        #     self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()
        #     iteration += 1
        #     if self.isConverged(iteration):
        #         break

