from baseclass.IterativeRecommender import IterativeRecommender
from data.social import SocialDAO
from tool import config
from os.path import abspath
class SocialRecommender(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(SocialRecommender, self).__init__(conf,trainingSet,testSet,fold)
        self.sao = SocialDAO(self.config,relation) #social relations access control

        # data clean
        cleanList = []
        cleanPair = []
        for user in self.sao.followees:
            if not self.dao.user.has_key(user):
                cleanList.append(user)
            for u2 in self.sao.followees[user]:
                if not self.dao.user.has_key(u2):
                    cleanPair.append((user, u2))
        for u in cleanList:
            del self.sao.followees[u]

        for pair in cleanPair:
            if self.sao.followees.has_key(pair[0]):
                del self.sao.followees[pair[0]][pair[1]]

        cleanList = []
        cleanPair = []
        for user in self.sao.followers:
            if not self.dao.user.has_key(user):
                cleanList.append(user)
            for u2 in self.sao.followers[user]:
                if not self.dao.user.has_key(u2):
                    cleanPair.append((user, u2))
        for u in cleanList:
            del self.sao.followers[u]

        for pair in cleanPair:
            if self.sao.followers.has_key(pair[0]):
                del self.sao.followers[pair[0]][pair[1]]

    def readConfiguration(self):
        super(SocialRecommender, self).readConfiguration()
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regS = float(regular['-s'])

    def printAlgorConfig(self):
        super(SocialRecommender, self).printAlgorConfig()
        print 'Social dataset:',abspath(self.config['social'])
        print 'Social size ','(User count:',len(self.sao.user),'Trust statement count:'+str(len(self.sao.relation))+')'
        print 'Social Regularization parameter: regS %.3f' % (self.regS)
        print '=' * 80

