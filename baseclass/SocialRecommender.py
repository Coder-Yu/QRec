from baseclass.IterativeRecommender import IterativeRecommender
from data.social import SocialDAO
from tool import config
from os.path import abspath
class SocialRecommender(IterativeRecommender):
    def __init__(self,conf,trainingSet,testSet,relation,fold='[1]'):
        super(SocialRecommender, self).__init__(conf,trainingSet,testSet,fold)
        self.social = SocialDAO(self.config,relation) #social relations access control

        # data clean
        cleanList = []
        cleanPair = []
        for user in self.social.followees:
            if not self.data.user.has_key(user):
                cleanList.append(user)
            for u2 in self.social.followees[user]:
                if not self.data.user.has_key(u2):
                    cleanPair.append((user, u2))
        for u in cleanList:
            del self.social.followees[u]

        for pair in cleanPair:
            if self.social.followees.has_key(pair[0]):
                del self.social.followees[pair[0]][pair[1]]

        cleanList = []
        cleanPair = []
        for user in self.social.followers:
            if not self.data.user.has_key(user):
                cleanList.append(user)
            for u2 in self.social.followers[user]:
                if not self.data.user.has_key(u2):
                    cleanPair.append((user, u2))
        for u in cleanList:
            del self.social.followers[u]

        for pair in cleanPair:
            if self.social.followers.has_key(pair[0]):
                del self.social.followers[pair[0]][pair[1]]

    def readConfiguration(self):
        super(SocialRecommender, self).readConfiguration()
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regS = float(regular['-s'])

    def printAlgorConfig(self):
        super(SocialRecommender, self).printAlgorConfig()
        print 'Social dataset:',abspath(self.config['social'])
        print 'Social size ','(User count:',len(self.social.user),'Trust statement count:'+str(len(self.social.relation))+')'
        print 'Social Regularization parameter: regS %.3f' % (self.regS)
        print '=' * 80

