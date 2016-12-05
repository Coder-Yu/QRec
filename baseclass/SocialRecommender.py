from baseclass.IterativeRecommender import IterativeRecommender
from data.social import SocialDAO
from tool import config
from os.path import abspath
class SocialRecommender(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,relation=list(),fold='[1]'):
        super(SocialRecommender, self).__init__(conf,trainingSet,testSet,fold)
        self.sao = SocialDAO(self.config,relation) #social relations access control

    def readConfiguration(self):
        super(SocialRecommender, self).readConfiguration()
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regS = float(regular['-s'])

    def printAlgorConfig(self):
        super(SocialRecommender, self).printAlgorConfig()
        print 'Social dataset:',abspath(self.config['social'])
        print 'Social Regularization parameter: regS %.3f' % (self.regS)
        print '=' * 80
