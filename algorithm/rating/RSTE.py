from baseclass.SocialRecommender import SocialRecommender
from tool import config
class RSTE(SocialRecommender):
    def __init__(self,conf):
        super(RSTE, self).__init__(conf)

    def readConfiguration(self):
        super(SocialRecommender, self).readConfiguration()
        alpha = config.LineConfig(self.config['RSTE'])
        self.alpha = float(alpha['-alpha'])

    def printAlgorConfig(self):
        super(RSTE, self).printAlgorConfig()
        print 'Specified Arguments of',self.config['recommender']+':'
        print 'alpha: .3f' %self.alpha
        print '='*80

    def initModel(self):
        super(RSTE, self).initModel()

    def buildModel(self):
        pass