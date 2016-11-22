import sys
sys.path.append("..")
from algorithm.rating.UserKNN import UserKNN
from algorithm.rating.ItemKNN import ItemKNN
from algorithm.rating.BasicMF import BasicMF
from algorithm.rating.SlopeOne import SlopeOne
from algorithm.rating.RSTE import RSTE
from algorithm.rating.UserMean import UserMean
from algorithm.rating.ItemMean import ItemMean
from algorithm.rating.SVD import SVD
from tool.config import Config

if __name__ == '__main__':
    print '='*80
    print '   RecQ: An effective python-based recommender algorithm library.   '
    print '='*80

    print '1. UserKNN   2. ItemKNN   3. BasicMF   4. SlopeOne   5. RSTE   6. UserMean'
    print '7. ItemMean   8. SVD'
    algor = 0
    order = input('please enter the num of the algorithm to run it:')
    import time
    s = time.clock()
    if order == 1:
        conf = Config('../config/UserKNN.conf')
        algor = UserKNN(conf)
    elif order == 2:
        conf = Config('../config/ItemKNN.conf')
        algor = ItemKNN(conf)
    elif order == 3:
        conf = Config('../config/BasicMF.conf')
        algor = BasicMF(conf)
    elif order == 4:
        conf = Config('../config/SlopeOne.conf')
        algor = SlopeOne(conf)
    elif order == 5:
        conf = Config('../config/RSTE.conf')
        algor = RSTE(conf)
    elif order == 6:
        conf = Config('../config/UserMean.conf')
        algor = UserMean(conf)
    elif order == 7:
        conf = Config('../config/ItemMean.conf')
        algor = ItemMean(conf)
    elif order == 8:
        conf = Config('../config/SVD.conf')
        algor = SVD(conf)
    else:
        print 'Error num!'
        exit()
    algor.execute()
    e = time.clock()
    print "Run time: %f s" % (e - s)
