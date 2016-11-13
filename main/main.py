from algorithm.rating.UserKNN import UserKNN
from algorithm.rating.ItemKNN import ItemKNN
from algorithm.rating.BasicMF import BasicMF
from algorithm.rating.SlopeOne import SlopeOne
from tool.config import Config

if __name__ == '__main__':
    print '='*80
    print '   RecQ: An effective python-based recommender algorithm library.   '
    print '='*80

    print '1. UserKNN   2. ItemKNN   3. BasicMF   4. SlopeOne'
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
    else:
        print 'Error num!'
        exit()
    algor.execute()
    e = time.clock()
    print "Run time: %f s" % (e - s)
