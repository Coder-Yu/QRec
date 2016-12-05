import sys
sys.path.append("..")
from RecQ import RecQ
from tool.config import Config
from visual.display import Display
from algorithm.rating.TrustMF import TrustMF


if __name__ == '__main__':

    print '='*80
    print '   RecQ: An effective python-based recommender algorithm library.   '
    print '='*80
    print '0. Analyze the input data.(Configure the visual.conf in config/visual first.)'
    print '1. UserKNN   2. ItemKNN   3. BasicMF   4. SlopeOne   5. RSTE   6. UserMean'
    print '7. ItemMean   8. SVD   9. PMF   10. TrustMF   11. SocialMF'
    algor = -1
    conf = -1
    print '-'*80
    order = input('please enter the num of the algorithm to run it:')
    import time
    s = time.clock()
    if order == 0:
        try:
            import seaborn as sns
        except ImportError:
            print '!!!To obtain nice data charts, ' \
                  'we strongly recommend you to install the third-party package <seaborn>!!!'
        conf = Config('../config/visual/visual.conf')
        Display(conf).render()
        exit(0)
    elif order == 1:
        conf = Config('../config/UserKNN.conf')

    elif order == 2:
        conf = Config('../config/ItemKNN.conf')

    elif order == 3:
        conf = Config('../config/BasicMF.conf')

    elif order == 4:
        conf = Config('../config/SlopeOne.conf')

    elif order == 5:
        conf = Config('../config/RSTE.conf')

    elif order == 6:
        conf = Config('../config/UserMean.conf')

    elif order == 7:
        conf = Config('../config/ItemMean.conf')

    elif order == 8:
        conf = Config('../config/SVD.conf')

    elif order == 9:
        conf = Config('../config/PMF.conf')

    elif order == 10:
        conf = Config('../config/TrustMF.conf')

    elif order == 11:
        conf = Config('../config/SocialMF.conf')


    else:
        print 'Error num!'
        exit(-1)
    RecQ = RecQ(conf)
    RecQ.execute()
    e = time.clock()
    print "Run time: %f s" % (e - s)
