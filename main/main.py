from algorithm.rating.UserKNN import UserKNN
from algorithm.rating.ItemKNN import ItemKNN
from tool.config import Config

if __name__ == '__main__':
    print '='*80
    print '   RecQ: An effective python-based recommender algorithm library.   '
    print '='*80

    print '1. UserKNN   2. ItemKNN'
    algor = 0
    order = input('please enter the num of the algorithm to run it:')
    if order == 1:
        conf = Config('../config/UserKNN')
        algor = UserKNN(conf)
    elif order == 2:
        conf = Config('../config/itemKNN')
        algor = UserKNN(conf)
    else:
        print 'Error num!'
        exit()
    algor.execute()
