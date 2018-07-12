import math
class Measure(object):
    def __init__(self):
        pass
    @staticmethod
    def ratingMeasure(res):
        measure = []
        mae = Measure.MAE(res)
        measure.append('MAE:'+str(mae)+'\n')
        rmse = Measure.RMSE(res)
        measure.append('RMSE:' + str(rmse)+'\n')

        return measure

    @staticmethod
    def hits(origin, res):
        hitCount = {}
        for user in origin:
            items = origin[user].keys()
            predicted = [item[0] for item in res[user]]
            hitCount[user] = len(set(items).intersection(set(predicted)))
        return hitCount

    @staticmethod
    def rankingMeasure(origin, res, N):
        measure = []
        for n in N:
            predicted = {}
            for user in res:
                predicted[user] = res[user][:n]
            indicators = []
            if len(origin) != len(predicted):
                print 'The Lengths of test set and predicted set are not match!'
                exit(-1)
            hits = Measure.hits(origin, predicted)
            prec = Measure.precision(hits, n)
            indicators.append('Precision:' + str(prec) + '\n')
            recall = Measure.recall(hits, origin)
            indicators.append('Recall:' + str(recall) + '\n')
            F1 = Measure.F1(prec, recall)
            indicators.append('F1:' + str(F1) + '\n')
            MAP = Measure.MAP(origin, predicted, n)
            indicators.append('MAP:' + str(MAP) + '\n')
            NDCG = Measure.NDCG(origin, predicted, n)
            indicators.append('NDCG:' + str(NDCG) + '\n')
            # AUC = Measure.AUC(origin,res,rawRes)
            # measure.append('AUC:' + str(AUC) + '\n')
            measure.append('Top ' + str(n) + '\n')
            measure += indicators
        return measure

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return float(prec) / (len(hits) * N)

    @staticmethod
    def MAP(origin, res, N):
        sum_prec = 0
        for user in res:
            hits = 0
            precision = 0
            for n, item in enumerate(res[user]):
                if origin[user].has_key(item[0]):
                    hits += 1
                    precision += hits / (n + 1.0)
            sum_prec += precision / (min(len(origin[user]), N) + 0.0)
        return sum_prec / (len(res))

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if origin[user].has_key(item[0]):
                    DCG+= 1.0/math.log(n+2)
            for n, item in enumerate(origin[user].keys()[:N]):
                IDCG+=1.0/math.log(n+2)
            sum_NDCG += DCG / IDCG
        return sum_NDCG / (len(res))
    # @staticmethod
    # def AUC(origin, res, rawRes):
    #
    #     from random import choice
    #     sum_AUC = 0
    #     for user in origin:
    #         count = 0
    #         larger = 0
    #         itemList = rawRes[user].keys()
    #         for item in origin[user]:
    #             item2 = choice(itemList)
    #             count += 1
    #             try:
    #                 if rawRes[user][item] > rawRes[user][item2]:
    #                     larger += 1
    #             except KeyError:
    #                 count -= 1
    #         if count:
    #             sum_AUC += float(larger) / count
    #
    #     return float(sum_AUC) / len(origin)

    @staticmethod
    def recall(hits, origin):
        recallList = [float(hits[user]) / len(origin[user]) for user in hits]
        recall = sum(recallList) / float(len(recallList))
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return 2 * prec * recall / (prec + recall)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error+=abs(entry[2]-entry[3])
            count+=1
        if count==0:
            return error
        return float(error)/count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3])**2
            count += 1
        if count==0:
            return error
        return math.sqrt(float(error)/count)




