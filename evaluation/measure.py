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
    def hits(origin,res):
        hitCount = {}
        for user in origin:
            items = origin[user].keys()
            predicted = [item[0] for item in res[user]]
            hitCount[user] = len(set(items).intersection(set(predicted)))
        return hitCount

    @staticmethod
    def hits_threshold(origin,res):
        hits = {}
        for user in origin:
            if len(origin[user]) > 0:
                items = origin[user].keys()
                predicted = [item[0] for item in res[user] if item[0]]
                hits[user] = len(set(items).intersection(set(predicted)))
        return hits


    @staticmethod
    def rankingMeasure(origin,res,N):
        measure = []
        if len(origin)!= len(res):
            print 'Lengths do not match!'
            exit(-1)
        hits = Measure.hits(origin,res)
        prec = Measure.precision(hits,N)
        measure.append('Precision:' + str(prec)+'\n')
        recall = Measure.recall(hits,origin)
        measure.append('Recall:' + str(recall)+'\n')
        F1 = Measure.F1(prec,recall)
        measure.append('F1:' + str(F1) + '\n')
        MAP = Measure.MAP(origin,res,N)
        measure.append('MAP:' + str(MAP) + '\n')
        #AUC = Measure.AUC(origin,res,rawRes)
        #measure.append('AUC:' + str(AUC) + '\n')
        return measure

    @staticmethod
    def rankingMeasure_threshold(origin,res,list_N):
        measure = []
        if len(origin) != len(res):
            print 'Lengths do not match!'
            exit(-1)
        hits = Measure.hits_threshold(origin,res)
        prec = Measure.precision_threshold(hits, list_N)
        measure.append('Precision:' + str(prec) + '\n')
        recall = Measure.recall(hits, origin)
        measure.append('Recall:' + str(recall) + '\n')
        F1 = Measure.F1(prec, recall)
        measure.append('F1:' + str(F1) + '\n')
        #MAP = Measure.MAP(origin, res)
        #measure.append('MAP:' + str(MAP) + '\n')
        return measure


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
            error += abs(entry[2] - entry[3])**2
            count += 1
        if count==0:
            return error
        return math.sqrt(float(error)/count)

    @staticmethod
    def precision(hits,N):
        prec = sum([hits[user] for user in hits])
        return float(prec)/(len(hits)*N)

    @staticmethod
    def precision_threshold(hits,list_N):
        sum  = 0
        denom = 0
        for user in hits:
            sum+=hits[user]
            denom+=list_N[user]
        if denom < 0:
            return 0
        return float(sum)/denom

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
    def AUC(origin,res,rawRes):

        from random import choice
        sum_AUC = 0
        for user in origin:
            count = 0
            larger = 0
            itemList = rawRes[user].keys()
            for item in origin[user]:
                item2 = choice(itemList)
                count+=1
                try:
                    if rawRes[user][item]>rawRes[user][item2]:
                        larger+=1
                except KeyError:
                    count-=1
            if count:
                sum_AUC+=float(larger)/count

        return float(sum_AUC)/len(origin)




    @staticmethod
    def recall(hits,origin):
        recallList = [float(hits[user])/len(origin[user]) for user in hits]
        recall = sum(recallList)/float(len(recallList))
        return recall

    @staticmethod
    def F1(prec,recall):
        if (prec+recall)!=0:
            return 2*prec*recall/(prec+recall)
        else:
            return 0



