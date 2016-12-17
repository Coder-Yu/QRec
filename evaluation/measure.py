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
    def rankingMeasure(origin,res,N):
        measure = []
        if len(origin)!= len(res):
            print 'Lengths do not match!'
            exit(-1)
        hits={}
        for user in origin:
            hits[user] = 0
            items = [item[0] for item in origin[user]]
            predicted = [item[0] for item in res[user]]
            hits[user] += len(set(items).intersection(set(predicted)))
        prec = Measure.precision(hits,N)
        measure.append('Precision:' + str(prec)+'\n')
        recall = Measure.recall(hits,res)
        measure.append('Recall:' + str(recall)+'\n')
        F1 = Measure.F1(prec,recall)
        measure.append('F1:' + str(F1) + '\n')
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
    def recall(hits,origin):
        recall = sum([float(hits[user])/len(origin[user]) for user in hits])/len(hits)
        return recall

    @staticmethod
    def F1(prec,recall):
        if (prec+recall)!=0:
            return 2*prec*recall/(prec+recall)
        else:
            return 0



