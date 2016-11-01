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
        measure.append('RMSE:' + str(rmse))

        return measure


    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for key1 in res:
            for lst in res[key1]:
                error+=abs(lst[1]-lst[2])
                count+=1
        if count==0:
            return error
        return float(error)/count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for key1 in res:
            for lst in res[key1]:
                error+=(lst[1]-lst[2])**2
                count+=1
        if count==0:
            return error
        return math.sqrt(float(error)/count)

    # @staticmethod
    # def precision(hit,groundTruth,N):
    #     #* @param groundTruth: a collection of positive/correct item IDs
    #     return float(hit)/(userCount*N)

    # @staticmethod
    # def recall(hit,):
