class Measure(object):
    def __init__(self):
        pass

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

