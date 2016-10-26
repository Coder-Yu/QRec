test = []
with open('ratings.txt') as f:
    import random
    for line in f:
        if random.random()<0.01:
            test.append(line)

with open('testset.txt','w') as f:
    f.writelines(test)