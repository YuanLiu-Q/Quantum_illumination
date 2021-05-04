import numpy as np
import random

if __name__ =='__main__':
    # random seed
    random.seed(123)
    np.random.seed(123)
    n = 2000
    m = 50
    da_train = np.zeros([n, 2])
    for i in range(n):
        da_train[i][1] = random.uniform(0.5, 1) #env
        da_train[i][0] = random.uniform(0, 1) #gamma
    np.savetxt('train.txt', da_train)
    da_test = np.zeros([m+1, 2])
    for i in range(m+1):
        da_test[i][1] = 0.5
        da_test[i][0] = i/m
    np.savetxt('test.txt', da_test)
