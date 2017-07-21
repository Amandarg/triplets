import triplet_network_randomized
import os
import numpy as np

#make array of parameters
parameter = []

step = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
item = [10, 20, 40, 80, 160]

for experiment in range(100):
    for i in item:
        for s in step:
            parameter.append([experiment, i, s])

data = np.array([float(os.getenv('PBS_ARRAYID'))])   # get m from PBS array

index = int(data[0])

triplet_network_randomized.find_embedding(int(parameter[index][0]),int(parameter[index][1]), parameter[index][2])

