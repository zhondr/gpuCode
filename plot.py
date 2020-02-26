import numpy as np
import matplotlib.pyplot as plt

data_file = np.loadtxt('output.txt',delimiter=',')

matrixsize = data_file[:,0]
gpu = data_file[:,1]
gpuwithoverhead = data_file[:,2]
cpu = data_file[:,3]

plt.plot(matrixsize,cpu,matrixsize,gpu,matrixsize,gpuwithoverhead)
plt.legend(['CPU','GPU without copy overhead','GPU with overhead'],loc='best')
plt.xlabel('matrix size')
plt.ylabel('time')
plt.show()
