import Modules.GRConnPy as GRC
from Utils import Local

import numpy as np
import matplotlib.pyplot as plt

rData, data_lengths = Local.ClusteredEEGLoader(event_number = 1)

subject = 10
inc_channels = [0, 3]

Data = np.mean(rData[subject][:int(data_lengths[subject]), inc_channels, :], axis = 0)

kernels = ['LRB_GC', 'PLI']
ordMat = 10

fig, axs = plt.subplots(1, 2, figsize = (10, 10), layout = 'constrained', sharex = True, sharey = True)

for i, kernel in enumerate(kernels):

    ConData = GRC.DynamicConnectivityMeasure(Data, kernel = kernel, orders_matrix = ordMat)

    time_p = np.linspace(-0.3, 1.1, ConData.shape[0])
    
    axs[i].plot(time_p, ConData[:, 0, 1], color = 'b')
    axs[i].plot(time_p, ConData[:, 1, 0], color = 'r')
    axs[i].set_title(kernel)
    axs[i].set_xlabel("time (s)")
    axs[i].set_ylabel("Connectivity Strength")

fig.suptitle("Connectivity Between two clusters")
fig.show()
plt.show()