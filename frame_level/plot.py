import numpy as np
from matplotlib import pyplot as plt
pre=np.load('frame_pre.npy')[100000:500000]
gt=np.load('frame_gt.npy')[100000:500000]
plt.plot(pre)
plt.plot(gt)
plt.show()