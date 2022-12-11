import matplotlib.pyplot as plt
import numpy as np
from my_kf import KF


plt.ion()
plt.figure()

kf = KF(init_x = 0.0, init_v= 1.0,accel_var=0)

# constants
DT = 0.1
NUM_STEPS = 1000

data =[]
covs = []

for i in range(NUM_STEPS):
    covs.append(kf.cov)
    data.append(kf.mean)

    kf.predict(dt=DT)

plt.subplot(2,1,1)
plt.title("Position")
plt.plot([da[0] for da in data], 'r')
plt.plot([da[0] + 2*np.sqrt(cov[0,0]) for da, cov in zip(data,covs)], 'r--') # upper uncertainty bound 
plt.plot([da[0] - 2*np.sqrt(cov[0,0]) for da, cov in zip(data,covs)], 'r--') # lower uncertainty bound

plt.subplot(2,1,2)
plt.title("Velocity")
plt.plot([da[1] for da in data], 'b')
plt.plot([da[1] + 2*np.sqrt(cov[1,1]) for da, cov in zip(data,covs)], 'b--') # upper uncertainty bound 
plt.plot([da[1] - 2*np.sqrt(cov[1,1]) for da, cov in zip(data,covs)], 'b--') # lower uncertainty bound

plt.show()
plt.ginput(1)



