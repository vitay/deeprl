
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


dt =  0.1
theta  =  0.15  # friction: strength to pull towards the mean 
sigma = 0.3 # noise
mu = 0.0 # global mean

processes = 3
samples = 1000

X = np.zeros(shape=(samples, processes))
for t in range(1, samples - 1):
    dw = norm.rvs(scale=dt, size=processes) # W: Wierner process, dw: brownian velocity
    dx = theta * (mu - X[t]) * dt + sigma * dw
    X[t+1] = X[t] + dx

plt.plot(X)
plt.title('Ornstein-Uhlenbeck Process')
plt.xlabel("Time")
plt.ylabel("x")
plt.show()
