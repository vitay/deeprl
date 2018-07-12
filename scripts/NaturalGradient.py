import numpy as np
import matplotlib.pyplot as plt

# Parameters
A = 1.0 # 
B = 1.0 # 
sigma_system = 1.0
Q = 1.0
R = 1.0

def next_state(x, a):
    "linear system with Gaussian noise"
    return np.random.normal(A*x + B*a, sigma_system, 1)[0]

def select_action(x, theta1, theta2):
    "a linear policy with Gaussian exploration"
    return np.random.normal(theta1*x, theta2, 1)[0]

def reward(x, a):
    "quadratic reward."
    return -Q*x**2 - R*a**2

# Policy
theta1 = -1
theta2 = 1

# Initial state
x = -1

history_x = [x]
history_a = []
history_r = []

for t in range(100):
    # Select an action
    a = select_action(x, theta1, theta2)
    history_a.append(a)

    # Next state
    x = next_state(x, a)
    history_x.append(x)

    # Reward
    r = reward(x, a)
    history_r.append(r)

plt.subplot(131)
plt.plot(history_x)
plt.subplot(132)
plt.plot(history_a)
plt.subplot(133)
plt.plot(history_r)
plt.show()