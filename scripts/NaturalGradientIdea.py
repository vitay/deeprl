import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

x = np.linspace(-10, 10, 1000)

def gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2/(sigma**2))/np.sqrt(2*np.pi*sigma**2)

narrow1 = gaussian(x, 0, 0.2)
narrow2 = gaussian(x, 1, 0.2)

wide1 = gaussian(x, 0, 10)
wide2 = gaussian(x, 1, 10)

sns.set()
sns.set_style("ticks")
sns.set_context("talk")

plt.subplot(121)
plt.plot(x, narrow1)
plt.plot(x, narrow2)
plt.xlim((-1,2))
plt.ylim((0, np.max(narrow1)*1.1))
plt.yticks([])

plt.subplot(122)
plt.plot(x, wide1)
plt.plot(x, wide2)
plt.ylim((0,np.max(wide1)*1.1))
plt.yticks([])

sns.despine(top=True, left=True, right=True)
plt.show()
