import matplotlib.pyplot as plt
import numpy as np
import time

plt.ion()
fig = plt.figure(113)
ax = plt.subplot(111)
plt.pause(0.001)

x = np.linspace(0, 10*np.pi, 100)
y = np.sin(x)

b = ax.bar(x,y)
# plt.show()
for phase in np.linspace(0, 10*np.pi, 100):
    for x,patch in enumerate(b.patches):
        patch.set_height(np.sin(0.5 * x + phase))
    plt.draw()
    plt.pause(0.1)
