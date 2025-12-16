import numpy as np
import matplotlib.pyplot as plt

def RandPoint():
    x = np.random.rand()
    return [x, np.sin(2*np.pi*x) + (np.random.uniform(low=-0.2, high=0.2))]

N = 200
random_points = np.array([RandPoint() for i in np.arange(N)]).transpose()
print (random_points)
real_curve = np.array([[x, np.sin(2*np.pi*x)] for x in np.linspace(0,1, 1000)]).transpose()
fig, ax = plt.subplots()

ax.set_ylabel("y", loc="top").set_fontsize("xx-large")
ax.set_xlabel("x", loc="right").set_fontsize("xx-large")
title = ax.set_title("Coppie di dati generati $(x_{i}, y_{i})$")
title.set_fontsize("xx-large")
ax.grid(True, ls="dotted")
ax.plot(random_points[0],random_points[1], ".b")
ax.plot (real_curve[0], real_curve[1], "r")
plt.show()