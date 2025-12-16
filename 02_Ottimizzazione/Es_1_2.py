import numpy as np
import matplotlib.pyplot as plt

def polynomial_regression(x, w):
    if((not isinstance(w, list)) and (not isinstance(w, np.ndarray))):
        return w
    return(np.vander(x,len(w), increasing=True)@w)

D = [2, 3, 10, 101]

w=[np.array([(np.random.uniform(low=-0.5, high= 0.5)) for n in range(D[i])]) for i in range(len(D))]
x = np.linspace(0.,1, 1000)

# print(x)
# print(polynomial_regression(x, w[0].transpose()))
# print(polynomial_regression(x, w[0]))

figs, axs = plt.subplots(2,2)
for i in range(2):
    for j in range(2):
        axs[i,j].grid(True, ls="dotted")
axs[0,0].plot(x, polynomial_regression(x, w[0]) )
axs[0,1].plot(x, polynomial_regression(x, w[1]) )
axs[1,0].plot(x, polynomial_regression(x, w[2]) )
axs[1,1].plot(x, polynomial_regression(x, w[3]) )

plt.show()