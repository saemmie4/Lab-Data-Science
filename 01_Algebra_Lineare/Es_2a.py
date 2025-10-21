import numpy as np
import matplotlib.pyplot as plt

MAX_VALUE = 0.00001
N_POINTS = 1000
A = np.array([[0,1,0,0],[0,0,2,0],[0,0,0,3],[0,0,0,0]])

eps_points  = np.linspace(0,MAX_VALUE, N_POINTS)
eigenvals   = np.array([np.linalg.eigvals(A + np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[eps,0,0,0]])) for eps in eps_points])
sing_values = np.array([np.linalg.svd(A + np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[eps,0,0,0]]))[1] for eps in eps_points] )

fig, axs = plt.subplots(2, 4)
for j in np.arange(4):
    
    axs[0,j].set_xlabel("$\\epsilon$")
    axs[0,j].set_ylabel("{0}° autovalore ".format(j+1))
    axs[0,j].set_ylim(-0.1,0.1)
    axs[0,j].grid(True, ls="dotted")
    axs[0,j].plot(eps_points, np.real(eigenvals[:,j]), "r")
    axs[0,j].plot(eps_points, np.imag(eigenvals[:,j]), "b")

    axs[1,j].set_xlabel("$\\epsilon$")
    axs[1,j].set_ylabel("{0}° valore singolare".format(j+1))

    #the last behaves a little differently
    if j!= 3:
        axs[1,j].set_ylim(-0.5,4)
    else:
        axs[1,j].set_xticks(np.arange(0, 1.1e-5, step=0.25e-5))
        axs[1,j].set_yticks(np.arange(0, 1.1e-5, step=0.25e-5))


    axs[1,j].grid(True, ls="dotted")
    #i valori singolari sono sempre reali quindi basta una linea
    axs[1,j].plot(eps_points, np.real(sing_values[:,j]), "r" )
    axs[1,j].plot(eps_points, np.imag(sing_values[:,j]), "b" )


plt.show()