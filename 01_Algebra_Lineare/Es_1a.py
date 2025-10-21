import numpy as np
import matplotlib.pyplot as plt
X_MIN = 0
X_MAX = 5
N = 1000
A = np.array([[-1,1],[-1,-1]])
print("Matrix A:")
print(A)
      
eig_values, eig_vectors = np.linalg.eig(A)
#eig_vectors è la matrice di autovettori messi in colonna
#eig_values è un vettore con gli autovalori

print("\nEigenvalues with corresponding Eisenvector:")
for i in np.arange(len(eig_values)):
    print("{0} --- {1}".format(eig_values[i], eig_vectors[:,i]) )
    
evolution = np.zeros((2,2, N),dtype=np.complex64)
eig_vectors_inv = np.linalg.inv(eig_vectors)
print("\nMatrix U (i.e. the eigenvectors matrix):")
print(eig_vectors)
print("\nMatrix U^(-1) (i.e. the invers of the eigenvectors matrix):")
print(eig_vectors_inv)
for i in np.arange(N):
    t = X_MIN + i*(X_MAX-X_MIN)/(N-1)

    D = np.diag(np.exp(t * eig_values))
    res = eig_vectors @ D @ eig_vectors_inv
    
    evolution[0,0,i] = res[0,0]
    evolution[0,1,i] = res[0,1]
    evolution[1,0,i] = res[1,0]
    evolution[1,1,i] = res[1,1]


fig, axs = plt.subplots(2,2)
fig.suptitle("Elementi di $e^{At}$ in funzione di t", size=18, y=0.96)
for i in np.arange(2):
    for j in np.arange(2):
        axs[i,j].set_xlabel("t")
        axs[i,j].set_ylabel(("elemento [%d, %d]" %(i, j)))
        axs[i,j].set_xlim(X_MIN,X_MAX)
        axs[i,j].set_ylim(-0.5,1)
        axs[i,j].grid(True, ls="dotted")

        axs[i,j].plot(np.linspace(X_MIN,X_MAX,N),np.real(evolution[i,j]), "r")  
        axs[i,j].plot(np.linspace(X_MIN,X_MAX,N),np.imag(evolution[i,j]), "b")  


plt.show()