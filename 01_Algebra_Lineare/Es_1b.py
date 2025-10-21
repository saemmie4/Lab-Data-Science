import numpy as np
import matplotlib.pyplot as plt

NUMBER_OF_EIGENVALUES = 20000
n = [[4, 10], [100, 100]]
N = [[5000, 2000], [200,20]]

AXES_LIM = [[[-2.5, 2.5], (-2,2)],[(-1.2, 1.2), (-1.2, 1.2)]]

figs, axs = plt.subplots(2,2)

for i in np.arange(2):
    for j in np.arange(2):
        axs[i,j].set_xlabel("Re(autovalore)")
        axs[i,j].set_ylabel("Im(autovalore)")
        axs[i,j].grid(True, ls="dotted")
        axs[i,j].set_title("Matrici %dx%d" %(n[i][j], n[i][j]))
        axs[i,j].set_aspect("equal")
        axs[i,j].set_xlim(AXES_LIM[i][j])
        axs[i,j].set_ylim(AXES_LIM[i][j])
        # eigenvals = np.empty(1,dtype=np.complex64)
        # print(eigenvals)
        eigappend = []
        print("Started n = %d" %(n[i][j]))
        for k in np.arange(N[i][j]):
            A = np.random.normal(loc=0, scale=1/np.sqrt(n[i][j]), size=(n[i][j],n[i][j]))
            # A = np.random.standard_normal((n[i][j],n[i][j]))
            eigappend.append(np.linalg.eigvals(A))
            if k == int(3*N[i][j]/4):
                print("\t 75% ")
            elif k == int(N[i][j]/2):
                print("\t 50% ")
            elif k == int(N[i][j]/4):
                print("\t 25% ")

        eigenvalues = np.array(eigappend).ravel()
        axs[i,j].plot (np.real(eigenvalues), np.imag(eigenvalues), ",k", markersize=2)
        print("\tResults of matrices: %dx%d" %(n[i][j], n[i][j]))
        print("\tVariance of the real part: %f" %(np.var(np.real(eigenvalues))))
        print("\tVariance of the imaginary part: %f" %(np.var(np.imag(eigenvalues))))
        print("Done %d" %n[i][j])



plt.show()
