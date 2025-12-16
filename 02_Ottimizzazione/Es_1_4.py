import numpy as np
import matplotlib.pyplot as plt
import colorsys
from sklearn.utils import shuffle
import sys
import time
D = 6
print_percentage = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

N_EPOCHS = 50000
N = 200
BATCH_SIZE = 5
N_BATCHES = int(N/BATCH_SIZE) #must be an integer
if N%BATCH_SIZE != 0:
    print("[ERRORE]: N NON DIVISIBILE PER LA BATCH_SIZE!!!!!!!")
    sys.exit()
eta = 0.1 #tasso di apprendimento

N_GRAPHS = 5
# print_percentage = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

#optimized functions


def CostPreVanderMatrix(vander_x, y, w_values):
#w_matrix must have the w values of the same iteration on the same row
    y_hat_matrix = w_values @ vander_x.transpose()
    diff = np.subtract(y_hat_matrix, y) #subtract the vector y. 
    return np.sum(diff*diff, axis = 1).transpose()/(2*len(y))


def GradientOfCostPreVander(vander_x, sample_y, w):
    y_hat = PolynomialRegressionPreVander(vander_x, w)
    return -(sample_y - y_hat)@vander_x/len(sample_y)

def CostPreVander(vander_x, y, w):
    diff = (y-PolynomialRegressionPreVander(vander_x, w))
    return np.dot(diff, diff)/(2*len(y))

def PolynomialRegressionPreVander(vander_x, w):
    return(vander_x @ w.transpose())


# def GradientOfCost(sample_x, sample_y, w):
#     y_hat = polynomial_regression(sample_x, w)
#     return -np.vander(sample_x, len(w), increasing=True).transpose()@(sample_y - y_hat)/N

def polynomial_regression(x, w):
    if((not isinstance(w, list)) and (not isinstance(w, np.ndarray))):
        return w
    return(np.vander(x,len(w), increasing=True)@w)

def RandPoint():
    x = np.random.rand()
    return [x, np.sin(2*np.pi*x) + (np.random.uniform(low=-0.2, high=0.2))]

# def Cost(x, y, w):
#     return np.sum((y-polynomial_regression(x, w))**2)/(2*len(x))


def clamp(x): 
    return max(0, min(x, 255))

def hsv_to_hex(h,s,v):
    r,g,b = colorsys.hsv_to_rgb(h,s,v)
    r = clamp(int(r*255))
    g = clamp(int(g*255))
    b = clamp(int(b*255))
    return ("#{0:02x}{1:02x}{2:02x}".format(r,g,b))

# def shuffle(sample, vander_x):
#     max_val = len(sample[0])
#     for j in range(max_val):
#         new_pos = (random.randint(1, max_val)-1)
#         sample[0][j], sample[0][new_pos] = sample[0][new_pos], sample[0][j]
#         sample[1][j], sample[1][new_pos] = sample[1][new_pos], sample[1][j]
#         vander_x[[j, new_pos]] = vander_x[[new_pos, j]]
#     return sample, vander_x


sample = np.array([RandPoint() for i in np.arange(N)]).transpose()
w=np.array([(np.random.uniform(low=-0.5, high= 0.5)) for n in range(D)])

#for optimization
vander_x = np.vander(sample[0], len(w), increasing=True)

#preallocate space
w_values = np.zeros((N_EPOCHS*N_BATCHES+1, len(w)))
cost_values = np.zeros(N_EPOCHS*N_BATCHES+1)

w_values[0] = w
cost_values[0] = CostPreVander(vander_x, sample[1], w)
print("Started the optimization...")
start_time = time.time()
for epoch in range(N_EPOCHS):
    for batch in range(N_BATCHES):
        i = epoch * N_BATCHES + batch
        w_values[i+1] = w_values[i] - eta * GradientOfCostPreVander(vander_x[batch * BATCH_SIZE:batch * BATCH_SIZE + BATCH_SIZE], sample[1][batch * BATCH_SIZE:batch * BATCH_SIZE + BATCH_SIZE], w_values[i])
    sample[0], sample[1], vander_x = shuffle(sample[0], sample[1], vander_x)
    for percentage in print_percentage:
        if epoch == (int)(percentage*N_EPOCHS):
            print("\t{0}% done [{1}/{2}]".format(100*percentage, epoch, N_EPOCHS))
    # print("\tDone epoch: {0}/{1}".format((epoch), N_EPOCHS))
end_time = time.time()
print("Finished optimization.")
print("Time Taken: {0} s".format(end_time-start_time))
print("Calulating the cost values...")
cost_values = CostPreVanderMatrix(vander_x, sample[1], w_values)
print("Done")
x_draw = np.linspace(0.,1., 1000)
fig, ax = plt.subplots(2)

ax[0].set_title("")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[1].set_xlabel("indice di iterazione")
ax[1].set_ylabel("")

ax[0].grid(True, ls="dotted")
ax[1].grid(True, ls="dotted")
ax[0].plot(sample[0], sample[1], ".b")

for i in range(N_GRAPHS):
    hue = i/(N_GRAPHS - 1) * (1./3.) 
    iteration = min(len(w_values) - 1, int(i * len(w_values)/N_GRAPHS))
    ax[0].plot(x_draw, polynomial_regression(x_draw, w_values[iteration]),hsv_to_hex(hue, 1,1))
ax[1].plot((cost_values))

plt.show()

