import numpy as np
import matplotlib.pyplot as plt
import colorsys


N_GRAPHS = 5
N = 200
D = 7 #il polinomio avrà grado D-1
eta = 0.01 #tasso di apprendimento
N_ITERATIONS = 5
print_percentage = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

#time at 200'000 iterations without precalculating vander (balanced): 19.51 s
#time at 200'000 iterations precalculating vander in gradientofcost (balanced): 15.84 s
#time at 200'000 iterations || + preallocating memory (balanced): 14.74 s
#time at 200'000 iterations || + precalculating vander in Cost (balanced): 10.28 s
#time at 200'000 iterations || + using PolynomialRegressionPreVander in GradientOfCostPreVanderTrans (balanced): 6.22 s
#time at 200'000 iterations || + calculating cost at the end (balanced): 5.34 s


#optimized functions

def GenerateHessian(x):
    vand = np.vander(x, 2*D-1, increasing=True)
    vector_to_be_shifted = np.sum(vand, axis = 0) #(sum(xi^0), sum(xi^1), sum(xi^2), ..., sum(xi^(2*(D-1))))
    hessian = np.zeros((D,D))
    for j in range(D):  
        hessian[j] = vector_to_be_shifted[j: j+D]
    return hessian/N

def CostPreVanderMatrix(vander_x, y, w_values):
#w_matrix must have the w values of the same iteration on the same row
    y_hat_matrix = w_values @ vander_x.transpose()
    diff = np.subtract(y_hat_matrix, y) #subtract the vector y. 
    return np.sum(diff*diff, axis = 1).transpose()/(2*N)


def GradientOfCostPreVander(vander_x, sample_y, w):
    y_hat = PolynomialRegressionPreVander(vander_x, w)
    return -(sample_y - y_hat)@vander_x/N

def CostPreVander(vander_x, y, w):
    diff = (y-PolynomialRegressionPreVander(vander_x, w))
    return np.dot(diff, diff)/(2*N)

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

  


sample = np.array([RandPoint() for i in np.arange(N)]).transpose()
w=np.array([(np.random.uniform(low=-0.5, high= 0.5)) for n in range(D)])

#for optimization
vander_x = np.vander(sample[0], len(w), increasing=True)
hessian = GenerateHessian(sample[0])
hessian_inverse = np.linalg.inv(hessian)
# print("\n\nHESSIANA:")
# print(hessian)
# print("\n\nINVERSA:")
# print(hessian_inverse)
#preallocate space
w_values = np.zeros((N_ITERATIONS+1, len(w)))
cost_values = np.zeros(N_ITERATIONS+1)

w_values[0] = w
cost_values[0] = CostPreVander(vander_x, sample[1], w)
print("Started the optimization...")
for i in range(N_ITERATIONS):
    w_values[i+1] = w_values[i] - hessian_inverse @ GradientOfCostPreVander(vander_x, sample[1], w_values[i])
    
    # # #optimized
    # if i < 0.6*N_ITERATIONS:
    #     w_values[i+1] = w_values[i] - 0.5 * GradientOfCostPreVander(vander_x, sample[1], w_values[i])
    # elif i < 0.7*N_ITERATIONS:
    #     w_values[i+1] = w_values[i] - 0.1 * GradientOfCostPreVander(vander_x, sample[1], w_values[i])
    # elif i < 0.9*N_ITERATIONS:
    #     w_values[i+1] = w_values[i] - 0.05 * GradientOfCostPreVander(vander_x, sample[1], w_values[i])
    # else:
    #     w_values[i+1] = w_values[i] - hessian_inverse @ GradientOfCostPreVander(vander_x, sample[1], w_values[i])

    # w -= eta * GradientOfCost(sample[0], sample[1], w)
    # cost_values[i+1] = Cost(sample[0], sample[1], w_values[i])
    # cost_values[i+1] = CostPreVander(vander_x, sample[1], w_values[i])
    for percentage in print_percentage:
        if i == (int)(percentage*N_ITERATIONS):
            print("\t{0}% done".format(100*percentage))
print("Finished optimization.")
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
ax[1].plot(cost_values)

plt.show()

