import numpy as np
from sklearn.utils import shuffle

# import matplotlib.pyplot as plt
# import colorsys
# import random
# def clamp(x): 
#     return max(0, min(x, 255))  

# def hsv_to_hex(h,s,v):
#     r,g,b = colorsys.hsv_to_rgb(h,s,v)
#     r = clamp(int(r*255))
#     g = clamp(int(g*255))
#     b = clamp(int(b*255))
#     return ("#{0:02x}{1:02x}{2:02x}".format(r,g,b))

  
# def shuffle(sample, vander_x):
#     max_val = len(sample[0])
#     for j in range(max_val):
#         new_pos = (random.randint(1, max_val)-1)
#         sample[0][j], sample[0][new_pos] = sample[0][new_pos], sample[0][j]
#         sample[1][j], sample[1][new_pos] = sample[1][new_pos], sample[1][j]
#         vander_x[[j, new_pos]] = vander_x[[new_pos, j]]
#     return sample, vander_x


# # def polynomial_regression(x, w):
# #     if((not isinstance(w, list)) and (not isinstance(w, np.ndarray))):
# #         return w
# #     return(np.vander(x,len(w), increasing=True)*w)

# # x = [1,2,3,4]
# # w = [5,6,7]
# # print(np.vander(x, len(w), increasing=True))
# # print(np.vander(x, len(w), increasing=True)@w)
# # A = np.array([[1,2,3,4], [5,6,7,8], [9,10, 11,12], [13, 14, 15 ,16]])
# # print("A:")
# # print(A)
# # print("A[0:2] (prime due righe?)")
# # print(A[0:2])
# # print("A[1:3] (righe in mezzo?)")
# # print(A[1:3])
# # print("A[2:4] (ultime due righe?)")
# # print(A[2:4])
# # print(hsv_to_hex(1,1,1))
# # print(hsv_to_hex(0.5,1,1))


# N = 4
D = 5
# def RandPoint():
#     x = np.random.rand()
#     return [x, np.sin(2*np.pi*x) + (np.random.uniform(low=-0.2, high=0.2))]

# sample = np.array([RandPoint() for i in np.arange(N)]).transpose()
sample = np.array([[1,2,3,4], [5, 6, 7,8]])
vander_x = np.vander(sample[0], D, increasing=True)
print("BEFORE:")
print(sample)
print(vander_x)
batch = 1
BATCH_SIZE = 2
print(vander_x[batch * BATCH_SIZE:batch * BATCH_SIZE + BATCH_SIZE])

# sample[0], sample[1], vander_x = shuffle(sample[0], sample[1], vander_x)
# print("\n\nAFTER:")
# print(sample)
# print(vander_x)

