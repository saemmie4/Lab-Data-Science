import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('image.jpg')

CUTOFF_R = 50
CUTOFF_G = 50
CUTOFF_B = 50
# CUTOFF_A = 2

img_r = image[..., 0]  
img_g = image[..., 1] 
img_b = image[..., 2] 

img_r_SVD = np.linalg.svd(img_r, full_matrices=True)
img_g_SVD = np.linalg.svd(img_g, full_matrices=True)
img_b_SVD = np.linalg.svd(img_b, full_matrices=True)
# img_a_SVD = np.linalg.svd(img_a, full_matrices=True)

figs_sing_val, axs_sing_val = plt.subplots(1,3)
# axs_sing_val[0].set_yscale('log')
# axs_sing_val[1].set_yscale('log')
# axs_sing_val[2].set_yscale('log')
# axs_sing_val[1,1].set_yscale('log')

for j in np.arange(3):
    axs_sing_val[j].grid(True, ls="dotted")
    axs_sing_val[j].set_ylim(0.,1.05)
    axs_sing_val[j].set_yticks(np.arange(0, 1.01, step=0.1))
    axs_sing_val[j].set_ylabel("somma cumulativa normalizzata")
    axs_sing_val[j].set_xlabel("indice valori singolari")
axs_sing_val[0].set_title("Rosso")
axs_sing_val[1].set_title("Verde")
axs_sing_val[2].set_title("Blu")

axs_sing_val[0].fill_between(np.arange(0,len(img_r_SVD[1])), 0., np.cumsum(img_r_SVD[1])/np.sum(img_r_SVD[1]),color='r')
axs_sing_val[1].fill_between(np.arange(0,len(img_g_SVD[1])), 0., np.cumsum(img_g_SVD[1])/np.sum(img_g_SVD[1]),color='g')
axs_sing_val[2].fill_between(np.arange(0,len(img_b_SVD[1])), 0., np.cumsum(img_b_SVD[1])/np.sum(img_b_SVD[1]),color='b')
# axs_sing_val[1,1].plot(np.cumsum(img_a_SVD[1])/np.sum(img_a_SVD[1]))

img_r_SVD_small = [img_r_SVD[0][:,:CUTOFF_R], img_r_SVD[1][:CUTOFF_R], img_r_SVD[2][:CUTOFF_R,:]]
img_g_SVD_small = [img_g_SVD[0][:,:CUTOFF_G], img_g_SVD[1][:CUTOFF_G], img_g_SVD[2][:CUTOFF_G,:]]
img_b_SVD_small = [img_b_SVD[0][:,:CUTOFF_B], img_b_SVD[1][:CUTOFF_B], img_b_SVD[2][:CUTOFF_B,:]]
# img_a_SVD_small = [img_a_SVD[0][:,:CUTOFF_A], img_a_SVD[1][:CUTOFF_A], img_a_SVD[2][:CUTOFF_A,:]]


img_r_small = np.zeros(image.shape)
img_g_small = np.zeros(image.shape)
img_b_small = np.zeros(image.shape)
# img_a_small = np.zeros(image.shape)

img_r_small [:,:,0] = np.clip(img_r_SVD_small[0] @ np.diag(img_r_SVD_small[1]) @ img_r_SVD_small[2], a_min=0., a_max=255.)/255.
img_g_small [:,:,1] = np.clip(img_g_SVD_small[0] @ np.diag(img_g_SVD_small[1]) @ img_g_SVD_small[2], a_min=0., a_max=255.)/255.
img_b_small [:,:,2] = np.clip(img_b_SVD_small[0] @ np.diag(img_b_SVD_small[1]) @ img_b_SVD_small[2], a_min=0., a_max=255.)/255.
# img_a_small [:,:,3] = img_a_SVD_small[0] @ np.diag(img_a_SVD_small[1]) @ img_a_SVD_small[2]
img_small = img_r_small + img_g_small + img_b_small #+ img_a_small


img_r_diff = np.zeros(image.shape)
img_g_diff = np.zeros(image.shape)
img_b_diff = np.zeros(image.shape)

img_r_diff [:,:,0] = img_r/255
img_g_diff [:,:,1] = img_g/255
img_b_diff [:,:,2] = img_b/255

img_r_diff -= img_r_small 
img_g_diff -= img_g_small 
img_b_diff -= img_b_small 

img_r_diff = abs(img_r_diff)
img_g_diff = abs(img_g_diff)
img_b_diff = abs(img_b_diff)

img_r_diff/=np.max(img_r_diff[:,:,0]) #maximize contrast
img_g_diff/=np.max(img_g_diff[:,:,1]) #maximize contrast
img_b_diff/=np.max(img_b_diff[:,:,2]) #maximize contrast

img_diff = img_r_diff + img_g_diff + img_b_diff

# img_r_small [:,:,3] = np.ones((image.shape[0],image.shape[1])) 
# img_g_small [:,:,3] = np.ones((image.shape[0],image.shape[1])) 
# img_b_small [:,:,3] = np.ones((image.shape[0],image.shape[1])) 

figs, axs = plt.subplots(2,4)

# axs[0,0].imshow(img_r_small)
# axs[0,1].imshow(img_g_small)
# axs[1,0].imshow(img_b_small)
# axs[1,1].imshow(img_small)

for i in range(2):
    for j in range(4):
        axs[i,j].get_yaxis().set_visible(False)
        axs[i,j].get_xaxis().set_visible(False)


axs[0,0].set_title("Rosso")
axs[0,0].imshow(img_r_small)

axs[0,1].set_title("Rosso diff")
axs[0,1].imshow(img_r_diff)

axs[0,2].set_title("Verde")
axs[0,2].imshow(img_g_small)

axs[0,3].set_title("Verde diff")
axs[0,3].imshow(img_g_diff)

axs[1,0].set_title("Blu")
axs[1,0].imshow(img_b_small)

axs[1,1].set_title("Blu diff")
axs[1,1].imshow(img_b_diff)

axs[1,2].set_title("Immagine")
axs[1,2].imshow(img_small)

axs[1,3].set_title("Immagine diff")
axs[1,3].imshow(img_diff)

N_vals_i = image.shape[0]*image.shape[1]*image.shape[2]
N_vals_r_f = CUTOFF_R*(image.shape[0] + image.shape[1] + 1)
N_vals_g_f = CUTOFF_G*(image.shape[0] + image.shape[1] + 1)
N_vals_b_f = CUTOFF_B*(image.shape[0] + image.shape[1] + 1)
N_vals_f = N_vals_r_f + N_vals_g_f + N_vals_b_f

print("N_vals_i = {0}".format(N_vals_i))
print("N_vals_f = {0}".format(N_vals_f))
print("Percentage = {0}%".format(N_vals_f/N_vals_i*100.))
print("Assuming 1 byte for each value initially (integers in [0,255]) and 4 at the end (floats in [0,1]):")
print("Size_i = {0} kB".format(N_vals_i/1024))
print("Size_f = {0} kB".format((N_vals_f * 4)/1024))

plt.show()