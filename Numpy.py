
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
img= imread("C:\\Users\\bhargav\\Desktop\\shuba.jpg")
plt.imshow(img, cmap = 'Greys')
plt.title("Shuba's Original pic")
print(img.shape)

img_flip = img.copy()
img_flip = np.fliplr(img_flip)
plt.figure()
plt.imshow(img_flip, cmap = 'Greys')
plt.title("Shuba's mirror pic")

img_comp = img.copy()
img_comp=img_comp[1400:4400,700:3050,:]
plt.figure()
plt.imshow(img_comp, cmap = 'Greys')
plt.title("Shuba's cropped pic")


# =============================================================================
# 
# import os.path
# import matplotlib.pyplot as plt 
# from skimage.io import imread 
# from skimage import data_dir
# import numpy as np 
# img = imread(os.path.join(data_dir, 'phantom.png')) 
# plt.imshow(img)
# plt.title('Original Image') 
# new_img = img.copy()
# new_img[img>0.15]=255
# new_img[img<=0.15]=0 
# plt.figure()
# plt.imshow(new_img)
# plt.title('Black and white thresholding')
# fliplr_img = np.fliplr(img) 
# plt.figure()
# plt.imshow(fliplr_img)
# plt.title('Flip Left & Right')
# print(fliplr_img.shape)
# print(fliplr_img.shape[0])
# compressed_img = np.compress([(i%2)==0 for i in range(fliplr_img.shape[0])], fliplr_img, axis=0) 
# compressed_img = np.compress([(i%2)==0 for i in range(compressed_img.shape[1])], compressed_img, axis=1)
# plt.figure()
# plt.imshow(compressed_img)
# plt.title('Compressed Flipped Image')
# plt.show()
# =============================================================================

#2x + 3y + 2z = 1
#x  + 0y + 3z = 2
#2x + 2y + 3z = 3

 
import numpy as np
# =============================================================================
# a = np.array([[2,3,2],[1,0,3],[2,2,3]])
# b = np.array([1,2,3])
# ''' Checking if system of equation has unique solution '''
# print(np.linalg.det(a)) 
# # 5.0
# ''' Since det = 5 which is non-zero. Hence, we have unique solutions
#  Finding unique solution '''
# print(np.linalg.solve(a, b))
# # [ 2.  3.]
# ''' Calculating Inverse: Since, determinant is non-zero 
#  hence, matrix is invertible '''
# print(np.linalg.inv(a))
# # [[ 0.4 -0.2]
# #  [-0.2  0.6]]
# ''' Calculating Rank of the matrix '''
# print(np.linalg.matrix_rank(a))
# =============================================================================


# =============================================================================
# import numpy as np
# a = np.arange(9).reshape(3,-1)
# print(a)
# b = np.ceil(np.linspace(7,15,9)).reshape(3,-1)
# print(b)
# print(np.greater_equal(a,b))
# print(np.count_nonzero(np.greater_equal(a,b)))
# =============================================================================









