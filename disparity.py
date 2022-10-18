import os 
import math
import matplotlib.pyplot as plt 
import numpy as np
import cv2
from tqdm import tqdm
import sys
import concurrent.futures

# Import images 
directory = './Dataset/'
allfiles = os.listdir(directory)

filenames = []

for filename in allfiles:
    if 'left' in filename:
        filenames.append(filename[0:filename.index('left')])

# List of files obtained

# Perform on first image for now 

file = filenames[0]

left_image_name = directory + file + 'left.jpg'
right_image_name = directory+ file + 'right.jpg'
disp_image_name = directory+ file + 'disparity.png'

# Read images and convert to RGB 
img_l = cv2.imread(left_image_name)
img_r = cv2.imread(right_image_name)

# Grey images
img_l_grey_full = np.int32(cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY))
img_r_grey_full = np.int32(cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY))
img_l_grey = img_l_grey_full
img_r_grey = img_r_grey_full

img_l = img_l[...,::-1]
img_r = img_r[...,::-1]

img_d = cv2.imread(disp_image_name)
img_d_grey = np.int32(cv2.cvtColor(img_d, cv2.COLOR_BGR2GRAY))


# note: can perform downscaling of the images to speed up computation at the cost of quality degradation
# scale_percentage = 50
# width = int(img_l_grey_full.shape[1] * scale_percentage / 100)
# height = int(img_l_grey_full.shape[0] * scale_percentage / 100)
# dim = (width, height)

# img_l_grey = np.int32(cv2.resize(img_l_grey_full.astype('float32'), dim, interpolation = cv2.INTER_AREA))
# img_r_grey = np.int32(cv2.resize(img_r_grey_full.astype('float32'), dim, interpolation = cv2.INTER_AREA))

# Optised plotting of the disparity map
h,w = img_l_grey.shape
calc_disp2 = np.zeros((h,w))
boundary_size = 10
neighbourhood_l = 200
nighbourhood_r = 50
step_size = 1
measure = "NCC"

def compute_disparity_for_row(row_index):
    row = img_l_grey[row_index]

    for col_index_left in range(boundary_size, len(row)-boundary_size, step_size):
        
        centre_l = [row_index, col_index_left]

        min_l_x = centre_l[1] - boundary_size
        max_l_x = centre_l[1] + boundary_size

        min_l_y = centre_l[0] - boundary_size
        max_l_y = centre_l[0] + boundary_size

        patch_l = img_l_grey[min_l_y:max_l_y, min_l_x:max_l_x]

        # Take horizontal window to search, ensuring values don't exceed bounds
        min_bound = max(col_index_left-neighbourhood_l, boundary_size)
        max_bound = min(col_index_left+nighbourhood_r, len(row)-boundary_size)
        
        # Indices of all column centres to check
        col_centres = np.arange(min_bound,max_bound) 
        
        # Generate arrays for each column centre in steps of 1 using the boundary size, each having 2*boundary_size elements
        col_linspace = np.int32(np.transpose(np.linspace(col_centres-boundary_size, col_centres+boundary_size, 2*boundary_size)))

        # Generate right patch using the constant rows in loop iteration and column arrays, transpose to get correct shape
        patch_r = np.transpose(img_r_grey[min_l_y:max_l_y, col_linspace], (1,0,2)) 
        
        # Choose distance measure to generate disparity map
        if measure == "NCC":
            norm_corr = np.sum(patch_l*patch_r, axis=(1,2))/(np.linalg.norm(patch_l)*np.linalg.norm(patch_r, axis=(1,2)))
            min_col_index = np.argmax(norm_corr) + min_bound - 1
            calc_disp2[row_index][col_index_left] = abs(min_col_index - col_index_left)

        elif measure == "SSD":
            error = np.sum((patch_l-patch_r)**2, axis=(1,2))
            min_col_index = np.argmin(error) + min_bound - 1
            calc_disp2[row_index][col_index_left] = abs(min_col_index - col_index_left)


def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        row_index = [i for i in range(boundary_size, len(img_l_grey)-boundary_size, step_size)]
        executor.map(compute_disparity_for_row, row_index)
    plt.imshow(calc_disp2, cmap='gray', vmin=0, vmax=255)
    plt.savefig("disparity_parallel.png")
    plt.show()

if __name__ == '__main__':
    main()


