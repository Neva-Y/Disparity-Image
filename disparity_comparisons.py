import os 
import math
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import multiprocessing as mp
import time

# Import images 
directory = './Dataset/'
allfiles = os.listdir(directory)
filenames = []

for filename in allfiles:
    if 'left' in filename:
        filenames.append(filename[0:filename.index('left')])

# List of files obtained

# Perform on first image for now 

file = filenames[3]

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

img_d = cv2.imread(disp_image_name,-1)/256.0


# note: can perform downscaling of the images to speed up computation at the cost of quality degradation
# scale_percentage = 30
# width = int(img_l_grey_full.shape[1] * scale_percentage / 100)
# height = int(img_l_grey_full.shape[0] * scale_percentage / 100)
# dim = (width, height)
# img_l_grey = np.int32(cv2.resize(img_l_grey_full.astype('float32'), dim, interpolation = cv2.INTER_AREA))
# img_r_grey = np.int32(cv2.resize(img_r_grey_full.astype('float32'), dim, interpolation = cv2.INTER_AREA))
# img_d = np.int32(cv2.resize(img_d.astype('float32'), dim, interpolation = cv2.INTER_AREA))

h,w = img_r_grey.shape
boundary_size = 3
neighbourhood_l = 200
neighbourhood_r = 50
step_size = 1
measure = "NCC"

# Take horizontal window to search, ensuring values don't exceed bounds
min_bound = np.int32(boundary_size)
max_bound = np.int32(len(img_l_grey[0])-boundary_size)


from skimage.graph import shortest_path, route_through_array, MCP
# Code adapted from skimage.shortest_path algorithm
def my_shortest_path(arr, reach=1, axis=-1, output_indexlist=False):
    # First: calculate the valid moves from any given position. Basically,
    # always move +1 along the given axis, and then can move anywhere within
    # a grid defined by the reach.
    if axis < 0:
        axis += arr.ndim
    offset_ind_shape = (2 * reach + 1,) * (arr.ndim - 1)
    offset_indices = np.indices(offset_ind_shape) - reach
    offset_indices = np.insert(offset_indices, axis,
                               np.ones(offset_ind_shape), axis=0)
    offset_size = np.multiply.reduce(offset_ind_shape)
    offsets = np.reshape(offset_indices, (arr.ndim, offset_size), order='F').T

    # Valid starting positions are anywhere on the hyperplane defined by
    # position 0 on the given axis. Ending positions are anywhere on the
    # hyperplane at position -1 along the same.
    non_axis_shape = arr.shape[:axis] + arr.shape[axis + 1:]
    non_axis_indices = np.indices(non_axis_shape)
    non_axis_size = np.multiply.reduce(non_axis_shape)
    start_indices = np.insert(non_axis_indices, axis,
                              np.zeros(non_axis_shape), axis=0)
    starts = np.reshape(start_indices, (arr.ndim, non_axis_size), order='F').T[0:1]
    end_indices = np.insert(non_axis_indices, axis,
                            np.full(non_axis_shape, -1,
                                    dtype=non_axis_indices.dtype), axis=0)
    ends = np.reshape(end_indices, (arr.ndim, non_axis_size), order='F').T[-2:-1]
    
    # Find the minimum-cost path to one of the end-points
    m = MCP(arr, offsets=offsets)
    costs, traceback = m.find_costs(starts, ends, find_all_ends=True)

    # Figure out which end-point was found
    for end in ends:
        cost = costs[tuple(end)]
        if cost != np.inf:
            break
    traceback = m.traceback(end)

    if not output_indexlist:
        traceback = np.array(traceback)
        traceback = np.concatenate([traceback[:, :axis],
                                    traceback[:, axis + 1:]], axis=1)
        traceback = np.squeeze(traceback)

    return traceback, cost

def compute_row_nosmooth(row_index):
    row = img_l_grey[row_index]
    disp = []
    for col_index_left in range(boundary_size,len(row)-boundary_size, step_size):
        
        centre_l = [row_index, col_index_left]

        min_l_x = centre_l[1] - boundary_size
        max_l_x = centre_l[1] + boundary_size + 1

        min_l_y = centre_l[0] - boundary_size
        max_l_y = centre_l[0] + boundary_size + 1

        patch_l = img_l_grey[min_l_y:max_l_y, min_l_x:max_l_x]

        # Take horizontal window to search, ensuring values don't exceed bounds
        min_bound = max(col_index_left-neighbourhood_l, boundary_size)
        max_bound = min(col_index_left+neighbourhood_r, len(row)-boundary_size)
        
        # Indices of all column centres to check
        col_centres = np.arange(min_bound,max_bound) 
        
        # Generate arrays for each column centre in steps of 1 using the boundary size, each having 2*boundary_size elements
        col_linspace = np.int32(np.transpose(np.linspace(col_centres-boundary_size, col_centres+boundary_size, 2*boundary_size+1)))

        # Generate right patch using the constant rows in loop iteration and column arrays, transpose to get correct shape
        patch_r = np.transpose(img_r_grey[min_l_y:max_l_y, col_linspace], (1,0,2)) 
        
        # Choose distance measure to generate disparity map
        if measure == "NCC":
            norm_corr = np.sum(patch_l*patch_r, axis=(1,2))/(np.linalg.norm(patch_l)*np.linalg.norm(patch_r, axis=(1,2)))
            min_col_index = np.argmax(norm_corr) + min_bound
            disp.append(abs(min_col_index - col_index_left))

        elif measure == "SSD":
            error = np.sum((patch_l-patch_r)**2, axis=(1,2))
            min_col_index = np.argmin(error) + min_bound
            disp.append(abs(min_col_index - col_index_left))
        
    return disp

def compute_row_path(row_index):
    row = img_l_grey[row_index]
    dsi = []
    for col_index_left in range(boundary_size,len(row)-boundary_size, step_size):

        centre_l = [row_index, col_index_left]

        min_l_x = centre_l[1] - boundary_size
        max_l_x = centre_l[1] + boundary_size + 1

        min_l_y = centre_l[0] - boundary_size
        max_l_y = centre_l[0] + boundary_size + 1

        patch_l = img_l_grey[min_l_y:max_l_y, min_l_x:max_l_x]

        # Indices of all column centres to check
        col_centres = np.arange(min_bound,max_bound) 

        # Generate arrays for each column centre in steps of 1 using the boundary size, each having 2*boundary_size elements
        col_linspace = np.int32(np.transpose(np.linspace(col_centres-boundary_size, col_centres+boundary_size, 2*boundary_size+1)))

        # Generate right patch using the constant rows in loop iteration and column arrays, transpose to get correct shape

        patch_r = np.transpose(img_r_grey[min_l_y:max_l_y, col_linspace], (1,0,2)) 


        # Choose distance measure to generate disparity map
        if measure == "NCC":
            norm_corr = np.sum(patch_l*patch_r, axis=(1,2))/(np.linalg.norm(patch_l)*np.linalg.norm(patch_r, axis=(1,2)))
            dsi.append(norm_corr)
            
        elif measure == "SSD":
            error = np.sum((patch_l-patch_r)**2, axis=(1,2))
            dsi.append(error)
    
    arr = np.transpose(np.array(dsi))
    if measure == "NCC":
        arr = np.array([[ 0.001 if i > j or i+150 < j else arr[i,j] for j in range(len(arr))] for i in range(len(arr))])
        arr = 1/(arr)
        
    elif measure == "SSD":
        arr = np.array([[ np.inf if i > j else arr[i,j] for j in range(len(arr))] for i in range(len(arr))])
    
    p, cost = my_shortest_path(arr, reach=5)
    p1 = np.arange(0,len(arr))
    points = np.array([[p[i], p1[i]] for i in range(len(p))])
    disp = abs(np.subtract(points[:,1], points[:,0]))

    return disp
    
def compute_RMSE(calc_disp2):
    h,w = img_d.shape
    assert(img_d.shape == calc_disp2.shape)
    MSE = []

    for i in range(h):
        for j in range(w):
            if img_d[i,j] != 0:
                MSE.append(abs(calc_disp2[i,j]-img_d[i,j]))

    MSE = np.array(MSE)
    for i in [0.25,0.5,1,2,4][::-1]:
        print(f'Fraction of {len(MSE[MSE < i])/len(img_d[img_d>0]):.2f} with pixel error less than {i} pixels')

def main():
    start = time.time()
    calc_disp1 = np.zeros((h,w))
    calc_disp2 = np.zeros((h,w))
    bilateral_disp = np.zeros((h,w))
    row_index = [i for i in range(boundary_size, len(img_l_grey)-boundary_size, step_size)]

    with mp.Pool(processes = mp.cpu_count()-1) as pool:
        disp_result = pool.map(compute_row_nosmooth, row_index)
        pool.close()
        pool.join()
        disp_result = np.array(disp_result)
        calc_disp1[row_index,min_bound:max_bound] = disp_result
        bilateral_disp = cv2.bilateralFilter(np.uint8(calc_disp1),7,50,50)

    with mp.Pool(processes = mp.cpu_count()-1) as pool:
        disp_result = pool.map(compute_row_path, row_index)
        pool.close()
        pool.join()
        disp_result = np.array(disp_result)
        calc_disp2[row_index,min_bound:max_bound] = disp_result


    print()
    print("RMSE with no smoothing:")
    compute_RMSE(calc_disp1)
    print()
    print("RMSE with bilateral filter smoothing:")
    compute_RMSE(bilateral_disp)
    print()    
    print("RMSE with shortest path smoothing:")
    compute_RMSE(calc_disp2)
    end = time.time()
    print(f"Time to process disparity images: {end-start:.2f}s")

    plt.subplots(2,2, figsize=(15,10))
    plt.subplot(2,2,1)
    plt.imshow(calc_disp1, cmap='plasma',vmin=0)
    plt.title('Disparity image with no smoothing')

    plt.subplot(2,2,2)
    plt.imshow(bilateral_disp, cmap='plasma',vmin=0)
    plt.title('Disparity image with bilateral filter smoothing')

    plt.subplot(2,2,3)
    plt.imshow(calc_disp2, cmap='plasma',vmin=0)
    plt.title('Disparity image using shortest path')

    plt.subplot(2,2,4)
    plt.imshow(img_d, cmap='plasma',vmin=0)
    plt.title('Ground truth')
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()