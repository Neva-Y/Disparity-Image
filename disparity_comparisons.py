import os 
import math
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import multiprocessing as mp
import time
from skimage.graph import MCP
from scipy import ndimage

# Import images 
directory = './Dataset/'
allfiles = os.listdir(directory)
filenames = []

for filename in allfiles:
    if 'left' in filename:
        filenames.append(filename[0:filename.index('left')])

file = filenames[0]
boundary_nosmooth = 10
second_nosmooth = 2
boundary_size = 3
scale_percentage = 100
neighbourhood_l = 150
neighbourhood_r = 1
step_size = 1
measure = "NCC"


def nosmooth(boundary_nosmooth=10, measure="NCC"):
    
    start = time.time()
    calc_disp1 = np.zeros((h,w))
    min_bound_nosmooth = np.int32(boundary_nosmooth)
    max_bound_nosmooth = np.int32(len(img_l_grey[0])-boundary_nosmooth)
    row_index_nosmooth = [i for i in range(boundary_nosmooth, len(img_l_grey)-boundary_nosmooth, step_size)]
    args = [[i, boundary_nosmooth, measure] for i in range(boundary_nosmooth, len(img_l_grey)-boundary_nosmooth, step_size)]
    with mp.Pool(processes = mp.cpu_count()-1) as pool:
        disp_result = pool.starmap(compute_row_nosmooth, args)
        pool.close()
        pool.join()
        disp_result = np.array(disp_result)
        calc_disp1[row_index_nosmooth, min_bound_nosmooth:max_bound_nosmooth] = disp_result
    duration = time.time() - start
    compute_RMSE(calc_disp1, duration)
    return calc_disp1

def smooth():
    start = time.time()
    calc_disp2 = np.zeros((h,w))
    min_bound = np.int32(boundary_size)
    max_bound = np.int32(len(img_l_grey[0])-boundary_size)
    row_index = [i for i in range(boundary_size, len(img_l_grey)-boundary_size, step_size)]
    with mp.Pool(processes = mp.cpu_count()-1) as pool:
        disp_result = pool.map(compute_row_path, row_index)
        pool.close()
        pool.join()
        disp_result = np.array(disp_result)
        calc_disp2[row_index,min_bound:max_bound] = disp_result
    duration = time.time() - start
    
    compute_RMSE(calc_disp2, duration)
    
    return calc_disp2
        


def main():
    
    median_filter_size =  20

    print()
    # print("No smoothing NCC:")
    # calc_disp1 = nosmooth()
    # # calc_disp11 = nosmooth(second_nosmooth)


    # print()
    # print("No smoothing SSD:")
    # calc_disp111 = nosmooth(measure="SSD")

    start = time.time()

    print()
    print("Smoothing NCC:")
    calc_disp2 = smooth()


    print()
    print("Smoothing NCC WITH FILTER:")
    calc_disp3 = ndimage.median_filter(calc_disp2, median_filter_size)
    end = time.time()
    compute_RMSE(calc_disp3, end-start)

    print()
    print("No smoothing:")
    # compute_RMSE(calc_disp1)

    # print()    
    # print("Shortest path smoothing:")
    # compute_RMSE(calc_disp2)
    # print()    
    # print(f"Shortest path smoothing with median filter of size {median_filter_size}:")
    # compute_RMSE(calc_disp3)
    # print(f"Time to process disparity images: {end-start:.2f}s")



    colourmap = 'jet'

    plt.subplots(2,1, figsize=(7,7))

    # # THIS IS FOR THE BASE IMAGES
    # plt.subplot(2,1,1)
    # plt.imshow(img_l)
    # plt.title(f'Left Image')

    # plt.subplot(2,1,2)
    # plt.imshow(img_r)
    # plt.title(f'Right Image')

    # plt.savefig('MethodImages.png', bbox_inches="tight")

    #### THIS IS FOR THE WINDOW SIZE COMPARISON
    # plt.subplot(3,1,1)
    # plt.imshow(calc_disp1, cmap=colourmap,vmin=0)
    # plt.title(f'Basic NCC Disparity Image (Window Size = {boundary_nosmooth*2+1}x{boundary_nosmooth*2+1})')

    # plt.subplot(3,1,2)
    # plt.imshow(calc_disp11, cmap=colourmap, vmin=0)
    # plt.title(f'Basic NCC Disparity Image (Window = {second_nosmooth*2+1}x{second_nosmooth*2+1})')

    # plt.subplot(3,1,3)
    # plt.imshow(calc_disp111, cmap=colourmap, vmin=0)
    # plt.title(f'Basic SSD Disparity Image (Window = {boundary_nosmooth*2+1}x{boundary_nosmooth*2+1})')

    # plt.savefig('WindowSizeComparison.png', bbox_inches="tight")

    plt.subplot(2,1,1)
    # plt.imshow(calc_disp2, cmap=colourmap,vmin=0)
    plt.imshow(img_l)
    plt.title(f'Reference Image')
    

    plt.subplot(2,1,2)
    plt.imshow(calc_disp3, cmap=colourmap,vmin=0)
    plt.title(f'Image 1 Disparity Image')
    plt.savefig('Image11.png', bbox_inches="tight")
    # plt.subplot(2,2,4)
    # plt.imshow(img_d, cmap=colourmap,vmin=0)
    # plt.title('Ground truth')
    # plt.savefig("Image 2.png")

    plt.show()

left_image_name = directory + file + 'left.jpg'
right_image_name = directory+ file + 'right.jpg'
disp_image_name = directory+ file + 'disparity.png'

# Read images
img_l = cv2.imread(left_image_name)
img_r = cv2.imread(right_image_name)
img_l_grey_raw = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
img_r_grey_raw = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

# img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
# img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

# Convert to grey images
img_l_grey_full = np.int32(cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY))
img_r_grey_full = np.int32(cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY))
img_l_grey = img_l_grey_full
img_r_grey = img_r_grey_full

img_l = img_l[...,::-1]
img_r = img_r[...,::-1]

img_d = cv2.imread(disp_image_name,-1)/256.0

# note: can perform downscaling of the images to speed up computation at the cost of quality degradation
width = int(img_l_grey_full.shape[1] * scale_percentage / 100)
height = int(img_l_grey_full.shape[0] * scale_percentage / 100)
dim = (width, height)
img_l_grey = np.int32(cv2.resize(img_l_grey_full.astype('float32'), dim, interpolation = cv2.INTER_AREA))
img_r_grey = np.int32(cv2.resize(img_r_grey_full.astype('float32'), dim, interpolation = cv2.INTER_AREA))
img_d = cv2.resize(img_d.astype('float32'), dim, interpolation = cv2.INTER_AREA)

h,w = img_r_grey.shape


# Code adapted from skimage.shortest_path algorithm to specify desired starting and final points
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

# Calcualte basic matching using NCC and SSD
def compute_row_nosmooth(row_index,boundary_nosmooth, measure="NCC"):
    row = img_l_grey[row_index]
    disp = []

    for col_index_left in range(boundary_nosmooth,len(row)-boundary_nosmooth, step_size):
        
        centre_l = [row_index, col_index_left]

        min_l_x = centre_l[1] - boundary_nosmooth
        max_l_x = centre_l[1] + boundary_nosmooth + 1

        min_l_y = centre_l[0] - boundary_nosmooth
        max_l_y = centre_l[0] + boundary_nosmooth + 1

        patch_l = img_l_grey[min_l_y:max_l_y, min_l_x:max_l_x]
        # patch_l = cv2.GaussianBlur(np.float32(patch_l), (boundary_nosmooth*2+1, boundary_nosmooth*2+1), 1)

        # Take horizontal window to search, ensuring values don't exceed bounds
        min_bound = max(col_index_left-neighbourhood_l, boundary_nosmooth)
        max_bound = min(col_index_left+neighbourhood_r, len(row)-boundary_nosmooth)
        
        # Indices of all column centres to check
        col_centres = np.arange(min_bound,max_bound) 
        
        # Generate arrays for each column centre in steps of 1 using the boundary size, each having 2*boundary_nosmooth elements
        col_linspace = np.int32(np.transpose(np.linspace(col_centres-boundary_nosmooth, col_centres+boundary_nosmooth, 2*boundary_nosmooth+1)))


        # Generate right patch using the constant rows in loop iteration and column arrays, transpose to get correct shape
        patch_r = np.transpose(img_r_grey[min_l_y:max_l_y, col_linspace], (1,0,2)) 
        # patch_r = cv2.GaussianBlur(np.float32(patch_r), (boundary_nosmooth*2+1, boundary_nosmooth*2+1), 1)

        
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

# Compute the smoothed version using the scanline consistency method 
def compute_row_path(row_index):
    row = img_l_grey[row_index]
    dsi = []
    for col_index_left in range(boundary_size,len(row)-boundary_size, step_size):

        centre_l = [row_index, col_index_left]

        min_l_x = centre_l[1] - boundary_size
        max_l_x = centre_l[1] + boundary_size + 1

        min_l_y = centre_l[0] - boundary_size
        max_l_y = centre_l[0] + boundary_size + 1

        min_bound = np.int32(boundary_size)
        max_bound = np.int32(len(img_l_grey[row_index])-boundary_size)

        patch_l = img_l_grey[min_l_y:max_l_y, min_l_x:max_l_x]
        patch_l = cv2.GaussianBlur(np.float32(patch_l), (boundary_size*2+1, boundary_size*2+1), 1)

        # Indices of all column centres to check
        col_centres = np.arange(min_bound,max_bound) 

        # Generate arrays for each column centre in steps of 1 using the boundary size, each having 2*boundary_size elements
        col_linspace = np.int32(np.transpose(np.linspace(col_centres-boundary_size, col_centres+boundary_size, 2*boundary_size+1)))

        # Generate right patch using the constant rows in loop iteration and column arrays, transpose to get correct shape

        patch_r = np.transpose(img_r_grey[min_l_y:max_l_y, col_linspace], (1,0,2)) 
        patch_r = cv2.GaussianBlur(np.float32(patch_r), (boundary_size*2+1, boundary_size*2+1), 1)

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
    

# Compute the metrics and print them to console
def compute_RMSE(disp, duration):
    h,w = img_d.shape
    assert(img_d.shape == disp.shape)
    MSE = []

    for i in range(h):
        for j in range(w):
            if img_d[i,j] != 0:
                MSE.append(np.square(disp[i,j]-img_d[i,j]))
    MSE = np.array(MSE)
    print(f'RMSE of {math.sqrt(MSE.mean()):.2f}')

    pixError = []
    for i in range(h):
        for j in range(w):
            if img_d[i,j] != 0:
                pixError.append(abs(disp[i,j]-img_d[i,j]))

    pixError = np.array(pixError)
    for i in [0.25,0.5,1,2,4][::-1]:
        print(f'Fraction of {len(pixError[pixError < i])/len(img_d[img_d>0]):.2f} with pixel error less than {i} pixels')

    print(f'Duration of {duration}')

if __name__ == '__main__':
    main()