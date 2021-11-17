# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 18:40:51 2021

@author: Jerry
"""
import os
import math
import array
import struct
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

# In[]

def read_image(path):  #path = os.path.join('data/', images[0])
    file_extention = path.split('.')[-1]
    
    if file_extention == 'bmp':
        f = open(path, 'rb')
        bmp_header_b = f.read(0x36) # read top 54 byte of bmp file 
        bmp_header_s = struct.unpack('<2sI2H4I2H6I', bmp_header_b) # parse data with struct.unpack()
        pixel_index = bmp_header_s[4] - 54
        bmp_rgb_data_b = f.read()[pixel_index:] # read pixels of bmp file 
        list_b = array.array('B', bmp_rgb_data_b).tolist()
        rgb_data_3d_list  = np.reshape(list_b, (bmp_header_s[6], bmp_header_s[7], bmp_header_s[8])).tolist() # reshape pixel with height, width, RGB channel of image
        
        image = []
        for row in range(len(rgb_data_3d_list)):
            image.insert(0, rgb_data_3d_list[row])
        
        
    elif file_extention == 'raw':
        f = open(path, 'rb').read()
        image = reshape_byte(f) # reshape byte
        
    image = np.array(image) # store into np.array
    
    if len(image.shape) != 3:
        image = np.reshape(image, (image.shape[0], image.shape[1]))
    
    return image

def reshape_byte(byte, size=[512,512]): 
    new_img = []
    
    for row in range(size[0]):
        new_img_row = []
        
        for col in range(size[1]):
            new_img_row.append(byte[row * size[1] + col])
            
        new_img.append(new_img_row)
    
    # new_img = np.reshape(new_img, (size[0], size[1], 1))
    
    return new_img

def gamma_Transform(image, gamma = 2):
    
    new_image = np.power(image / float(np.max(image)), gamma)
    
    new_image = np.around((new_image - np.min(new_image)) / (np.max(new_image) - np.min(new_image)) * 255)
    
    return new_image.astype('uint8')

def global_histogram_equalization(image):  # image = sub_array
    
    hist, bins = np.histogram(image.ravel(), 256, [0,255])
    pdf = hist / image.size
    cdf = pdf.cumsum()
    equ_value = np.around(cdf * 255).astype('uint8')
    new_image = equ_value[image]
    
    return new_image

def local_histogram_equalization(image, window_size = [5, 5]):
    
    midpoint = int(np.floor(window_size[0]/2))
    
    padded = replicate_padding(image, window_size)
    
    sliced = get_slices(padded, window_size[0], window_size[1])
    
    output_shape = tuple(np.array(padded.shape) - np.array(window_size) + 1)
    
    outer = []
    for i, sub_array in tqdm(enumerate(sliced)):
        new_sub_array = global_histogram_equalization(sub_array)
        val = new_sub_array[midpoint, midpoint]
        outer.append(np.uint8(val))
    
    new_image = np.reshape(outer, output_shape)
    
    return new_image

def replicate_padding(arr, window_size): # arr = image
    """Perform replicate padding on a numpy array."""
    # Calculate padding size for original image size
    midpoint = int(np.floor(window_size[0]/2))
    
    padding_size = int((window_size[0] - 1) / 2)
    
    new_pad_shape = tuple(np.array(arr.shape) + 2 * padding_size)
    padded_array = np.zeros(new_pad_shape) #create an array of zeros with new dimensions
    
    # perform replication
    padded_array[midpoint:-midpoint, midpoint:-midpoint] = arr        # result will be zero-pad
    padded_array[0:midpoint, midpoint:-midpoint] = arr[0:midpoint]        # perform edge pad for top row
    padded_array[-midpoint:, midpoint:-midpoint] = arr[-midpoint:]     # edge pad for bottom row
    padded_array.T[0:midpoint, midpoint:-midpoint] = arr.T[0:midpoint]   # edge pad for first column
    padded_array.T[-midpoint:, midpoint:-midpoint] = arr.T[-midpoint:] # edge pad for last column
    
    #at this point, all values except for the 4 corners should have been replicated
    padded_array[0:midpoint, 0:midpoint] = arr[0:midpoint, 0:midpoint]     # top left corner
    padded_array[-midpoint:, 0:midpoint] = arr[-midpoint:, 0:midpoint]   # bottom left corner
    padded_array[0:midpoint, -midpoint:] = arr[0:midpoint, -midpoint:]   # top right corner 
    padded_array[-midpoint:, -midpoint:] = arr[-midpoint:, -midpoint:] # bottom right corner

    return padded_array

def get_slices(arr, width, height):
    """Collects m (width) x n (height) slices for a padded array"""
    slices = []
    for i in range(len(arr) - width + 1): #get row
        for j in range(len(arr[i]) - height + 1): #get column
            r = i + width
            c = j + height
            sub_array = arr[i:r, j:c]
            slices.append(sub_array)
    return np.array(slices).astype('int')

def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        # lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
        
    return lookup_table

def match_histograms(src_image, ref_image):
    """
    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :return: image_after_matching
    :rtype: image (array)
    """
    # Split the images into the different color channels
    # b means blue, g means green and r means red
    
    
    # Compute the b, g, and r histograms separately
    # The flatten() Numpy method returns a copy of the array c
    # collapsed into one dimension.
    hist_src, bins_src = np.histogram(src_image.ravel(), 256, [0,255])
    
    # Compute the normalized cdf for the source and reference image
    pdf_src = hist_src / src_image.size
    cdf_src = pdf_src.cumsum()
    
    # Compute the b, g, and r histograms separately
    # The flatten() Numpy method returns a copy of the array c
    # collapsed into one dimension.
    hist_ref, bins_ref = np.histogram(ref_image.ravel(), 256, [0,255])
    
    # Compute the normalized cdf for the source and reference image
    pdf_ref = hist_ref / ref_image.size
    cdf_ref = pdf_ref.cumsum()
    
    
    # Make a separate lookup table for each color
    lookup_table = calculate_lookup(cdf_src, cdf_ref)
 
    # Use the lookup function to transform the colors of the original
    # source image
 
    image_after_matching = src_image.copy()
    for row in range(image_after_matching.shape[0]):
        for col in range(image_after_matching.shape[1]):
            image_after_matching[row, col] = lookup_table[image_after_matching[row, col]]
 
    return image_after_matching

def plot_hist(image, image_name):
    
    hist, bins = np.histogram(image.ravel(), 256, [0,255])
    plt.plot(hist)
    plt.title(f'Hitogram of {image_name}')
    plt.savefig(os.path.join(save_path, f'{image_name}_hist.png'))
    plt.close()

def create_Gaussian_kernel(kernel_size = 3, sigma = 1):
    
    half_length = kernel_size // 2
    x, y = np.mgrid[-half_length : half_length + 1, -half_length : half_length + 1]
    gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp((-(x**2+y**2)) / (2 * sigma))

    #Normalization
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    
    # plt.imshow(gaussian_kernel, cmap=plt.get_cmap('jet'), interpolation='nearest')
    plt.imshow(gaussian_kernel, cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.title(f'kernel size = {kernel_size}')
    plt.savefig(os.path.join(save_path_2, f'kernel size_{kernel_size}.png'))
    plt.close()
    
    return gaussian_kernel

def gaussian_filter(image, kernel_size = 3, sigma = 1, stride = 1):
    height, width = image.shape
    padding_size = np.floor((height * (stride - 1) - stride + kernel_size) / 2).astype(int)
    
    temp_image = np.zeros([height + 2 * padding_size, height + 2 * padding_size])
    temp_image[padding_size : -padding_size, padding_size : -padding_size] = image
    
    gaussian_kernel = create_Gaussian_kernel(kernel_size, sigma)
    new_image = np.zeros([height, height])
    
    for row in range(height):
        for col in range(height):
            new_image[row, col] = np.sum(temp_image[row : row + kernel_size, col : col + kernel_size] * gaussian_kernel)
    
    new_image = new_image.astype(np.uint8)

    return new_image

def create_averaging_kernel(kernel_size = 3):
    
    average_kernel = np.ones([kernel_size, kernel_size]) / kernel_size**2
    
    return average_kernel


def averaging_filter(image, kernel_size = 3, stride = 1):
    height, width = image.shape
    padding_size = np.floor((height * (stride - 1) - stride + kernel_size) / 2).astype(int)
    
    temp_image = np.zeros([height + 2 * padding_size, height + 2 * padding_size])
    temp_image[padding_size : -padding_size, padding_size : -padding_size] = image
    
    average_kernel = create_averaging_kernel(kernel_size)
    
    new_image = np.zeros([height, height])
    
    for row in range(height):
        for col in range(height):
            new_image[row, col] = np.sum(temp_image[row : row + kernel_size, col : col + kernel_size] * average_kernel)
    
    return new_image

def laplacian_filter(image, kernel_size, stride = 1):
    # Formatted this way for readability
    if kernel_size == 3:
        laplacian_kernel = np.array((
                           	[0, 1, 0],
                           	[1, -4, 1],
                           	[0, 1, 0]), dtype="int")
        
    elif kernel_size == 5:
        laplacian_kernel = np.array((
                            [0, -1, -1, -1, 0],
                            [-1, -1, -1, -1, -1],
                            [-1, -1, 21, -1, -1],
                            [-1, -1, -1, -1, -1],
                            [0, -1, -1, -1, 0]), dtype="int")
        
    elif kernel_size == 7:
        laplacian_kernel = np.array((
                            [0, 0, -1, -1, -1, 0, 0],
                            [0, -1, -3, -3, -3, -1, 0],
                            [-1, -3, 0, 7, 0, -3, -1],
                            [-1, -3, 7, 24, 7, -3, -1],
                            [-1, -3, 0, 7, 0, -3, -1],
                            [0, -1, -3, -3, -3, -1, 0],
                            [0, 0, -1, -1, -1, 0, 0],), dtype="int")
    
    height, width = image.shape
    padding_size = np.floor((height * (stride - 1) - stride + kernel_size) / 2).astype(int)
    
    temp_image = np.zeros([height + 2 * padding_size, height + 2 * padding_size])
    temp_image[padding_size : -padding_size, padding_size : -padding_size] = image
    
    new_image = np.zeros([height, height])
    
    for row in range(height):
        for col in range(height):
            new_image[row, col] = np.sum(temp_image[row : row + kernel_size, col : col + kernel_size] * laplacian_kernel)
    
    return new_image

def create_sobel_kernel(kernel_size = 3, stride = 1): 
    
    if kernel_size == 3:
        sobel_kernel = np.array([[1, 2, 1]]).T * np.array([[1, 0, -1]])
    
    if kernel_size == 5:
        small_kernel_size = kernel_size - 2
        small_sobel_kernel = create_sobel_kernel(kernel_size = small_kernel_size)
        
        smooth_kernel = np.array([[1, 2, 1]]).T * np.array([[1, 2, 1]])
        
        kernel_size = smooth_kernel.shape[0]
    
        height = small_sobel_kernel.shape[0]
        padding_size = np.floor((height * (stride - 1) - stride + kernel_size) / 2).astype(int)
        target_height = height + 2
            
        temp_sobel_kernel = np.zeros([target_height + 2 * padding_size, target_height + 2 * padding_size])
        
        temp_sobel_kernel[padding_size * 2 : -padding_size * 2, padding_size * 2 : -padding_size * 2] = small_sobel_kernel
        
        sobel_kernel = np.zeros([target_height, target_height])
        
        for row in range(target_height):
            for col in range(target_height):
                sobel_kernel[row, col] = np.sum(temp_sobel_kernel[row : row + kernel_size, col : col + kernel_size] * smooth_kernel)
    
    if kernel_size == 7:
        small_kernel_size = kernel_size - 2
        small_sobel_kernel = create_sobel_kernel(kernel_size = small_kernel_size)
        
        smooth_kernel = np.array([[1, 2, 1]]).T * np.array([[1, 2, 1]])
        
        kernel_size = smooth_kernel.shape[0]
    
        height = small_sobel_kernel.shape[0]               
        padding_size = np.floor((height * (stride - 1) - stride + kernel_size) / 2).astype(int)
        target_height = height + 2
            
        temp_sobel_kernel = np.zeros([target_height + 2 * padding_size, target_height + 2 * padding_size])
        
        temp_sobel_kernel[padding_size * 2 : -padding_size * 2, padding_size * 2 : -padding_size * 2] = small_sobel_kernel
        
        sobel_kernel = np.zeros([target_height, target_height])
        
        for row in range(target_height):
            for col in range(target_height):
                sobel_kernel[row, col] = np.sum(temp_sobel_kernel[row : row + kernel_size, col : col + kernel_size] * smooth_kernel)
    
    return sobel_kernel


def sobel_filter(image, kernel_size = 3, stride = 1):
    sobel_kernel = create_sobel_kernel(kernel_size)

    # Here we define the matrices associated with the Sobel filter
    Gx = sobel_kernel
    Gy = sobel_kernel.T
    
    height, width = image.shape
    padding_size = np.floor((height * (stride - 1) - stride + kernel_size) / 2).astype(int)
    
    temp_image = np.zeros([height + 2 * padding_size, height + 2 * padding_size])
    temp_image[padding_size : -padding_size, padding_size : -padding_size] = image
    
    new_image = np.zeros([height, height])
    
    for row in range(height):
        for col in range(height):
            gx = np.sum(temp_image[row : row + kernel_size, col : col + kernel_size] * Gx)
            gy = np.sum(temp_image[row : row + kernel_size, col : col + kernel_size] * Gy)
            
            new_image[row, col] = np.sqrt(gx ** 2 + gy ** 2)
    
    return new_image
    
def convolve(image, kernel, stride = 1):

    height, width = image.shape
    kernel_size = kernel.shape[0]
    padding_size = np.floor((height * (stride - 1) - stride + kernel_size) / 2).astype(int)
    
    temp_image = np.zeros([height + 2 * padding_size, height + 2 * padding_size])
    temp_image[padding_size : -padding_size, padding_size : -padding_size] = image
    
    new_image = np.zeros([height, height])
    
    for row in range(height):
        for col in range(height):
            new_image[row, col] = np.sum(temp_image[row : row + kernel_size, col : col + kernel_size] * kernel)    
    
    return new_image

def medium_filter(image, kernel_size = 3):
    
    height, width = image.shape
    weight = (kernel_size ** 2) - (2 * kernel_size) + 1 # 計算保留線性紋理的最小 weight
    
    new_image = image.copy()
    
    for row in range(height - kernel_size + 1):
        for col in range(height - kernel_size + 1):
            temp_kernel = image[row : row + kernel_size, col : col + kernel_size]
            weighted_center = np.array([[temp_kernel[int(kernel_size / 2), int(kernel_size / 2)]] * weight]) 
            
            sorted_array = np.sort(np.concatenate((temp_kernel.reshape((1,-1)), weighted_center), axis=1))
            sorted_array_median  = sorted_array[0, int(sorted_array.shape[1]/2)]
            
            new_image[row + int(kernel_size / 2), col + int(kernel_size / 2)] = sorted_array_median 
    
    return new_image

def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)

def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))
    # return math.exp(- (x ** 2) / (2 * sigma ** 2))

# def apply_bilateral_filter(image, temp_image, row, col, kernel_size, sigma_c, sigma_s, gaussian_c = True):
#     hl = kernel_size // 2
#     i_filtered = 0
#     Wp = 0
    
#     for temp_row in range(kernel_size):
#         for temp_col in range(kernel_size):
#             # gc = gaussian(temp_image[k][l] - temp_image[row][col], sigma_c)
#             # gs = gaussian(distance(k, l, row, col), sigma_s)
            
#             # # w = gc * gs
#             # if gaussian_c:
#             #     w = gc * gs
#             # else:
#             #     w = gs
#             temp_row = 1
#             temp_col = 1
            
#             k = row + temp_row
#             l = col + temp_col
            
#             term1 = ((row - k) ** 2 + (col - l) ** 2) / (2 * sigma_c ** 2) # denoise by spatial parameter
#             term2 = (image[row][col] - temp_image[k][l]) ** 2 / (2 * sigma_s ** 2) # feature preserving by range parameter
            
#             w = np.exp(-term1 - term2)
                
#             i_filtered += temp_image[k][l] * w
#             Wp += w
    
#     i_filtered = i_filtered / Wp
    
#     return i_filtered

def apply_bilateral_filter(image, window_centered, row, col, kernel_size, sigma_c, sigma_s, gaussian_c = True):
    radius = kernel_size // 2
    i_filtered = 0
    Wp = 0
    
    for k in range(-radius, radius + 1):
        for l in range(-radius, radius + 1):
            # print(f'{k + radius}_{l + radius}')
            
            # gc = gaussian(temp_image[k][l] - temp_image[row][col], sigma_c)
            # gs = gaussian(distance(k, l, row, col), sigma_s)
            
            # # w = gc * gs
            # if gaussian_c:
            #     w = gc * gs
            # else:
            #     w = gs
            
            # term1 = ((row - k) ** 2 + (col - l) ** 2) / (2 * sigma_c ** 2) # denoise by spatial parameter
            term1 = (k ** 2 + l ** 2) / (2 * sigma_c ** 2) # denoise by spatial parameter
            term2 = (image[row][col] - window_centered[k + radius][l + radius]) ** 2 / (2 * sigma_s ** 2) # feature preserving by range parameter
            if gaussian_c == True:
                w = np.exp(-term1 - term2)
            else:
                w = np.exp(-term2)
                
            i_filtered += window_centered[k][l] * w
            Wp += w
    
    i_filtered = i_filtered / Wp
    
    return i_filtered

def Bilateral_filter(image, kernel_size = 3, sigma_c = 3, sigma_s = 3, gaussian_c = True): # image = noisy_image
    new_image = np.zeros(image.shape)
    
    height, width = image.shape
    stride = 1
    padding_size = np.floor((height * (stride - 1) - stride + kernel_size) / 2).astype(int)
    
    temp_image = np.zeros([height + 2 * padding_size, height + 2 * padding_size])
    temp_image[padding_size : -padding_size, padding_size : -padding_size] = image
    
    ## edge padding
    temp_image[padding_size : -padding_size, 0:padding_size] = image[:, 0:padding_size]
    temp_image[padding_size : -padding_size, -padding_size:] = image[:, -padding_size:]
    temp_image[:padding_size, : ] = temp_image[padding_size : padding_size*2, : ]
    temp_image[-padding_size:, : ] = temp_image[-padding_size*2 : -padding_size, : ]

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            
            window_centered = temp_image[row : row + kernel_size, col : col + kernel_size]
            
            # new_image[row, col] = apply_bilateral_filter(image, temp_image, row, col, kernel_size, sigma_c, sigma_s, gaussian_c)
            new_image[row, col] = apply_bilateral_filter(image, window_centered, row, col, kernel_size, sigma_c, sigma_s, gaussian_c)
    
    new_image = new_image.astype('uint8')
    
    return new_image

def make_kernel(f):
    kernel = np.zeros((2 * f + 1, 2 * f + 1), np.float32)
    for d in range(1, f + 1):
        kernel[f - d : f + d + 1, f - d : f + d + 1] += (1.0 / ((2 * d + 1) ** 2))

    return kernel/kernel.sum()

def NLmeans_filter(src, f, t, h):
    '''
    Parameters
    ----------
    src : noisy image.
    f : 相似窗口的半径.
    t : 搜索窗口的半径.
    h : 高斯函数平滑参数(一般取为相似窗口的大小).

    Returns
    -------
    out : new image.

    '''
    
    H, W = src.shape
    out = np.zeros((H, W), np.uint8)
    pad_length = f+t
    src_padding = np.pad(src, (pad_length, pad_length), mode='symmetric').astype(np.float32)
    kernel = make_kernel(f)
    h2 = h*h

    for i in tqdm(range(0, H)):
        for j in range(0, W):
            i1 = i + f + t
            j1 = j + f + t
            W1 = src_padding[i1-f:i1+f+1, j1-f:j1+f+1] # 领域窗口W1
            w_max = 0
            aver = 0
            weight_sum = 0
            # 搜索窗口
            for r in range(i1-t, i1+t+1):
                for c in range(j1-t, j1+t+1):
                    if (r==i1) and (c==j1):
                        continue
                    else:
                        W2 = src_padding[r-f:r+f+1, c-f:c+f+1] # 搜索区域内的相似窗口
                        Dist2 = (kernel*(W1-W2)*(W1-W2)).sum()
                        w = np.exp(-Dist2/h2)
                        if w > w_max:
                            w_max = w
                        weight_sum += w
                        aver += w*src_padding[r, c]
            aver += w_max*src_padding[i1, j1] # 自身领域取最大的权重
            weight_sum += w_max
            out[i, j] = aver/weight_sum

    return out


def NonLocal_Means_filter(image):

    new_image = image
    
    return new_image
    
def improved_NonLocal_Means_filter(image):

    new_image = image
    
    return new_image


# In[] Task 1

path = 'data/'

## task1 i
task1i_imagelist = ['Lena', 'peppers', 'F16']

save_path = 'output/task1'
if not os.path.exists(save_path):
    os.makedirs(save_path)

### save original images
for i_image_name in task1i_imagelist:
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    image = read_image(i_image_path)
    
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(f'original_{i_image_name}.raw')
    plt.savefig(os.path.join(save_path, f'{i_image_name}.png'))
    plt.close()

for i_image_name in task1i_imagelist:
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    image = read_image(i_image_path)
    
    plt.figure(figsize = (16, 12))
    parameters = {'xtick.labelsize': 20,
                  'ytick.labelsize': 20,
                  'axes.titlesize': 40}
    
    plt.subplot(3, 2, 1)
    plt.imshow(image, cmap = 'gray')
    plt.axis('off')
    plt.title(f'Original {i_image_name}.raw')
    
    gamma = 0.25
    
    gamma_trans = gamma_Transform(image.copy(), gamma)
    plt.subplot(3, 2, 3)
    plt.imshow(gamma_trans, cmap='gray')
    plt.axis('off')
    plt.title(f'{i_image_name}.raw with gamma = {gamma}(before HE)')
    
    plt.subplot(3, 2, 4)
    plt.hist(gamma_trans.ravel(), bins = 256, range = (0, 255), color = 'b')
    plt.axis('off')
    plt.title(f'Histrogram of {i_image_name}.raw with gamma = {gamma}(before HE)')
    
    gamma_trans_HE = global_histogram_equalization(gamma_trans)
    plt.subplot(3, 2, 5)
    plt.imshow(gamma_trans_HE, cmap='gray')
    plt.axis('off')
    plt.title(f'{i_image_name}.raw with gamma = {gamma}(after HE)')
        
    plt.subplot(3, 2, 6)
    plt.hist(gamma_trans_HE.ravel(), bins = 256, range = (0, 255), color = 'b')
    plt.axis('off')
    plt.title(f'Histrogram of {i_image_name}.raw with gamma = {gamma}(after HE)')
    
    plt.savefig(os.path.join(save_path, f'{i_image_name}_global_HE.png'))
    plt.close()


## task1 ii
window_list = [3, 5, 7, 33]

for i_image_name in task1i_imagelist:
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    image = read_image(i_image_path)

    plt.figure(figsize = (16, 12))
    parameters = {'xtick.labelsize': 20,
                  'ytick.labelsize': 20,
                  'axes.titlesize': 60}

    for index, i_window_size in enumerate(window_list):
        window_size = [i_window_size, i_window_size]
        local_HE = local_histogram_equalization(image, window_size)
        
        plt.subplot(2, 2, index + 1)
        plt.imshow(local_HE, cmap='gray')
        plt.axis('off')
        plt.title(f'{i_image_name}.raw with window size = {window_size}', fontsize = 12)
        plt.savefig(os.path.join(save_path, f'{i_image_name}_local_HE.png'))
    plt.close()

## task1 iii
ref_image = read_image(os.path.join(path, 'flower.raw'))

plt.imshow(ref_image, cmap='gray')
plt.axis('off')
plt.title('original_flower.raw')
plt.savefig(os.path.join(save_path, 'flower.png'))
plt.close()

plot_hist(ref_image, 'flower.raw')

for i_image_name in task1i_imagelist:
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    src_image = read_image(i_image_path)
    plot_hist(src_image, i_image_name)
    
    image_after_matching = match_histograms(src_image, ref_image)
    plt.imshow(image_after_matching, cmap='gray')
    plt.axis('off')
    plt.title(f'{i_image_name}.raw histogram matching with flower.raw')
    plt.savefig(os.path.join(save_path, f'{i_image_name}_HM.png'))
    plt.close()
    
    plot_hist(image_after_matching, f'{i_image_name}_HM')


# In[] Task 2

save_path_2 = 'output/task2'
if not os.path.exists(save_path_2):
    os.makedirs(save_path_2)

## task1 i
task2i_imagelist = ['Lena', 'peppers', 'F16']

kernel_size_list = [3, 5, 7, 33]

for i_image_name in task2i_imagelist:
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    image = read_image(i_image_path)
    
    for kernel_size in kernel_size_list:
        image_GF = gaussian_filter(image, kernel_size = kernel_size, sigma = 25)
        plt.imshow(image_GF, cmap='gray')
        plt.axis('off')
        plt.title(f'{i_image_name}_kernel size = {kernel_size}')
        plt.savefig(os.path.join(save_path_2, f'{i_image_name}_GF_{kernel_size}.png'))
        plt.close()
        

## task1 ii

for i_image_name in task2i_imagelist:
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    image = read_image(i_image_path)
    
    for kernel_size in kernel_size_list:
        image_AF = averaging_filter(image, kernel_size = kernel_size)
        plt.imshow(image_AF, cmap='gray')
        plt.axis('off')
        plt.title(f'{i_image_name}_kernel size = {kernel_size}')
        plt.savefig(os.path.join(save_path_2, f'{i_image_name}_AF_{kernel_size}.png'))
        plt.close()

## task1 iii
weight = 0.25

for i_image_name in task2i_imagelist:
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    image = read_image(i_image_path)
    
    # for kernel_size in kernel_size_list:
    #     image_GF = gaussian_filter(image, kernel_size = kernel_size, sigma = 25)
    #     image_UmF = image - weight * image_AF
    #     plt.imshow(image_UmF, cmap='gray')
    #     plt.axis('off')
    #     plt.title(f'{i_image_name}_kernel_size = {kernel_size} ')
    #     plt.savefig(os.path.join(save_path_2, f'{i_image_name}_UmF_{kernel_size}.png'))
    #     plt.close()

    image_GF = gaussian_filter(image, kernel_size = 3, sigma = 25)
    image_UmF = image - weight * image_AF
    plt.imshow(image_UmF, cmap='gray')
    plt.axis('off')
    plt.title(f'{i_image_name}_kernel size = {kernel_size}')
    plt.savefig(os.path.join(save_path_2, f'{i_image_name}_UmF_{kernel_size}.png'))
    plt.close()

## task1 iv
kernel_size_list = [3, 5, 7]

for i_image_name in task2i_imagelist:
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    image = read_image(i_image_path)
    
    for kernel_size in kernel_size_list:
        image_LF = laplacian_filter(image, kernel_size = kernel_size)
        plt.imshow(image_LF, cmap='gray')
        plt.axis('off')
        plt.title(f'{i_image_name}_kernel size = {kernel_size}')
        plt.savefig(os.path.join(save_path_2, f'{i_image_name}_LF_{kernel_size}.png'))
        plt.close()

## task1 v

kernel_size_list = [3, 5, 7]

for i_image_name in task2i_imagelist:
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    image = read_image(i_image_path)
    
    for kernel_size in kernel_size_list:
        image_SF = sobel_filter(image, kernel_size = kernel_size)
        plt.imshow(image_SF, cmap='gray')
        plt.axis('off')
        plt.title(f'{i_image_name}_kernel size = {kernel_size}')
        plt.savefig(os.path.join(save_path_2, f'{i_image_name}_SF_{kernel_size}.png'))
        plt.close()

# In[] Task 3

save_path_3 = 'output/task3'
if not os.path.exists(save_path_3):
    os.makedirs(save_path_3)

## task1 i

special_kernel = np.array([[-1, 0, -1],
                           [0, 6, 0],
                           [-1, 0, -1]], dtype="int")

for i_image_name in task2i_imagelist:
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    image = read_image(i_image_path)
    
    image_sk1 = convolve(image, kernel = special_kernel)
    plt.imshow(image_sk1, cmap='gray')
    plt.axis('off')
    plt.title(f'{i_image_name}_special kernel 1')
    plt.savefig(os.path.join(save_path_3, f'{i_image_name}_sk1_{kernel_size}.png'))
    plt.close()

## task1 ii

special_kernel = (1/25) * np.array([[1, 2, 1],
                                    [0, 5, 0],
                                    [4, 2, 4]], dtype="int")

for i_image_name in task2i_imagelist:
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    image = read_image(i_image_path)
    
    image_sk2 = convolve(image, kernel = special_kernel)
    plt.imshow(image_sk2, cmap='gray')
    plt.axis('off')
    plt.title(f'{i_image_name}_special kernel 2')
    plt.savefig(os.path.join(save_path_3, f'{i_image_name}_sk2_{kernel_size}.png'))
    plt.close()


# In[] Task 4
save_path_4 = 'output/task4'
if not os.path.exists(save_path_4):
    os.makedirs(save_path_4)

i_image_path = os.path.join(path, 'Noisy.raw')
noisy_image = read_image(i_image_path)
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')
plt.title('original_Noisy.raw ')
plt.savefig(os.path.join(save_path_4, 'Noisy.png'))
plt.close()

denoise_image = medium_filter(noisy_image)
plt.imshow(denoise_image, cmap='gray')
plt.axis('off')
plt.title('Noisy_kernel size = 3')
plt.savefig(os.path.join(save_path_4, 'Noisy_denoisy_3.png'))
plt.close()


# In[] Task 5
save_path_5 = 'output/task5'
if not os.path.exists(save_path_5):
    os.makedirs(save_path_5)

i_image_path = os.path.join(path, 'Noisy.raw')
noisy_image = read_image(i_image_path)

setting_list = [(3, 3, 3), (3, 9, 3), (3, 27, 3), (3, 27, 9), (3, 27, 27),
                (5, 3, 3), (5, 9, 3), (5, 27, 3), (5, 27, 9), (5, 27, 27),
                (7, 3, 3), (7, 9, 3), (7, 27, 3), (7, 27, 9), (7, 27, 27),
                (11, 3, 3), (11, 9, 3), (11, 27, 3), (11, 27, 9), (11, 27, 27),
                (13, 3, 3), (13, 9, 3), (13, 27, 3), (13, 27, 9), (13, 27, 27),
                ]
gussian_list = [True, False]
for g_set in gussian_list:

    for kernel_size, sigma_c, sigma_s in setting_list:
        print(f'{kernel_size}_{sigma_c}_{sigma_s}')
        noisy_image_BF = Bilateral_filter(noisy_image, kernel_size, sigma_c, sigma_s, gaussian_c = g_set)
        plt.imshow(noisy_image_BF, cmap='gray')
        plt.axis('off')
        plt.title(f'{kernel_size}_{sigma_c}_{sigma_s}')
        plt.savefig(os.path.join(save_path_5, f'Noisy_denoisy_BF_{kernel_size}_{sigma_c}_{sigma_s}_G{g_set}.png'))
        plt.close()




# In[] Task 6
save_path_6 = 'output/task6'
if not os.path.exists(save_path_6):
    os.makedirs(save_path_6)

i_image_path = os.path.join(path, 'Noisy.raw')
noisy_image = read_image(i_image_path)

kernel_size_list = [5, 7]

for kernel_size in kernel_size_list:
    noisy_NLM = NLmeans_filter(noisy_image, kernel_size, kernel_size, 10)
    plt.imshow(noisy_NLM, cmap='gray')
    plt.axis('off')
    plt.title(f'noisy_NLM_kernel size = {kernel_size}')
    plt.savefig(os.path.join(save_path_6, f'noisy_NLM_{kernel_size}.png'))
    plt.close()

task6_imagelist = ['Lena', 'peppers', 'F16']

for i_image_name in task6_imagelist:
    
    i_image_path = os.path.join(path, i_image_name + '.raw')
    image = read_image(i_image_path)
    
    for kernel_size in kernel_size_list:
        noisy_NLM = NLmeans_filter(image, kernel_size, kernel_size, 10)
        plt.imshow(noisy_NLM, cmap='gray')
        plt.axis('off')
        plt.title(f'{i_image_name}_NLM_kernel size = {kernel_size}')
        plt.savefig(os.path.join(save_path_6, f'{i_image_name}_NLM_{kernel_size}.png'))
        plt.close()











# In[] Task 7
save_path_7 = 'output/task7'
if not os.path.exists(save_path_7):
    os.makedirs(save_path_7)




# In[] 



