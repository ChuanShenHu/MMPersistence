#####################################################################################################
## 2D Image Proccessing
#####################################################################################################

import sys
import numpy as np
import numba as nb
import math
import gudhi
from gudhi.representations import Landscape
import persim
import matplotlib.pyplot as plt
import matplotlib
import cv2
import shutil
import heapq
import os
from os import listdir
from os.path import isfile, isdir, join, splitext
from PIL import Image
import PIL
import imageio.v3 as iio
import pydicom
from pydicom import dcmread
from pydicom.data import get_testdata_files
from typing import Sequence, Tuple

## The following code is for solving the latex problem (New problem in Google Colab 2024)
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

## Import time module
import time

#####################################################################################################
## The following code are designed for image proccessing
#####################################################################################################

## The following normalization code come from https://zhuanlan.zhihu.com/p/372317590
def normalization(img_array, maximal_pixel_value=255):
  # 獲取像素最大/最小值
  max_val = np.max(img_array)
  min_val = np.min(img_array)
  if (max_val == maximal_pixel_value) and (min_val == 0):
    return img_array
  # 像素 Normalization
  img_array = (img_array - min_val) / (max_val - min_val)
  img_tmp = (img_array * maximal_pixel_value).astype(np.uint8)
  return img_tmp

## Given a path of a dcm file, read and normalze it
def read_normal_from_DCM(file_path, maximal_pixel_value=255):
  # Read the DICOM file
  ds = dcmread(file_path)
  # Normalization
  normed_image = normalization(ds.pixel_array,maximal_pixel_value=maximal_pixel_value)
  return normed_image

## Given a path of a dcm file, simply read it
def read_from_DCM(file_path):
  # Read the DICOM file
  ds = dcmread(file_path)
  return ds.pixel_array

## Given a path of a png file, simply read it
def read_from_PNG(file_path):
  # Read the PNG file
  img = iio.imread(file_path)
  return img

## Given a 2D np.array, a dpi, and a color map, plot it
def plot_img(img, dpi=100, cmap='gray'):
  _, axes = plt.subplots(dpi=dpi)
  plt.imshow(img, cmap=cmap)
  plt.show()

## Given a 2D np.array, a dpi, and a color map, plot it without ticks
def plot_img_without_ticks(img, dpi=100, cmap='gray', title='', saveImg = True, filename='tmp.png'):
  plt.rcParams["font.serif"] = "Times New Roman"
  _, axes = plt.subplots(dpi=dpi)
  plt.imshow(img, cmap=cmap)
  axes.set_xticks([])
  axes.set_yticks([])
  axes.set_title(title, fontsize=20)
  plt.show()
  #if saveImg == True:
  #  axes.savefig(filename)

# Basic tool for find indices in a numpy array
def find(condition):
  res = np.nonzero(condition)
  return res

# Thresholding operation (smaller, the typical one) on an image
def biImg_by_threshold_leq(img, threshold):
  output_img = np.copy(img)
  idxs_0 = find(img <= threshold)
  idxs_1 = find(img > threshold)
  output_img[idxs_0] = 0
  output_img[idxs_1] = 1
  return output_img

# Thresholding operation (greater) on an image
def biImg_by_threshold_geq(img, threshold):
  output_img = np.copy(img)
  idxs_0 = find(img >= threshold)
  idxs_1 = find(img < threshold)
  output_img[idxs_0] = 0
  output_img[idxs_1] = 1
  return output_img

# Thresholding operation on an image via range
# For example, if thresholds=[a,b], output the binary image by the following definition:
# New_img[i,j] = 0, if a <= img[i,j] <= b
# New_img[i,j] = 1, for otherwise
def biImg_by_range(img, thresholds=[0,255]):
   input_np_array_shape = np.shape(img)
   ## Declare the output np array...
   output_np_array = np.ones(input_np_array_shape)
   ## For the (i,j)-entry, update the pixel value...
   for i in range(input_np_array_shape[0]):
     for j in range(input_np_array_shape[1]):
       if (img[i,j] >= thresholds[0]) and (img[i,j] <= thresholds[1]):
         output_np_array[i,j] = 0
   ##
   return output_np_array

# Given a binary image, 0 -> 1 and 1 -> 0
def biImg_by_complement(img):
  image_shape = np.shape(img)
  output_img = np.ones(image_shape)
  output_img = output_img - img
  return output_img

#####################################################################################################
## 2D Image Mathematical Morphology Tools
#####################################################################################################
@nb.jit()
def erosion(input_np_array,
            input_list_of_points,
            minimal_pixel_value=0):
  ## Find the shape of the input np array...
  input_np_array_shape = np.shape(input_np_array)
  ## Declare the output np array...
  output_np_array = np.zeros(input_np_array_shape)
  ## For the (i,j)-entry, update the pixel value...
  for i in range(input_np_array_shape[0]):
    for j in range(input_np_array_shape[1]):
      if input_np_array[i,j] == minimal_pixel_value:
        output_np_array[i,j] = minimal_pixel_value
        continue
      ## Find of the collection of relevant pixel values...
      relevant_pixel_values = []
      for k in range(len(input_list_of_points)):
        ##
        m = i - input_list_of_points[k][1]
        n = j + input_list_of_points[k][0]
        ##
        if (m >= 0) and (m < input_np_array_shape[0]) and (n >= 0) and (n < input_np_array_shape[1]):
           relevant_pixel_values.append(input_np_array[m,n])
      ##
      output_np_array[i,j] = min(relevant_pixel_values)
  ## Return the result...
  return output_np_array

@nb.jit()
def dilation(input_np_array,
             input_list_of_points,
             maximal_pixel_value=1):
   ## Find the shape of the input np array...
   input_np_array_shape = np.shape(input_np_array)
   ## Declare the output np array...
   output_np_array = np.zeros(input_np_array_shape)
   ## For the (i,j)-entry, update the pixel value...
   for i in range(input_np_array_shape[0]):
     for j in range(input_np_array_shape[1]):
       if input_np_array[i,j] == maximal_pixel_value:
         output_np_array[i,j] = maximal_pixel_value
         continue
       ## Find of the collection of relevant pixel values...
       relevant_pixel_values = []
       for k in range(len(input_list_of_points)):
         ##
         m = i + input_list_of_points[k][1]
         n = j - input_list_of_points[k][0]
         ##
         if (m >= 0) and (m < input_np_array_shape[0]) and (n >= 0) and (n < input_np_array_shape[1]):
            relevant_pixel_values.append(input_np_array[m,n])
       ##
       output_np_array[i,j] = max(relevant_pixel_values)
   ## Return the result...
   return output_np_array

def opening(input_np_array,
            input_list_of_points):
  return dilation(erosion(input_np_array, input_list_of_points), input_list_of_points)

def closing(input_np_array,
            input_list_of_points):
  return erosion(dilation(input_np_array, input_list_of_points), input_list_of_points)

## Given a 2D np array, view it rectangle. Regard the nearest left coner point as the origin, and output the coordinates as a list of 1D np arrays...
@nb.jit()
def get_rectangle_coordinates(input_np_array):
  ## Find the shape of the input np array...
  input_np_array_shape = np.shape(input_np_array)
  ## Declare the output np array...
  output_list = []
  ## output_list.append(np.array([0,0]))
  ## For the (i,j)-entry, update the pixel value...
  ## Get the (i,j)-indices of the origin...
  origin_i = int(input_np_array_shape[0]/2)
  origin_j = int(input_np_array_shape[1]/2)
  ## For the (i,j)-entry, update the coordinates...
  for i in range(input_np_array_shape[0]):
    for j in range(input_np_array_shape[1]):
      ## The x-value of the (i,j) points is origin_j - j
      ## The y-value of the (i,j) points is origin_i - i
      output_list.append(np.array([origin_j-j, origin_i-i]))
  ## Return the result...
  return output_list

#####################################################################################################
## Persistent homology computation: The following code are designed for computing the PD of 2D images (cubical Gudhi)
#####################################################################################################

# Given a 2d np.array, we output the 1d-arrray for Gudhi.
def img_to_1d_array(img):
  result = []
  img_shape = np.shape(img)
  for i in range(img_shape[0]):
    result = list(img[i,:]) + result
  return np.array(result)

#
def persistence_of_img(img, do_plot=False, plotdpi=500):
  img_shape = np.shape(img)
  nx = img_shape[1]
  ny = img_shape[0]
  filt_values = img_to_1d_array(img)
  cubical_complex = gudhi.CubicalComplex(dimensions = [nx ,ny], top_dimensional_cells = filt_values)
  cubical_complex.compute_persistence()
  persistence_0 = cubical_complex.persistence_intervals_in_dimension(0)
  persistence_1 = cubical_complex.persistence_intervals_in_dimension(1)
  if (do_plot == True):
    plt.figure(dpi=plotdpi)
    persim.plot_diagrams(list([persistence_0, persistence_1]), title="Persistence Diagram")
    plt.show()
  return [persistence_0, persistence_1]

#####################################################################################################
## Mathematical Morphology-based PH computation: Filtrations induced by Erosion, Dilation,Opening, and Closing
#####################################################################################################

# Generate the kernel list of horizontal SEs...
def get_horizontal_SE_list(maximal_SE_lengths):
  kernel_list = []
  for i in range(2, maximal_SE_lengths+1):
    buffer_list = []
    buffer_list.append(np.array([0,0]))
    counter = int(1)
    left_counter = int(-1)
    right_counter = int(1)
    while len(buffer_list) < i:
      ## If counter is an odd number:
      if counter % 2 == 1:
        buffer_list.append(np.array([right_counter,0]))
        counter = counter + 1
        right_counter = right_counter + 1
      ## If counter is an even number:
      else:
        buffer_list.append(np.array([left_counter,0]))
        counter = counter + 1
        left_counter = left_counter - 1
    kernel_list.append(buffer_list)
    ##
  return kernel_list

# Generate the kernel list of vertical SEs...
def get_vertical_SE_list(maximal_SE_lengths):
  kernel_list = []
  for i in range(2, maximal_SE_lengths+1):
    buffer_list = []
    buffer_list.append(np.array([0,0]))
    counter = int(1)
    left_counter = int(-1)
    right_counter = int(1)
    while len(buffer_list) < i:
      ## If counter is an odd number:
      if counter % 2 == 1:
        buffer_list.append(np.array([0, right_counter]))
        counter = counter + 1
        right_counter = right_counter + 1
      ## If counter is an even number:
      else:
        buffer_list.append(np.array([0, left_counter]))
        counter = counter + 1
        left_counter = left_counter - 1
    kernel_list.append(buffer_list)
    ##
  return kernel_list

# Generate the kernel list of square SEs...
def get_square_SE_list(maximal_SE_lengths):
  kernel_list = []
  for i in range(2, maximal_SE_lengths+1):
    kernel_list.append(get_rectangle_coordinates(input_np_array=np.zeros((i,i))))
    ##
  return kernel_list

# For example, the kernel_list can have the following form...
# kernel_list = [[np.array([0,0]), np.array([1,0])], [np.array([-1,0]), np.array([0,0]), np.array([1,0]), np.array([2,0])]]
def plot_morph_filtration(img, kernel_list,
                          morph_type='opening',
                          plotdpi=500,
                          cmap='gray'):
  img_shape = np.shape(img)
  img_buff = np.zeros(img_shape)
  img_buff = img_buff + img
  plot_img_without_ticks(img_buff, dpi=plotdpi, cmap=cmap)
  ## Accumulate the morphological images
  for the_kernel in kernel_list:
    if morph_type == 'opening':
      #morphed_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, the_kernel)
      morphed_img = opening(input_np_array=img, input_list_of_points=the_kernel)
    elif morph_type == 'closing':
      #morphed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, the_kernel)
      morphed_img = closing(input_np_array=img, input_list_of_points=the_kernel)
    elif morph_type == 'erosion':
      #morphed_img = cv2.morphologyEx(img, cv2.MORPH_ERODE, the_kernel)
      morphed_img = erosion(input_np_array=img, input_list_of_points=the_kernel)
    elif morph_type == 'dilation':
      #morphed_img = cv2.morphologyEx(img, cv2.MORPH_DILATE, the_kernel)
      morphed_img = dilation(input_np_array=img, input_list_of_points=the_kernel)
    else:
      break
    plot_img_without_ticks(morphed_img, dpi=plotdpi, cmap=cmap)

# For example, the kernel_list can have the following form...
# kernel_list = [[np.array([0,0]), np.array([1,0])], [np.array([-1,0]), np.array([0,0]), np.array([1,0]), np.array([2,0])]]
def persistence_of_morph_filtration(img, kernel_list,
                                    morph_type='opening',
                                    do_step_plot=False,
                                    do_PD_plot=False,
                                    plotdpi=500,
                                    cmap='gray'):
  img_shape = np.shape(img)
  img_buff = np.zeros(img_shape)
  img_buff = img_buff + img
  ## Accumulate the morphological images
  if (do_step_plot == True):
    plot_img_without_ticks(img_buff, dpi=plotdpi, cmap=cmap)
  for the_kernel in kernel_list:
    if morph_type == 'opening':
      morphed_img = opening(input_np_array=img, input_list_of_points=the_kernel)
    elif morph_type == 'closing':
      morphed_img = closing(input_np_array=img, input_list_of_points=the_kernel)
    elif morph_type == 'erosion':
      morphed_img = erosion(input_np_array=img, input_list_of_points=the_kernel)
    elif morph_type == 'dilation':
      morphed_img = dilation(input_np_array=img, input_list_of_points=the_kernel)
    else:
      break
    ##
    img_buff = img_buff + morphed_img
    if (do_step_plot == True):
      plot_img_without_ticks(morphed_img, dpi=plotdpi, cmap=cmap)
  ## Finally addition
  ##
  if (do_step_plot == True):
    plot_img_without_ticks(img_buff, dpi=plotdpi, cmap=cmap)
  ## Compute the persistence diagrams
  PDs = persistence_of_img(img_buff, do_PD_plot, plotdpi=plotdpi)
  ##
  return PDs

#####################################################################################################
## Persistent homology-based curves
#####################################################################################################

def mid_of_intervals(input_PD):
  result_list = []
  ##
  for i in range(len(input_PD)):
    the_mid = (input_PD[i][0] + input_PD[i][1])/2
    result_list.append(the_mid)
  ##
  return result_list

def lifespan_of_intervals(input_PD):
  result_list = []
  ##
  for i in range(len(input_PD)):
    the_mid = input_PD[i][1] - input_PD[i][0]
    result_list.append(the_mid)
  ##
  return result_list

def deaths_of_intervals(input_PD):
  result_list = []
  ##
  for i in range(len(input_PD)):
    the_d = input_PD[i][1]
    result_list.append(the_d)
  ##
  return result_list

def gen_hitogram(input_list,
                 histogram_bins=np.linspace(0,100,10),
                 plot_the_histogram=False):
  ## Step 1: Clean data
  arr = np.array(input_list, dtype=float)
  arr = arr[np.isfinite(arr)]  # drop NaN/Inf
  ## Step 2: Generate the histogram
  histogram = np.histogram(arr, bins=histogram_bins)
  ## Step 3 (optional): Plot the histogram
  if plot_the_histogram == True:
    plt.hist(arr, bins=histogram_bins)
    plt.title("Histogram")
    plt.show()
  ## Step 4: Return the result
  return histogram

#####################################################################################################
## Tool 1: Gen MMPH histograms...
#####################################################################################################

def gen_mid_histogram(input_binary_image,
                      SE_list,
                      morph_type='opening',
                      histogram_Betti=1,
                      histogram_bins=np.linspace(0,100,10),
                      plot_the_histogram=False):
  ## Step 1: Compute the MMPD
  PD = persistence_of_morph_filtration(input_binary_image,
                                       kernel_list=SE_list,
                                       morph_type=morph_type,
                                       do_step_plot=False,
                                       do_PD_plot=False)
  ## Step 2: Generate the mid histogram
  mid_list = mid_of_intervals(PD[histogram_Betti])
  histogram = gen_hitogram(input_list=mid_list,
                           histogram_bins=histogram_bins,
                           plot_the_histogram=plot_the_histogram)
  ##
  return histogram

#####################################################################################################
## Tool 2: Heatmap generation for a binary image...
#####################################################################################################
# range_of_rows: [a, b] means a <= i < b
# range_of_columns: [c, d] means c <= j < d
# dimension_of_holes = 0 or 1
def num_of_interval_2_3_of_region(img,
                                  range_of_rows,
                                  range_of_columns,
                                  dimension_of_holes=0):
  img_shape = np.shape(img)
  # Construct img_1
  img_1 = np.ones(img_shape)
  img_1[range_of_rows[0]:range_of_rows[1], range_of_columns[0]:range_of_columns[1]] = img[range_of_rows[0]:range_of_rows[1], range_of_columns[0]:range_of_columns[1]]
  img_1[range_of_rows[0], range_of_columns[0]: range_of_columns[1]] = 1
  img_1[range_of_rows[1] - 1, range_of_columns[0]: range_of_columns[1]] = 1
  img_1[range_of_rows[0] : range_of_rows[1], range_of_columns[0]] = 1
  img_1[range_of_rows[0] : range_of_rows[1], range_of_columns[1] - 1] = 1
  # Construct img_2
  img_2 = np.copy(img)
  img_2[range_of_rows[0], range_of_columns[0]: range_of_columns[1]] = 1
  img_2[range_of_rows[1] - 1, range_of_columns[0]: range_of_columns[1]] = 1
  img_2[range_of_rows[0] : range_of_rows[1], range_of_columns[0]] = 1
  img_2[range_of_rows[0] : range_of_rows[1], range_of_columns[1] - 1] = 1
  # Construct img_3
  img_3 = 1 + img_1 + img_2 + img
  #plt.imshow(img_3, cmap='gray')
  #plt.show()
  persistence = persistence_of_img(img_3)
  #print(persistence[dimension_of_holes])
  result = 0
  for element in persistence[dimension_of_holes]:
    if (element[0] == 2) and (element[1] == 3):
      result = result + 1
  #print(persistence[1])
  #plt.imshow(img_1, cmap='gray')
  #plt.show()
  #plt.imshow(img_2, cmap='gray')
  #plt.show()
  #plt.imshow(img, cmap='gray')
  #plt.show()
  return result

#
def num_of_birth_1_3_of_region(img,
                               range_of_rows,
                               range_of_columns,
                               dimension_of_holes=1):
  img_shape = np.shape(img)
  # Construct img_1
  img_1 = np.ones(img_shape)
  img_1[range_of_rows[0]:range_of_rows[1], range_of_columns[0]:range_of_columns[1]] = img[range_of_rows[0]:range_of_rows[1], range_of_columns[0]:range_of_columns[1]]
  img_1[range_of_rows[0], range_of_columns[0]: range_of_columns[1]] = 1
  img_1[range_of_rows[1] - 1, range_of_columns[0]: range_of_columns[1]] = 1
  img_1[range_of_rows[0] : range_of_rows[1], range_of_columns[0]] = 1
  img_1[range_of_rows[0] : range_of_rows[1], range_of_columns[1] - 1] = 1
  # Construct img_2
  img_2 = np.copy(img)
  img_2[range_of_rows[0], range_of_columns[0]: range_of_columns[1]] = 1
  img_2[range_of_rows[1] - 1, range_of_columns[0]: range_of_columns[1]] = 1
  img_2[range_of_rows[0] : range_of_rows[1], range_of_columns[0]] = 1
  img_2[range_of_rows[0] : range_of_rows[1], range_of_columns[1] - 1] = 1
  # Construct img_3
  img_3 = 1 + img_1 + img_2 + img
  #plt.imshow(img_3, cmap='gray')
  #plt.show()
  persistence = persistence_of_img(img_3)
  #print(persistence[dimension_of_holes])
  result = 0
  for element in persistence[dimension_of_holes]:
    if (element[0] == 3) or (element[0] == 1):
      result = result + 1
  #print(persistence[1])
  #plt.imshow(img_1, cmap='gray')
  #plt.show()
  #plt.imshow(img_2, cmap='gray')
  #plt.show()
  #plt.imshow(img, cmap='gray')
  #plt.show()
  return result

# For example, dividing_numbers=[5,10], i.e., dividing the image as 5x5 and 10x10 blocks.
def heatmap_of_local_merging_numbers(img, dividing_numbers=[5]):
  ## 開始測量
  start = time.time()
  img_shape = np.shape(img)
  result = np.zeros(img_shape)
  ## Do for-loop
  for dividing_number in dividing_numbers:
    for i in range(int(np.floor(img_shape[0]/dividing_number))):
      for j in range(int(np.floor(img_shape[1]/dividing_number))):
        # For rows:
        range_of_rows = [i * dividing_number, (i + 1) * dividing_number]
        range_of_columns = [j * dividing_number, (j + 1) * dividing_number]
        buff_img = np.zeros(img_shape)
        buff_img[range_of_rows[0]:range_of_rows[1], range_of_columns[0]:range_of_columns[1]] = num_of_interval_2_3_of_region(img, range_of_rows, range_of_columns, 0)
        result = result + buff_img
  ## 結束測量
  end = time.time()
  ## 輸出結果
  print("執行時間：%f 秒" % (end - start))
  return result

# For example, dividing_numbers=[5,10], i.e., dividing the image as 5x5 and 10x10 blocks.
def heatmap_of_loop_regions(img, dividing_numbers=[5]):
  ## 開始測量
  start = time.time()
  img_shape = np.shape(img)
  result = np.zeros(img_shape)
  ## Do for-loop
  for dividing_number in dividing_numbers:
    for i in range(int(np.floor(img_shape[0]/dividing_number))):
      for j in range(int(np.floor(img_shape[1]/dividing_number))):
        # For rows:
        range_of_rows = [i * dividing_number, (i + 1) * dividing_number]
        range_of_columns = [j * dividing_number, (j + 1) * dividing_number]
        buff_img = np.zeros(img_shape)
        buff_img[range_of_rows[0]:range_of_rows[1], range_of_columns[0]:range_of_columns[1]] = num_of_birth_1_3_of_region(img, range_of_rows, range_of_columns, 1)
        result = result + buff_img
  ## 結束測量
  end = time.time()
  ## 輸出結果
  print("執行時間：%f 秒" % (end - start))
  return result

#####################################################################################################
## PD normalization
#####################################################################################################

def PD_noramlization(input_PD,
                     original_min_value=0,
                     original_max_value=255):
  output_PD = []
  for i in range(len(input_PD)):
    output_PD.append((input_PD[i]-original_min_value)/(original_max_value - original_min_value))
  return output_PD
