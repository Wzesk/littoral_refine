#%%capture
!pip install simplification==0.7.12
!pip install numpy==1.26.4

%%capture
import os
from PIL import Image,ImageEnhance
import pandas as pd

import io
from io import BytesIO
import tarfile
import numpy as np

from skimage import measure
import skimage.morphology as morphology
from simplification.cutil import simplify_coords
from scipy.spatial import distance




shoreline, buffer = get_shoreline(mask_img)

#flip shoreline axis
flipped_shoreline = np.array([shoreline[:,1],shoreline[:,0]]).T
shoreline_filename = f"{name}_sl.csv"
np.savetxt(contour_folder + "/" + shoreline_filename, flipped_shoreline, delimiter=",", fmt="%f")




def get_shoreline(mask_img):

  if isinstance(mask_img, str):
    img_in = Image.open(mask_img)
    mask_array = np.array(img_in)
  else:
    i_arr = np.array(mask_img)
    i_arr = i_arr[:,:,0]
    i_arr = i_arr.squeeze()
    mask_array = i_arr.squeeze()

  #get the min of the img_in dimensions
  min_dim = min(mask_array.shape)
  dilator = np.floor(min_dim/20)

  # Find contours (boundaries) in the mask
  contours = measure.find_contours(mask_array, 0.5)

  # Assuming you want the longest contour (outer boundary)
  longest_contour = max(contours, key=len)

  # Extract the coordinates of the shoreline points
  shoreline_points = np.array(longest_contour).astype(int).squeeze()

  # create a pixel mask with ja buffer around the longest contour
  shoreline_mask = np.zeros_like(mask_array)
  #set the longest_contour points equal to 1
  shoreline_mask[tuple(np.transpose(longest_contour.astype(np.uint32)))] = 1
  se = morphology.disk(dilator)
  im_ref_boolean = morphology.binary_dilation(shoreline_mask, se)
  #convert im_ref_boolean from boolean values to integer values
  im_ref_buffer_out = im_ref_boolean.astype(np.uint8)
  # # convert to visible image
  #im_ref_img = Image.fromarray(im_ref_buffer.astype(np.uint8)*255)

  # Simplify the shoreline points using the Visvalingam-Whyatt algorithm
  simplified_shoreline = simplify_coords(shoreline_points, 1)  # Adjust 0.01 as needed for desired simplification level

  # Smooth the simplified shoreline using a moving average filter
  window_size = 3  # Adjust as needed for desired smoothing level
  smoothed_shoreline = np.convolve(simplified_shoreline[:, 0], np.ones(window_size)/window_size, mode='same')
  smoothed_shoreline = np.stack((smoothed_shoreline, np.convolve(simplified_shoreline[:, 1], np.ones(window_size)/window_size, mode='same')), axis=-1)

  smoothed_shoreline = smoothed_shoreline[1:-1,:]
  #append the first point to the end of smooth shoreline
  return np.vstack((smoothed_shoreline,smoothed_shoreline[0,:])),im_ref_buffer_out
