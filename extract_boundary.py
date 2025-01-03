from PIL import Image,ImageEnhance
import numpy as np
from skimage import measure
import skimage.morphology as morphology
from simplification.cutil import simplify_coords


def get_shoreline(mask_img_path, simplification=1, smoothing=3):
  img_in = Image.open(mask_img_path)

  i_arr = np.array(img_in)
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

  # Simplify the shoreline points using the Visvalingam-Whyatt algorithm
  simplified_shoreline = simplify_coords(shoreline_points, simplification)  # Adjust as needed for desired simplification level

  # Smooth the simplified shoreline using a moving average filter
  window_size = smoothing  # Adjust as needed for desired smoothing level
  smoothed_shoreline = np.convolve(simplified_shoreline[:, 0], np.ones(window_size)/window_size, mode='same')
  smoothed_shoreline = np.stack((smoothed_shoreline, np.convolve(simplified_shoreline[:, 1], np.ones(window_size)/window_size, mode='same')), axis=-1)

  # Remove the first and last points from the smoothed shoreline -- for some reason these are at the centroid of the image
  smoothed_shoreline = smoothed_shoreline[1:-1,:]

  # Append the first point to the end of the smoothed shoreline to close the loop
  smoothed_shoreline = np.vstack((smoothed_shoreline,smoothed_shoreline[0,:]))
  
  #switch the x and y coordinates
  smoothed_shoreline = np.array([smoothed_shoreline[:,1],smoothed_shoreline[:,0]]).T

  #save the shoreline to a csv file
  shoreline_filepath = mask_img_path.replace('.png','_sl.csv')
  np.savetxt(shoreline_filepath, smoothed_shoreline, delimiter=",", fmt="%f")
  
  #append the first point to the end of smooth shoreline
  return smoothed_shoreline,im_ref_buffer_out,shoreline_filepath
