from PIL import Image,ImageEnhance
import numpy as np
import os
from skimage import measure
import skimage.morphology as morphology
from simplification.cutil import simplify_coords

from . import tario
import pandas as pd

def get_shorelines_from_folder(folder_path, simplification=1, smoothing=3, periodic=True):
  """Extract shorelines from all mask images in a folder.

  Parameters
  ----------
  folder_path : str
      Path to the folder containing mask images.
  simplification : float, optional
      Simplification factor for the shoreline points, by default 1.
  smoothing : int, optional
      Smoothing window size for the shoreline points, by default 3.
  periodic : bool, optional
      Whether the shoreline is periodic (closed loop), by default True.

  Returns
  -------
  list
      List of tuples containing:
      - np.ndarray : Smoothed shoreline points.
      - np.ndarray : Buffer mask around the shoreline.
      - str : Path to the saved shoreline CSV file.

  Examples
  --------
  >>> shorelines = get_shorelines_from_folder('path/to/mask/folder', simplification=0.5, smoothing=2)
  """
  shorelines = []

  # List all PNG files in the folder
  for filename in os.listdir(folder_path):
    if filename.endswith('.png') and 'mask' in filename:
      mask_img_path = os.path.join(folder_path, filename)

      shoreline, buffer, shoreline_filepath = get_shoreline(mask_img_path, simplification, smoothing, periodic)
      if shoreline is not None:
        shorelines.append(shoreline_filepath)

  return shorelines

def get_shoreline(mask_img_path, simplification=1, smoothing=3,periodic=True):
  """Extract the shoreline from a mask image.

  Parameters
  ----------
  mask_img_path : str
      Path to the mask image file.
  simplification : float, optional
      Simplification factor for the shoreline points, by default 1.
  smoothing : int, optional
      Smoothing window size for the shoreline points, by default 3.
  periodic : bool, optional
      Whether the shoreline is periodic (closed loop), by default True.

  Returns
  -------
  tuple
      - np.ndarray : Smoothed shoreline points.
      - np.ndarray : Buffer mask around the shoreline.
      - str : Path to the saved shoreline CSV file.

  Examples
  --------
  >>> shoreline, buffer, shoreline_filepath = get_shoreline('path/to/mask.png', simplification=0.5, smoothing=2)
  """
  img_in = Image.open(mask_img_path)

  i_arr = np.array(img_in)

  #check shape of the image, if it is 3D, take the first channel
  if len(i_arr.shape) > 2:
    i_arr = i_arr[:,:,0]

  mask_array = i_arr.squeeze()


  #get the min of the img_in dimensions
  min_dim = min(mask_array.shape)
  dilator = np.floor(min_dim/20)

  # Find contours (boundaries) in the mask
  contours = measure.find_contours(mask_array, 0.5)

  # If there are no countours return None and continue
  if len(contours) < 1:
    return None, None, None

  # Assuming you want the longest contour (outer boundary)
  longest_contour = max(contours, key=len)

  # Extract the coordinates of the shoreline points
  shoreline_points = np.array(longest_contour).astype(int).squeeze()

  # create a pixel mask with a buffer around the longest contour
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

  if(periodic):
    # Remove the first and last points from the smoothed shoreline -- for some reason these are at the centroid of the image
    smoothed_shoreline = smoothed_shoreline[1:-1,:]

    # Append the first point to the end of the smoothed shoreline to close the loop
    smoothed_shoreline = np.vstack((smoothed_shoreline,smoothed_shoreline[0,:]))

  #switch the x and y coordinates to match expected format for geotransform
  # Note: Image coordinates are (row, col) but we need (x, y) which is (col, row)
  # This accounts for the coordinate system difference between image and geographic coordinates
  smoothed_shoreline = np.array([smoothed_shoreline[:,1],smoothed_shoreline[:,0]]).T

  #create file paths
  shoreline_filepath = mask_img_path.replace('MASK','SHORELINE')
  shoreline_filepath = shoreline_filepath.replace('mask.png','sl.csv')
  buff_img = Image.fromarray(im_ref_buffer_out*255)
  buffer_path = shoreline_filepath.replace('sl.csv','buffer.png')

  #create directory if it does not exist
  os.makedirs(os.path.dirname(shoreline_filepath), exist_ok=True)

  #save the files
  np.savetxt(shoreline_filepath, smoothed_shoreline, delimiter=",", fmt="%f")
  buff_img.save(buffer_path)

  #append the first point to the end of smooth shoreline
  return smoothed_shoreline,im_ref_buffer_out,shoreline_filepath

def contours_from_tar(path):
  table_path= path + "/proj_track.csv"
  df = pd.read_csv(table_path)
  df['contour'] = False

  tar_path= path+"/mask.tar"

  # get names where upsampled is true
  names = df.loc[df['segmented'] == True]['name'].tolist()
  # contours are in a folder, not a tar
  contour_folder = path+"/countour"
  if not os.path.exists(contour_folder):
    os.makedirs(contour_folder)

  for name in names:
    mask_tar = tario.tar_io(upsample_tar_path)
    img = mask_tar.get_from_tar(name+'_sr.png')
    mask_img = get_from_tar(tar_path, name+'_mask.png')

    shoreline, buffer = get_shoreline(mask_img)

    # #flip shoreline axis
    # flipped_shoreline = np.array([shoreline[:,1],shoreline[:,0]]).T

    shoreline_filename = f"{name}_sl.csv"
    np.savetxt(contour_folder + "/" + shoreline_filename, shoreline, delimiter=",", fmt="%f")

    df.loc[df['name'] == name, 'contour'] = True

  df.to_csv(table_path, index=False)

  return df