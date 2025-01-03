########!pip install geomdl

#  from PIL import Image
# import json
# import numpy as np

# import math
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# import os
# import csv
# import numpy as np
# from datetime import datetime, timedelta
# import pandas as pd

# from PIL import Image,ImageEnhance

# import geomdl
# from geomdl import fitting

# import io
# from io import BytesIO
# import tarfile

# from sklearn.cluster import KMeans





class boundary_transform:
    def __init__(self):
        return self
    

    def do_something(self):

        return self
    






def contour_properties(points):
    """
    Calculates the ratio of the area enclosed by a closed polyline to its perimeter. to determine if it is a value contour

    Args:
        points: A list of (x, y) tuples representing the vertices of the polyline.

    Returns:
        The ratio of the area to the perimeter, or None if the input is invalid.
    """
    if len(points) < 3:
        return None  # Need at least 3 points for a closed shape

    # Close the polyline if it isn't already
    if not np.all(points[0] == points[-1]):
        points.append(points[0])

    # Calculate the area using the shoelace formula
    area = 0
    for i in range(len(points) - 1):
        area += (points[i][0] * points[i+1][1] - points[i+1][0] * points[i][1])
    area = abs(area) / 2

    # Calculate the perimeter
    perimeter = 0
    for i in range(len(points) - 1):
        perimeter += math.dist(points[i], points[i+1])

    # Return the ratio
    if perimeter == 0:
        return 0  # Avoid division by zero
    return area, perimeter

################################################################################


def get_normal_vector_along_nurbs(crv,delta):
    """
    Get the normal vectors along a nurbs curve

    Arguments:
    -----------
    crv: bezier curve
    delta: float
        spacing for the normal vectors

    Returns:
    -----------
    points: list
        x and y coordinates of points along crv
    normal_vectors: list
        x and y coordinates of the normal vector
    """
    crv.delta = delta
    curve_points = crv.evalpts

    #create list numbers from 0 to 1 with 0.01 increments
    t_values = np.arange(0,1,delta)

    normals = []
    #get first ad second derivatives
    for i in range(len(t_values)):
        ders = crv.derivatives(u=t_values[i], order=1)
        #add first dir (position) to second dir (direction) to visualize tangent vector
        tan_vec = np.array([ders[1][0],ders[1][1]])  # tangent vector at u = 0.35
        #normalize vector length
        tan_vector = tan_vec/np.linalg.norm(tan_vec)

        #get angle of tangent vector
        angle = np.arctan2(tan_vec[1], tan_vec[0])
        #get the perpendicular angle
        perpendicular_angle = angle + np.pi/2
        #get the perpendicular vector
        perpendicular_vector = np.array([np.cos(perpendicular_angle), np.sin(perpendicular_angle)])
        #normalize vector length
        normal_vector = perpendicular_vector/np.linalg.norm(perpendicular_vector)

        #append to list
        normals.append(normal_vector)

    return curve_points,normals

def get_planes_along_nurbs(crv,delta):
    """
    Get planes along a nurbs curve

    Arguments:
    -----------
    crv: bezier curve
    delta: float
        spacing for the normal vectors

    Returns:
    -----------
    points: list
        x and y coordinates of points along crv
    normal_vectors: list
        x and y coordinates of the normal vector
    """
    crv.delta = delta
    curve_points = crv.evalpts

    #create list numbers from 0 to 1 with 0.01 increments
    t_values = np.arange(0,1,delta)

    planes = []
    #get first ad second derivatives
    for i in range(len(t_values)):
        ders = crv.derivatives(u=t_values[i], order=1)
        #add first dir (position) to second dir (direction) to visualize tangent vector
        tan_vec = np.array([ders[1][0],ders[1][1]])  # tangent vector at u = 0.35
        #normalize vector length
        tan_vector = tan_vec/np.linalg.norm(tan_vec)

        #get angle of tangent vector
        angle = np.arctan2(tan_vec[1], tan_vec[0])
        #get the perpendicular angle
        perpendicular_angle = angle + np.pi/2
        #get the perpendicular vector
        perpendicular_vector = np.array([np.cos(perpendicular_angle), np.sin(perpendicular_angle)])
        #normalize vector length
        normal_vector = perpendicular_vector/np.linalg.norm(perpendicular_vector)

        #append to list
        plane = dict([])
        plane['origin'] = curve_points[i]
        plane['xvec'] = tan_vector
        plane['yvec'] = normal_vector
        planes.append(plane)
    return planes

def get_plane_along_nurbs(crv,t_val):
    """
    Gets a plane along a nurbs curve

    Arguments:
    -----------
    crv: bezier curve
    delta: float
        spacing for the normal vectors

    Returns:
    -----------
    point: array
        x and y coordinates of points along crv
    plane: dict
        origin x and y coordinates, x and y vectors
    """
    curve_pt = crv.evaluate_single(t_val)


    ders = crv.derivatives(u=t_val, order=1)

    tan_vec = np.array([ders[1][0],ders[1][1]])  # tangent vector at u = 0.35
    #normalize vector length
    tan_vector = tan_vec/np.linalg.norm(tan_vec)

    #get angle of tangent vector
    angle = np.arctan2(tan_vec[1], tan_vec[0])
    #get the perpendicular angle
    perpendicular_angle = angle + np.pi/2
    #get the perpendicular vector
    perpendicular_vector = np.array([np.cos(perpendicular_angle), np.sin(perpendicular_angle)])
    #normalize vector length
    normal_vector = perpendicular_vector/np.linalg.norm(perpendicular_vector)

    #create plane
    plane = dict([])
    plane['origin'] = curve_pt
    plane['xvec'] = tan_vector
    plane['yvec'] = normal_vector
    return plane

def fit_nurbs(shoreline,degree=3,size=24,periodic=False):
    #if periodic, set the last point equal to the first point
    if periodic:
        shoreline[len(shoreline)-1] = shoreline[0]

    #convert shoreline into a list of tuples
    pline = [tuple(x) for x in shoreline]

    # Do global curve interpolation
    #crv = fitting.interpolate_curve(pline, degree)
    #crv = fitting.approximate_curve(pline, degree, ctrlpts_size=size, centripetal=False)
    crv = pline
    try:
      crv = fitting.approximate_curve(pline, degree, ctrlpts_size=size, centripetal=False)
    except Exception as e:
      print(str(e))
      crv = fitting.interpolate_curve(pline, 1)
    return crv

def tangent_dict(curve_points,tangents,extension_length):
    tangent_dict = dict([])
    for i in range(len(curve_points)):
        t_name = 'NA'+str(i+1)
        tangent_dict[t_name] = np.array([[ curve_points[i][0]-tangents[i][0], curve_points[i][1]-tangents[i][0] ],
                                         [ curve_points[i][0]+(tangents[i][0]*2), curve_points[i][1]+(tangents[i][0]*2) ]])

    return tangent_dict


def generate_offsets(curve_points,normals,distance):
  offset_out = np.zeros_like(curve_points)
  offset_in = np.zeros_like(curve_points)
  for i in range(len(curve_points)):
      offset_out[i] = curve_points[i] + normals[i]*distance
      offset_in[i] = curve_points[i] + normals[i]*(-distance)
  return offset_out, offset_in

def generate_sample_pts(curve_points,normals,count):
  sample_pts = []
  for i in range(len(curve_points)):
    sample_pt = curve_points[i] + normals[i]*(-count/2)
    t_samples = []
    t_samples.append(sample_pt)
    for j in range((count*2)-1):
      sample_pt = sample_pt + (normals[i]/2)
      t_samples.append(sample_pt)
    sample_pts.append(t_samples)
  return sample_pts


def sample_image(sample_pts,image_array,scale_down=1):
  sampled_pts = []
  for t_pts in sample_pts:
    t_samples = []
    for pt in t_pts:
      try:
        if int(pt[0]/scale_down) < image_array.shape[0] and int(pt[1]/scale_down) < image_array.shape[1]:
          t_samples.append([pt[0],pt[1],image_array[int(pt[0]/scale_down),int(pt[1]/scale_down)]])
        else:
          min_pixel = np.min(image_array.reshape(image_array.shape[0]*image_array.shape[1],3),axis=0)
          t_samples.append([pt[0],pt[1],min_pixel])
      except Exception as e:
        print(str(e))
        t_samples.append([pt[0],pt[1],np.average(image_array)])
    sampled_pts.append(t_samples)
  return sampled_pts

def image_from_tar(tar_path,search_string):
  tar = tarfile.open(tar_path, 'r')
  members = tar.getmembers()
  imgs = []
  for member in members:
    if search_string in member.name:
      img_bytes = BytesIO(tar.extractfile(member.name).read())
      img_lr = Image.open(img_bytes, mode='r').convert('RGB')
      imgs.append(img_lr)
  tar.close()
  return imgs


def cluster_transects(sampled_nir):
  segmentation_transects = []
  boundary_pts = []
  print(len(sampled_nir))
  for t_s in sampled_nir:
    seg_array = np.zeros((len(t_s),3))
    for i in range(len(t_s)):
      seg_array[i][0] = float(i)
      seg_array[i][1] = np.mean(t_s[i][2])
    seg_array[:,0] = seg_array[:,0]/np.max(seg_array[:,0])
    seg_array[:,1] = seg_array[:,1]/np.max(seg_array[:,1])

    #perform kmeans clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(seg_array)
    seg_array[:,2] = kmeans.predict(seg_array)

    #get the point at the cluster boundary
    boundary_val = find_cluster_boundary(seg_array[:,0],seg_array[:,2],t_s)

    segmentation_transects.append(seg_array)
    boundary_pts.append(boundary_val)
  return segmentation_transects,boundary_pts


def cluster_transects_new(sampled_nir):
  segmentation_transects = []
  boundary_pts = []

  top = 3
  seg_slopes = np.zeros((len(sampled_nir),top,3))

  for s in range(len(sampled_nir)):
    t_s = sampled_nir[s]
    seg_array = np.zeros((len(t_s),3))
    for i in range(len(t_s)):
      seg_array[i][0] = float(i)
      seg_array[i][1] = np.mean(t_s[i][2])
    seg_array[:,0] = seg_array[:,0]/np.max(seg_array[:,0])
    seg_array[:,1] = seg_array[:,1]/np.max(seg_array[:,1])
    segmentation_transects.append(seg_array)
    seg_slopes[s] = find_highest_derivatives(seg_array,top=top)

  boundary_pts = rolling_highest_slope(seg_slopes, sampled_nir)

  # #add a to a list at the start to average kmeans.  KMEANS APPROACH DOES NOT WORK AS WELL AS MAX SLOPE
  # segmentation_transects.insert(0,segmentation_transects[len(segmentation_transects)-1])
  # segmentation_transects.append(segmentation_transects[0])
  # #kmeans loop
  # for i in range(1,len(segmentation_transects)-1):
  #   seg_array = segmentation_transects[i]

  #   ex_seg_array = np.concatenate((segmentation_transects[i-1],segmentation_transects[i],segmentation_transects[i+1]), axis=0)

  #   #perform kmeans clustering
  #   kmeans = KMeans(n_clusters=2, random_state=0).fit(seg_array)
  #   predictions = kmeans.predict(seg_array)
  #   seg_array[:,2] = predictions

  #   #get the point at the cluster boundary
  #   boundary_val = find_cluster_boundary(seg_array[:,0],predictions,sampled_nir[i-1])

  #   # segmentation_transects[i] = seg_array
  #   boundary_pts.append(boundary_val)

  return segmentation_transects,boundary_pts


def rolling_highest_slope(seg_slopes, segmentation_transects,wz=3):
  boundary_pts = []

  sl = seg_slopes.shape[0]
  padded_slopes = np.concatenate((seg_slopes[sl-wz:,:,:],seg_slopes[:,:,:],seg_slopes[:wz,:,:]), axis=0)

  for i in range(sl):
    end_index = i + wz*2
    ave = np.mean(padded_slopes[i:end_index,0,1])

    selection_index = int(len(segmentation_transects[i]) * ave)

    boundary_pts.append(segmentation_transects[i][selection_index][:2])
  return boundary_pts

#THIS METHOD IS NOT WORKING AS EXPECTED
def rolling_average_seg_slopes(seg_slopes, segmentation_transects, wz=3):
  rolling_avg = np.zeros((seg_slopes.shape[0],3))
  sl = seg_slopes.shape[0]
  padded_slopes = np.concatenate((seg_slopes[sl-wz:,:,:],seg_slopes[:,:,:],seg_slopes[:wz,:,:]), axis=0)

  for i in range(wz,seg_slopes.shape[0]+ wz):
    # for j in range(seg_slopes.shape[2]):
    start_index = i - wz
    end_index = i + wz
    rolling_avg[i-wz, 0] = np.mean(padded_slopes[start_index:end_index,:,0])
    rolling_avg[i-wz, 1] = np.mean(padded_slopes[start_index:end_index,:,1])
    rolling_avg[i-wz, 2] = np.mean(padded_slopes[start_index:end_index,:,2])

  boundary_pts = []

  for i in range(seg_slopes.shape[0]):
    pts = segmentation_transects[i]
    #get the distance between each seg_slope point and the rolling average
    distances = np.zeros((3))
    for sl in range(seg_slopes.shape[1]):
      distances[sl] = np.linalg.norm(seg_slopes[i,sl,:] - rolling_avg[i,:])

    #find the seg_slope with the shortest distance
    closest_index = np.argmin(distances)
    closest_pt = seg_slopes[i,closest_index,1]

    selection_index = int(closest_pt * len(pts))
    if(selection_index >= len(pts)):
      selection_index = len(pts)-1

    # print(selection_index)
    boundary_pts.append(pts[selection_index][:2])

  return boundary_pts


def find_cluster_boundary(values,labels,pts):

  min_val_0 = np.min(values[labels == 0])
  max_val_0 = np.max(values[labels == 0])
  min_val_1 = np.min(values[labels == 1])
  max_val_1 = np.max(values[labels == 1])

  min = min_val_0
  if min_val_1 > min_val_0:
    min = min_val_1
  max = max_val_0
  if max_val_1 < max_val_0:
    max = max_val_1

  min = min*len(pts)
  max = max*len(pts)
  if(min >= len(pts)):
    min = len(pts)-1
  if(max >= len(pts)):
    max = len(pts)-1

  min_pt = pts[int(min)][:2]
  max_pt = pts[int(max)][:2]
  #average min_pt with max_pt
  ave_pt = np.add(min_pt,max_pt)/2

  return ave_pt



def find_highest_derivatives(points,top=3):
  derivatives = np.zeros(points.shape)

  for i in range(len(points)-1):
    #get slope of segment between point and subsquent point
    derivatives[i,0] = (points[i+1][1]-points[i][1])/(points[i+1][0]-points[i][0])
    derivatives[i,1] = (points[i,0]+points[i+1,0])/2
    derivatives[i,2] = (points[i,1]+points[i+1,1])/2

  #sort derivatives in descending order using first column
  derivatives = derivatives[derivatives[:,0].argsort()]
  #reverse the sorted order
  if (points[0][1]+points[1][1]+points[2][1]) < (points[-1][1]+points[-2][1]+points[-3][1]):
    derivatives = derivatives[::-1]

  return derivatives[:top]


def get_shoreline_csv_paths(folder_path):
  # Get a list of all files in the folder
  files = os.listdir(folder_path)

  # Filter the list
  shoreline_files = [file for file in files if file.endswith('_sl.csv')]

  names = [file.replace('_sl.csv','') for file in shoreline_files]
  # Sort the shoreline_files and names using names as the key
  shoreline_files = [x for _,x in sorted(zip(names,shoreline_files))]
  names = sorted(names)

  return shoreline_files,names

def save_points_as_csv(points, filename):
  with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(points)
    print(filename + " saved.")


def refine_shorelines(shoreline_csv_paths,names, base_path, folder_path):
  table_path= base_path + "/proj_track.csv"
  df = pd.read_csv(table_path)
  df['refined'] = False

  for i in range(len(shoreline_csv_paths)):
    print(names[i])
    img = image_from_tar(base_path+"/upsampled.tar",names[i]+'_sr.png')
    if len(img)> 0:#if there is an image with that name

      #base the sampling on the image size
      min_dim = min(img[0].size)
      sample_size = max(int(min_dim/50),16)
      nurbs_size = sample_size

      #get the shoreline
      shoreline = np.genfromtxt(folder_path + "/" + shoreline_csv_paths[i], delimiter=',')
      flipped_shoreline = np.array([shoreline[:,1],shoreline[:,0]]).T

      # get shoreline properties
      area, perimeter = contour_properties(flipped_shoreline)
      if area < (img[0].size[0] * img[0].size[1] / 20 ): # if the area is less than 5% of the image there must be a problem
        print("contour has area less than 5%")
        continue

      if perimeter > ((img[0].size[0] + img[0].size[1]) * 3 ): # if the length is super long there must be a problem
        print("contour is too long for the image")
        continue

      # generate sample points
      smooth_shoreline = fit_nurbs(flipped_shoreline,size=nurbs_size,periodic=True)
      curve_points, normals = get_normal_vector_along_nurbs(smooth_shoreline,0.005)
      print(sample_size)
      sample_pts = generate_sample_pts(curve_points,normals,sample_size)

      #sample the original image
      img_arr = np.array(img[0])
      sampled_nir = sample_image(sample_pts,img_arr)

      #derive transects
      segmentation_transects,boundary_pts = cluster_transects_new(sampled_nir)
      bd_arr = np.array(boundary_pts)

      filename = folder_path + "/" + names[i] + "_rsl.csv"
      save_points_as_csv(boundary_pts, filename)

      df.loc[df['name'] == names[i], 'refined'] = True
  df.to_csv(table_path, index=False)

  return df

def get_tar_filenames(tar_path):
  tar = tarfile.open(tar_path, 'r')
  names = []
  members = tar.getmembers()
  for member in members:
    names.append(member.name)
  tar.close()
  return names