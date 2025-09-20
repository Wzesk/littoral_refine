from sklearn.cluster import KMeans
from skimage import measure
import skimage.filters as filters
import os
import glob
import pandas as pd
import math
from geomdl import fitting
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import warnings

# Configure matplotlib to suppress font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
warnings.filterwarnings('ignore', message='.*findfont.*')

# Set matplotlib to use any available font without searching
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

from . import boundary_filter



class boundary_refine:
  """Class to refine and visualize shoreline boundaries.

  Attributes
  ----------
  img : PIL.Image
      The original image.
  sample_values : list
      Sampled NIR values.
  shoreline : np.ndarray
      Array of shoreline points.
  refined_boundary : np.ndarray
      Array of refined boundary points.
  """
  def __init__(self,shoreline_path,img_path,delta=0.005,periodic=True):
    """Initialize the boundary_refine class.

    Parameters
    ----------
    shoreline_path : str
        Path to the shoreline CSV file.
    img_path : str
        Path to the image file.
    delta : float, optional
        Delta value for NURBS fitting, by default 0.005.
    periodic : bool, optional
        Whether the shoreline is periodic (closed loop), by default True.
    """
    self.shoreline_path = shoreline_path
    self.refined_filepath = None
    self.periodic = periodic

    #load the image
    self.img = Image.open(img_path)

    #base the sampling on the image size
    min_dim = min(self.img.size)
    self.sample_size = max(int(min_dim/50),16)
    self.nurbs_size = self.sample_size

    #load the shoreline
    self.shoreline = np.genfromtxt(shoreline_path, delimiter=',')

    #if the shoreline is periodic, make sure it is closed
    if self.periodic:
      # Close the polyline if it isn't already
      if not np.all(self.shoreline[0] == self.shoreline[-1]):
          #add the first point to the end of the np array
          self.shoreline = np.vstack((self.shoreline,self.shoreline[0]))
    
    # initialize shoreline properties
    self.area = None
    self.perimeter = None

    # set shoreline properties
    self.boundary_properties()

    # determine if this is a useful boundary after area and perimeter have been set
    self.valid = True
    if self.periodic:
      if self.area < (self.img.size[0] * self.img.size[1] / 20 ): # if the area is less than 5% of the image there must be a problem
        self.valid = False

    if self.perimeter > ((self.img.size[0] + self.img.size[1]) * 3 ): # if the length is super long there must be a problem
      self.valid = False

    #initialize nurbs params
    self.nurbs = None
    self.delta = delta
    self.crv_pts = None
    self.normals = None

    #initialized sampled values
    self.sample_pts = None
    self.sample_values = None

    #kmeans clusters
    self.clustered_samples = None

    #initialize boundary
    self.refined_boundary = None

  def basic_thresholding(self,threshold):
    return self

  def buffer_thresholding(self,threshold,buffer):
    return self
  
  def normal_thresholding(self):
    """Refine the shoreline boundary using normal vectors along a NURBS curve.

    Returns
    -------
    np.ndarray
        Refined boundary points.
    """
    self.fit_nurbs()
    self.calc_normal_vector_along_nurbs()

    sample_pts = self.generate_normal_sample_pts()

    sampled = self.sample_image(sample_pts)

    self.refined_boundary,self.clustered_samples = self.threshold_samples(sampled)

    self.save_refined_shoreline()

    return self.refined_filepath

  def kmeans_thresholding(self):
    """Refine the shoreline boundary using normal vectors along a NURBS curve.

    Returns
    -------
    np.ndarray
        Refined boundary points.
    """
    self.fit_nurbs()
    self.calc_normal_vector_along_nurbs()

    sample_pts = self.generate_normal_sample_pts()

    sampled = self.sample_image(sample_pts)

    self.refined_boundary,self.clustered_samples = self.cluster_transects(sampled)

    self.save_refined_shoreline()

    return self.refined_filepath

  def fit_nurbs(self,degree=3,size=24):
    """Fit a NURBS curve to the shoreline points.

    Parameters
    ----------
    degree : int, optional
        Degree of the NURBS curve, by default 3.
    size : int, optional
        Number of control points for the NURBS curve, by default 24.

    Returns
    -------
    geomdl.NURBS.Curve
        Fitted NURBS curve.
    """
    shoreline = self.shoreline.copy()

    #if periodic, set the last point equal to the first point
    if self.periodic:
        shoreline[len(shoreline)-1] = shoreline[0]

    # check for adjacent duplicate points
    diff = np.diff(shoreline, axis=0)
    duplicate_indices = np.where((diff == 0).all(axis=1))[0] + 1
    if len(duplicate_indices) > 0:
        shoreline = np.delete(shoreline, duplicate_indices, axis=0)
        print(f"Removed {len(duplicate_indices)} duplicate points from shoreline for NURBS fitting.")

    #convert shoreline into a list of tuples
    pline = [tuple(x) for x in shoreline]

    # Adjust control point count to prevent self-overlap
    # For periodic curves, use a higher ratio of control points to data points
    min_control_pts = max(6, degree + 1)  # Minimum control points for given degree
    max_control_pts = min(size, len(pline) - 1)  # Don't exceed data points
    
    # For periodic curves, use more control points to reduce self-overlap
    if self.periodic:
        adjusted_size = max(min_control_pts, min(max_control_pts, len(pline) // 3))
    else:
        adjusted_size = max(min_control_pts, max_control_pts)

    try:
      # Use centripetal parameterization for better curve shape
      self.nurbs = fitting.approximate_curve(pline, degree, ctrlpts_size=adjusted_size, centripetal=True)
    except Exception as e:
      print(f"NURBS approximation failed: {str(e)}")
      try:
        # Fallback: try with fewer control points
        fallback_size = max(min_control_pts, adjusted_size // 2)
        print(f"Retrying with {fallback_size} control points...")
        self.nurbs = fitting.approximate_curve(pline, degree, ctrlpts_size=fallback_size, centripetal=True)
      except Exception as e2:
        print(f"Fallback approximation failed: {str(e2)}")
        # Final fallback: simple interpolation with degree 1
        self.nurbs = fitting.interpolate_curve(pline, 1)

    return self.nurbs
  
  def calc_normal_vector_along_nurbs(self):
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
      crv = self.nurbs
      crv.delta = self.delta
      curve_points = crv.evalpts

      #create list numbers from 0 to 1 with delta size steps
      t_values = np.arange(0,1,self.delta)

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

      self.crv_pts = np.array(curve_points)
      self.normals = normals
      return curve_points,normals

  def boundary_properties(self):
      """
      Calculates the ratio of the area enclosed by a closed polyline to its perimeter. to determine if it is a value contour

      Args:
          points: A list of (x, y) tuples representing the vertices of the polyline.

      Returns:
          The ratio of the area to the perimeter, or None if the input is invalid.
      """
      points = self.shoreline
      if len(points) < 3:
          return None  # Need at least 3 points for a closed shape

      # Calculate the area using the shoelace formula
      area = 0
      if self.periodic:
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
      
      self.area = area
      self.perimeter = perimeter
      return area, perimeter

  def generate_normal_sample_pts(self):
    """Generate sample points along the normal vectors of the NURBS curve.

    Returns
    -------
    list
        List of sample points along the normal vectors.
    """
    sample_pts = []
    curve_points = self.crv_pts
    count = self.sample_size
    normals = self.normals
    
    # Get image dimensions for bounds checking
    img_width, img_height = self.img.size  # PIL image size is (width, height)
  
    for i in range(len(curve_points)):
      # Calculate initial sample point
      sample_pt = curve_points[i] + normals[i]*(-count/2)
      t_samples = []
      
      # Check if vector intersects again with the curve
      max_distance = count  # Maximum sampling distance from curve point
      intersection_distance = self._find_shoreline_intersection(curve_points[i], normals[i], max_distance, nurbs_point_index=i)
      
      if intersection_distance is not None:
        # Clip the sampling range to stop before the intersection
        # But ensure minimum sample size for meaningful analysis
        min_sample_size = max(4, count // 3)  # Minimum 4 samples, or 1/3 of original
        clipped_count = int(intersection_distance * 0.8)  # Stop at 80% of intersection distance
        effective_count = max(min_sample_size, min(count, clipped_count))
        
        # If intersection is very close, use larger safety margin
        if intersection_distance < count * 0.3:
            effective_count = max(min_sample_size, min(count, int(intersection_distance * 0.6)))
      else:
        effective_count = count

      # Generate sample points along the normal vector
      for j in range(effective_count*2):
        current_pt = curve_points[i] + normals[i]*(-effective_count/2 + j/2)
        
        # Clip sample point to image bounds
        clipped_x = max(0, min(img_width - 1, current_pt[0]))
        clipped_y = max(0, min(img_height - 1, current_pt[1]))
        clipped_pt = np.array([clipped_x, clipped_y])
        
        t_samples.append(clipped_pt)

      sample_pts.append(t_samples)

    self.sample_pts = sample_pts

    return sample_pts

  def _find_shoreline_intersection(self, start_point, direction, max_distance, nurbs_point_index=None, min_distance=0.5):
    """
    Find the closest intersection between a normal vector and the fitted NURBS curve.
    
    Parameters
    ----------
    start_point : np.ndarray
        Starting point of the normal vector (point on NURBS curve)
    direction : np.ndarray  
        Direction vector of the normal
    max_distance : float
        Maximum distance to check for intersections
    nurbs_point_index : int, optional
        Index of the NURBS curve point (used for excluding nearby segments)
    min_distance : float
        Minimum distance from start_point to consider valid intersections
        
    Returns
    -------
    float or None
        Distance to closest intersection, or None if no intersection found
    """
    # Create the normal line endpoints
    end_point_pos = start_point + direction * max_distance
    end_point_neg = start_point - direction * max_distance
    
    closest_intersection_dist = None
    
    # Use the fitted NURBS curve points instead of original shoreline
    nurbs_curve = self.crv_pts
    
    # Calculate which NURBS curve segments to exclude based on the originating point
    excluded_segments = set()
    if nurbs_point_index is not None:
        # Exclude the exact segment containing the starting point to avoid the trivial intersection
        # Also exclude immediate neighbors to handle curve smoothness
        total_nurbs_segments = len(nurbs_curve) - 1
        if total_nurbs_segments > 0:
            # Exclude the originating segment and immediate neighbors
            for offset in [-1, 0, 1]:
                seg_idx = (nurbs_point_index + offset) % total_nurbs_segments
                excluded_segments.add(seg_idx)
    
    # Check intersection with each NURBS curve segment
    for j in range(len(nurbs_curve) - 1):
        # Skip segments that are too close to the originating point
        if j in excluded_segments:
            continue
            
        seg_start = nurbs_curve[j]
        seg_end = nurbs_curve[j + 1]
        
        # Check intersection with positive direction
        intersection_pos = self._line_segment_intersection(
            start_point, end_point_pos, seg_start, seg_end
        )
        
        if intersection_pos is not None:
            dist = np.linalg.norm(intersection_pos - start_point)
            if dist > min_distance:  # Ignore intersections too close to starting point
                if closest_intersection_dist is None or dist < closest_intersection_dist:
                    closest_intersection_dist = dist
        
        # Check intersection with negative direction
        intersection_neg = self._line_segment_intersection(
            start_point, end_point_neg, seg_start, seg_end
        )
        
        if intersection_neg is not None:
            dist = np.linalg.norm(intersection_neg - start_point)
            if dist > min_distance:  # Ignore intersections too close to starting point
                if closest_intersection_dist is None or dist < closest_intersection_dist:
                    closest_intersection_dist = dist
    
    return closest_intersection_dist

  def _line_segment_intersection(self, p1, p2, p3, p4):
    """
    Find intersection point between two line segments.
    
    Parameters
    ----------
    p1, p2 : np.ndarray
        Endpoints of first line segment
    p3, p4 : np.ndarray  
        Endpoints of second line segment
        
    Returns
    -------
    np.ndarray or None
        Intersection point, or None if no intersection
    """
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]
    
    # Calculate denominators
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # Lines are parallel
    if abs(denom) < 1e-10:
        return None
    
    # Calculate parameters
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    # Check if intersection is within both line segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        # Calculate intersection point
        intersect_x = x1 + t * (x2 - x1)
        intersect_y = y1 + t * (y2 - y1)
        return np.array([intersect_x, intersect_y])
    
    return None

  def sample_image(self, sample_pts, scale_down=1):
    """Sample the image at the given sample points.

    Parameters
    ----------
    sample_pts : list
        List of sample points.
    scale_down : int, optional
        Scale down factor for the image, by default 1.

    Returns
    -------
    list
        List of sampled points with their corresponding pixel values.
    """
    image_array = np.array(self.img)
    sampled_pts = []
    out_of_bounds_count = 0

    for t_pts in sample_pts:
      t_samples = []
      for pt in t_pts:
        try:
          # Apply scale_down and ensure we're within bounds
          x_idx = int(pt[0]/scale_down)
          y_idx = int(pt[1]/scale_down)
          
          if 0 <= x_idx < image_array.shape[1] and 0 <= y_idx < image_array.shape[0]:
            t_samples.append([pt[0],pt[1],image_array[y_idx, x_idx]])
          else:
            # This should be rare now due to clipping in generate_normal_sample_pts
            out_of_bounds_count += 1
            min_pixel = np.min(image_array.reshape(image_array.shape[0]*image_array.shape[1],image_array.shape[2]),axis=0)
            t_samples.append([pt[0],pt[1],min_pixel])
        except Exception as e:
          out_of_bounds_count += 1
          t_samples.append([pt[0],pt[1],np.average(image_array)])
      sampled_pts.append(t_samples)
    
    # Only print summary if there were out-of-bounds samples
    if out_of_bounds_count > 0:
      print(f"Warning: {out_of_bounds_count} sample points were out of bounds and substituted")
    
    self.sample_values = sampled_pts
    return sampled_pts

  def threshold_samples(self, sampled):
    """Threshold the sampled points to refine the boundary.

    Parameters
    ----------
    sampled : list
        List of sampled points.

    Returns
    -------
    np.ndarray
        Array of refined boundary points.
    """
    segmentation_transects = []
    refined_boundary_pts = []

    top = 3
    seg_slopes = np.zeros((len(sampled),top,3))

    for s in range(len(sampled)):
      t_s = sampled[s]
      seg_array = np.zeros((len(t_s),3))

      for i in range(len(t_s)):
        seg_array[i][0] = float(i)
        seg_array[i][1] = np.mean(t_s[i][2])

      if np.max(seg_array[:,0]) > 0:      
        seg_array[:,0] = seg_array[:,0]/np.max(seg_array[:,0])

      if np.max(seg_array[:,1]) > 0:
        seg_array[:,1] = seg_array[:,1]/np.max(seg_array[:,1])
        
      segmentation_transects.append(seg_array)

      derivatives = self.find_highest_derivatives(seg_array)
      # Handle case where fewer than expected derivatives are returned
      if len(derivatives) < top:
          # Pad with zeros if necessary
          padded_derivatives = np.zeros((top, 3))
          padded_derivatives[:len(derivatives)] = derivatives
          seg_slopes[s] = padded_derivatives
      else:
          seg_slopes[s] = derivatives[:top]

    refined_boundary_pts = np.array(self.rolling_highest_slope(seg_slopes, sampled))

    return refined_boundary_pts,segmentation_transects

  def visualize_results(self,draw_image=False,draw_sampling=False):
    """Visualize the results of the boundary refinement.

    Parameters
    ----------
    draw_image : bool, optional
        Whether to draw the original image, by default False.
    draw_sampling : bool, optional
        Whether to draw the sampled NIR values, by default False.
    """
    plt.axis('equal')
    plt.gca().invert_yaxis()

    self.sample_image
    if draw_image:
        img_arr = np.array(self.img)
        plt.imshow(img_arr) 

    if draw_sampling:
        sampled_nir = self.sample_values
        for t_s in sampled_nir:
            for pt in t_s:
                pixel = np.array([pt[2][0]/255,pt[2][1]/255,pt[2][2]/255])
                plt.plot(pt[0], pt[1], '.',ms=5,color=pixel)

    plt.plot(self.shoreline[:,0],self.shoreline[:,1],color='blue')
    plt.plot(self.refined_boundary[:,0],self.refined_boundary[:,1],color='red')
    # make the background light blue
    plt.gca().set_facecolor((0.8, 0.9, 1.0))
    plt.show()

  def visualize_clusters(self, bounds=slice(None,None)):    
    """
    Visualize the clustering results
    """
    if self.clustered_samples is not None:
      plt.axis('equal')
    
      for test_pts in self.clustered_samples:
        plt.scatter(test_pts[:,0], test_pts[:,1], c=test_pts[:,2])
      plt.show()

  def visualize_max_slopes(self,bounds=slice(None,None)):  
    """
    Visualize location of shoreline derivative
    """
    if self.clustered_samples is not None:
      plt.axis('equal')

      for test_pts in self.clustered_samples:
        dirs = self.find_highest_derivatives(test_pts)
        plt.scatter(test_pts[:,0], test_pts[:,1], c='grey')
        plt.scatter(dirs[:,1],dirs[:,2],c='red')
      plt.show()

  def rolling_highest_slope(self, seg_slopes, segmentation_transects,wz=3):
    """Calculate the rolling highest slope for segmentation.

    Parameters
    ----------
    seg_slopes : np.ndarray
        Array of segment slopes.
    segmentation_transects : list
        List of segmentation transects.
    wz : int, optional
        Window size for calculating the rolling average, by default 3.

    Returns
    -------
    list
        List of refined boundary points.
    """
    boundary_pts = []

    sl = seg_slopes.shape[0]
    padded_slopes = np.concatenate((seg_slopes[sl-wz:,:,:],seg_slopes[:,:,:],seg_slopes[:wz,:,:]), axis=0)

    for i in range(sl):
      end_index = i + wz*2
      ave = np.mean(padded_slopes[i:end_index,0,1])

      selection_index = int(len(segmentation_transects[i]) * ave)

      boundary_pts.append(segmentation_transects[i][selection_index][:2])

    return boundary_pts

  def find_highest_derivatives(self, points,top=3):
    """Find the highest derivatives (slopes) between points.

    Parameters
    ----------
    points : np.ndarray
        Array of points.
    top : int, optional
        Number of top derivatives to return, by default 3.

    Returns
    -------
    np.ndarray
        Array of top derivatives.
    """
    # Handle edge case where there are too few points
    if len(points) < 2:
        # Return a single point with zero derivative if insufficient points
        return np.array([[0.0, points[0][0], points[0][1]]] * min(top, 1))
    
    derivatives = np.zeros(points.shape)

    for i in range(len(points)-1):
      # Avoid division by zero
      dx = points[i+1][0] - points[i][0]
      if abs(dx) < 1e-10:
          derivatives[i,0] = 0.0
      else:
          derivatives[i,0] = (points[i+1][1]-points[i][1])/dx
      derivatives[i,1] = (points[i,0]+points[i+1,0])/2
      derivatives[i,2] = (points[i,1]+points[i+1,1])/2

    #sort derivatives in descending order using first column
    derivatives = derivatives[derivatives[:,0].argsort()]
    
    #reverse the sorted order - but only if we have enough points
    if len(points) >= 3:
        if (points[0][1]+points[1][1]+points[2][1]) < (points[-1][1]+points[-2][1]+points[-3][1]):
            derivatives = derivatives[::-1]
    elif len(points) == 2:
        # For 2 points, use simpler comparison
        if points[0][1] < points[-1][1]:
            derivatives = derivatives[::-1]

    # Return the top derivatives, but don't exceed the number of actual derivatives
    return derivatives[:min(top, len(points)-1)]

  def cluster_transects(self, sampled):
    """Cluster transects using KMeans clustering.

    Parameters
    ----------
    sampled : list
        List of sampled points.

    Returns
    -------
    tuple
        - list : List of refined boundary points.
        - list : List of segmentation transects.
    """
    segmentation_transects = []
    boundary_pts = []

    for t_s in sampled:
      seg_array = np.zeros((len(t_s),3))
      for i in range(len(t_s)):
        seg_array[i][0] = float(i)
        seg_array[i][1] = np.mean(t_s[i][2])
      seg_array[:,0] = seg_array[:,0]/np.max(seg_array[:,0])
      seg_array[:,1] = seg_array[:,1]/np.max(seg_array[:,1])

      #perform kmeans clustering
      kmeans = KMeans(n_clusters=2, random_state=0).fit(seg_array)
      seg_array[:,2] = kmeans.predict(seg_array)
      #the cluster labels are 0 or 1,make sure that the cluster with lower y values is 0
      if np.mean(seg_array[seg_array[:,2] == 0,1]) > np.mean(seg_array[seg_array[:,2] == 1,1]):
        seg_array[:,2] = np.where(seg_array[:,2] == 0,1,0)

      #get the point at the cluster boundary
      boundary_val = self.find_cluster_boundary(seg_array[:,0],seg_array[:,2],t_s)

      segmentation_transects.append(seg_array)
      boundary_pts.append(boundary_val)

    refined_boundary_pts = np.array(boundary_pts)

    return refined_boundary_pts,segmentation_transects

  def find_cluster_boundary(self, values,labels,pts):
    """Find the boundary point between clusters.

    Parameters
    ----------
    values : np.ndarray
        Array of values used for clustering.
    labels : np.ndarray
        Array of cluster labels.
    pts : list
        List of points corresponding to the values.

    Returns
    -------
    np.ndarray
        The average point between the cluster boundaries.
    """

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

  def save_refined_shoreline(self):
    """Save the refined shoreline to a CSV file."""
    self.refined_filepath = self.shoreline_path.replace('_sl','_rl')
    print("Saving refined shoreline to: " + self.refined_filepath)
    np.savetxt(self.refined_filepath, self.refined_boundary, delimiter=",", fmt="%f")


def find_wl_contours2(im_ms, nir, im_labels, im_ref_buffer):
  """Find waterline contours using the provided masks and labels.

  Note:  This functions is copied from the coastsat project, modified only to
  fit within the refiner class structure. It is included as a benchmark of existing refinement methods:
  https://github.com/kvos/CoastSat

  Parameters
  ----------
  im_ms : np.ndarray
      Multispectral image array.
  nir : np.ndarray
      Near-infrared image array.
  im_labels : np.ndarray
      Image mask array.
  im_ref_buffer : np.ndarray
      buffer mask.

  Returns
  -------
  tuple
      - list : Contours of the waterline.
      - float : Threshold value.
      - np.ndarray : Lens array.
  """
  np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

  # ensure labels and buffer are 0-1 by dividing by max value
  im_labels = (im_labels / np.max(im_labels)).astype(bool)
  im_ref_buffer = (im_ref_buffer / np.max(im_ref_buffer)).astype(bool)

  nrows = im_ref_buffer.shape[0]
  ncols = im_ref_buffer.shape[1]

  # calculate Normalized Difference Modified Water Index (NIR - G)
  im_wi = ndwi(nir[:,:,0], im_ms[:,:,1])

  #export index
  lens = im_wi

  # stack indices together --  todo:  removed the second index so stacking the same index twice.  
  im_ind = np.stack((im_wi, im_wi), axis=-1)
  vec_ind = im_ind.reshape(nrows*ncols,2)

  # reshape labels into vectors
  vec_sand = im_labels.reshape(ncols*nrows)
  vec_water = ~im_labels.reshape(ncols*nrows)

  # create a buffer around the sandy beach
  vec_buffer = im_ref_buffer.reshape(nrows*ncols)

  # select water/sand pixels that are within the buffer
  int_water = vec_ind[np.logical_and(vec_buffer,vec_water),:]
  int_sand = vec_ind[np.logical_and(vec_buffer,vec_sand),:]

  # make sure both classes have the same number of pixels before thresholding
  if len(int_water) > 0 and len(int_sand) > 0:
      if np.argmin([int_sand.shape[0],int_water.shape[0]]) == 1:
          int_sand = int_sand[np.random.choice(int_sand.shape[0],int_water.shape[0], replace=False),:]
      else:
          int_water = int_water[np.random.choice(int_water.shape[0],int_sand.shape[0], replace=False),:]

  # threshold the sand/water intensities
  int_all = np.append(int_water,int_sand, axis=0)

  t_wi = filters.threshold_otsu(int_all[:,1])

  # find contour with Marching-Squares algorithm
  im_wi_buffer = np.copy(im_wi)
  im_wi_buffer[~im_ref_buffer] = np.nan

  contours_wi = measure.find_contours(im_wi_buffer, t_wi)


  # remove contour points that are NaNs (around clouds)
  contours_wi = process_contours(contours_wi)

  # only return MNDWI contours and threshold
  return contours_wi, t_wi,lens

def ndwi(im1, im2):
  """Calculate the Normalized Difference Water Index (NDWI).

  Note:  This functions is copied from the coastsat project
  and included for validation and comparison and existing methods:
  https://github.com/kvos/CoastSat

  Parameters
  ----------
  im1 : np.ndarray
      First image array (e.g., NIR band).
  im2 : np.ndarray
      Second image array (e.g., Green band).

  Returns
  -------
  np.ndarray
      NDWI image array.
  """
  # reshape the two images
  vec1 = im1.reshape(im1.shape[0] * im1.shape[1])
  vec2 = im2.reshape(im2.shape[0] * im2.shape[1])

  # initialise with NaNs
  vec_nd = np.ones(len(vec1)) * np.nan

  # compute the normalised difference index
  temp = np.divide(vec1 - vec2,
                    vec1 + vec2)
  vec_nd = temp
  #normalize values in vec_nd to go from 0 to 1
  vec_nd = (vec_nd - np.min(vec_nd)) / (np.max(vec_nd) - np.min(vec_nd))
  #vec_nd = vec_nd * 2 - 1
  vec_nd[np.isnan(vec_nd)] = 1

  # reshape into image
  im_nd = vec_nd.reshape(im1.shape[0], im1.shape[1])

  inv_im_nd = 1 - im_nd
  return inv_im_nd

def process_contours(contours):
  """Process contours to remove any NaN values.

  Note:  This functions is copied from the coastsat project
  and included for validation and comparison and existing methods:
  https://github.com/kvos/CoastSat

  Parameters
  ----------
  contours : list
      List of contours.

  Returns
  -------
  list
      List of contours without NaN values.
  """

  contours_nonans = []
  # loop through contours and only keep the ones without NaNs
  for k in range(len(contours)):
      if np.any(np.isnan(contours[k])):
          index_nan = np.where(np.isnan(contours[k]))[0]
          contours_temp = np.delete(contours[k], index_nan, axis=0)
          if len(contours_temp) > 1:
              contours_nonans.append(contours_temp)
      else:
          contours_nonans.append(contours[k])
  return contours_nonans

def universal_simple_thresholding(im_ms, ave, im_labels, im_ref_buffer):
  """Apply universal simple thresholding to the image.

  note: this is a dumb version of thresholding, included as a benchmark/comparison

  Parameters
  ----------
  im_ms : np.ndarray
      Multispectral image array.
  ave : np.ndarray
      Average image array.
  im_labels : np.ndarray
      Image labels array.
  im_ref_buffer : np.ndarray
      Reference buffer mask.

  Returns
  -------
  tuple
      - np.ndarray : Blurred NDWI image.
      - list : Contours of the NDWI image.
      - list : Contours of the normalized average image.
      - np.ndarray : Normalized average image buffer.
  """
  np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

  nrows = im_ref_buffer.shape[0]
  ncols = im_ref_buffer.shape[1]

  im_wi = ndwi(im_ms[:,:,1], ave) # NDWI G - NIR / G + NIR
  norm_ave = (ave - np.min(ave)) / (np.max(ave) - np.min(ave))
  im_wi_buffer = blend_masks(im_wi,im_ref_buffer,im_labels[:,:,0])

  #apply gaussian blur to ndwi_buffer
  blurred_image = cv2.GaussianBlur(im_wi_buffer, (101,101), cv2.BORDER_DEFAULT)

  # Find contours at a constant value
  contours = measure.find_contours(im_wi_buffer, 0.5)

  nir_buffer = blend_masks(norm_ave,im_ref_buffer,im_labels[:,:,0])
  nir_contours = measure.find_contours(nir_buffer, 0.5)

  return blurred_image,contours,nir_contours,nir_buffer

def false_color_index(ndwi_raster):
  """Create an RGB false color visualization of an NDWI raster.

  Parameters
  ----------
  ndwi_raster : np.ndarray
      NDWI raster array.

  Returns
  -------
  np.ndarray
      RGB false color image.
  """
  #create empty array
  false_color_r = np.zeros_like(ndwi_raster)
  false_color_b = np.zeros_like(ndwi_raster)
  false_color_g = np.zeros_like(ndwi_raster)

  #loop through pixels
  for x in range(ndwi_raster.shape[0]):
    for y in range(ndwi_raster.shape[1]):
      #for pixel, if ndwi
      absolute_ndwi = np.abs(ndwi_raster[x,y])
      if ndwi_raster[x,y] < 0:
        false_color_r[x,y] = 0
        false_color_b[x,y] = 0
        false_color_g[x,y] = absolute_ndwi
      else:
        false_color_r[x,y] = 0
        false_color_b[x,y] = absolute_ndwi
        false_color_g[x,y] = 0

  false_color_img = np.stack((false_color_r,false_color_g,false_color_b), axis=-1)
  return false_color_img

def refine_shorelines(base_path):
  shoreline_folder = base_path + "/SHORELINE"
  shoreline_paths = os.listdir(shoreline_folder)
  shoreline_paths = [f for f in shoreline_paths if f.endswith('_sl.csv')]
  
  up_folder = base_path + "/NORMALIZED" # up_folder = base_path + "/UP"
  up_paths = os.listdir(up_folder)
  up_paths = [f for f in up_paths if f.endswith('_up.png')]

  #find the most *_processing.csv file path in the base_path
  processing_files = glob.glob(os.path.join(base_path, "*_processing.csv"))
  if processing_files:
      latest_processing_file = max(processing_files, key=os.path.getctime)
      print(f"Found latest processing file: {latest_processing_file}")
  else:
      print("No processing file found.")
      return None
  #get names from df names column
  table_path = latest_processing_file

  df = pd.read_csv(table_path)
  df['refined'] = False
  names = df['name'].values

  # if the total number of files is less than 50, skip filtering
  if len(shoreline_paths) < 50:
    print(f"Skipping filtering, only {len(shoreline_paths)} shorelines found.")
    good_files = shoreline_paths
    bad_files = []
    rejection_reasons = {}
  else:
    print(f"Filtering {len(shoreline_paths)} shorelines...")
    good_files, bad_files, rejection_reasons = boundary_filter.filter_shorelines(
        csv_files=shoreline_paths,
        shoreline_path=shoreline_folder,
        iqr_multiplier=2.0,           # Adjust for stricter/looser filtering
        location_eps=1.5,             # Adjust for location clustering sensitivity  
        enable_shape_filter=True,     # Enable/disable shape similarity filter
        shape_similarity_threshold=0.3, # Adjust shape similarity strictness
        verbose=True                  # Print progress information
    )

  # add the rejection reasons to the df
  df['rejection_reason'] = ''
  for i in range(len(df)):
    name = df.loc[i, 'name']
    if name in rejection_reasons:
      df.loc[i, 'rejection_reason'] = rejection_reasons[name]

  # filter names to only those in good_files
  good_names = [os.path.basename(f).replace('_nir_sl.csv','') for f in good_files]
  names = [n for n in names if n in good_names] 
  shoreline_paths = good_files


  print(f"Refining {len(names)} shorelines...")
  for i in range(len(names)):
    print(names[i])

    #get image from up_paths with matching name
    up_path_list = [f for f in up_paths if names[i] in f]
    shoreline_path_list = [f for f in shoreline_paths if names[i] in f]
    if not up_path_list or not shoreline_path_list:
      continue
    up_path = up_path_list[0]
    shoreline_path = shoreline_path_list[0]

    refiner = boundary_refine(shoreline_folder + "/" + shoreline_path, up_folder + "/" + up_path, periodic=True)
    refiner.fit_nurbs()
    nurbs_pts,normals = refiner.calc_normal_vector_along_nurbs()
    nml_pts = refiner.generate_normal_sample_pts()
    bnd = refiner.normal_thresholding()

    # check the results of the thresholding for self intersections.  If ther are are any, try kmeans thresholding
    if boundary_filter.check_self_intersection(bnd):
      print("Self intersection detected, trying kmeans thresholding...")
      bnd = refiner.kmeans_thresholding()
      if boundary_filter.check_self_intersection(bnd):
        # remove the self intersection
        print("unavoidable self-intersection detected, removing intersections...")
        bnd = boundary_filter.remove_self_intersections(bnd)

    df.loc[df['name'] == names[i], 'refined'] = True
  df.to_csv(table_path, index=False)

  return table_path