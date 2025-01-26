from sklearn.cluster import KMeans
from skimage import measure
import skimage.filters as filters

import math
from geomdl import fitting
import numpy as np
from PIL import Image



class boundary_refine:
  def __init__(self,shoreline_path,img_path,delta=0.005,periodic=True):
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

    #initialize boundary
    self.refined_boundary = None

  def basic_thresholding(self,threshold):
    return self

  def buffer_thresholding(self,threshold,buffer):
    return self
  
  def normal_thresholding(self):
    """
    Refine the shoreline boundary using normal vectors along a nurbs curve

    Returns:
    -----------
    refined_boundary: np.array
    """
    self.fit_nurbs()
    self.calc_normal_vector_along_nurbs()

    sample_pts = self.generate_normal_sample_pts()

    sampled = self.sample_image(sample_pts)

    self.refined_boundary = self.threshold_samples(sampled)

    self.save_refined_shoreline()

    return self.refined_filepath

  def fit_nurbs(self,degree=3,size=24):
    shoreline = self.shoreline

    #if periodic, set the last point equal to the first point
    if self.periodic:
        shoreline[len(shoreline)-1] = shoreline[0]

    #convert shoreline into a list of tuples
    pline = [tuple(x) for x in shoreline]

    try:
      self.nurbs = fitting.approximate_curve(pline, degree, ctrlpts_size=size, centripetal=False)
    except Exception as e:
      print(str(e))
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
    sample_pts = []
    curve_points = self.crv_pts
    count = self.sample_size
    normals = self.normals
  
    for i in range(len(curve_points)):
      sample_pt = curve_points[i] + normals[i]*(-count/2)
      t_samples = []
      t_samples.append(sample_pt)

      for j in range((count*2)-1):
        sample_pt = sample_pt + (normals[i]/2)
        t_samples.append(sample_pt)

      sample_pts.append(t_samples)

    self.sample_pts = sample_pts

    return sample_pts

  def sample_image(self, sample_pts, scale_down=1):
    image_array = np.array(self.img)
    sampled_pts = []

    for t_pts in sample_pts:
      t_samples = []
      for pt in t_pts:
        try:
          if int(pt[0]/scale_down) < image_array.shape[1] and int(pt[1]/scale_down) < image_array.shape[0]:
            t_samples.append([pt[0],pt[1],image_array[int(pt[1]/scale_down),int(pt[0]/scale_down)]])
          else:
            print("sample out of bounds")
            min_pixel = np.min(image_array.reshape(image_array.shape[0]*image_array.shape[1],image_array.shape[2]),axis=0)
            t_samples.append([pt[0],pt[1],min_pixel])
        except Exception as e:
          print(str(e))
          t_samples.append([pt[0],pt[1],np.average(image_array)])
      sampled_pts.append(t_samples)
    self.sample_values = sampled_pts
    return sampled_pts

  def threshold_samples(self, sampled):
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

      seg_slopes[s] = self.find_highest_derivatives(seg_array)

    refined_boundary_pts = np.array(self.rolling_highest_slope(seg_slopes, sampled))

    return refined_boundary_pts

  def cluster_transects(self, sampled):
    segmentation_transects = []
    refined_boundary_pts = []
    print(len(sampled))
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

      #get the point at the cluster boundary
      boundary_val = self.find_cluster_boundary(seg_array[:,0],seg_array[:,2],t_s)

      segmentation_transects.append(seg_array)
      refined_boundary_pts.append(boundary_val)

    return refined_boundary_pts

  #this function is directly from coastsat!!
  #use mask as labels
  def find_wl_contours2(self, im_ms, nir, im_labels, im_ref_buffer):
    np.seterr(all='ignore') # raise/ignore divisions by 0 and nans
    # create array with same shape as im_ref_buffer with all zeros to use as cloud mask since we do not have one
    cloud_mask = np.zeros_like(im_ref_buffer, dtype=bool)

    nrows = im_ref_buffer.shape[0]
    ncols = im_ref_buffer.shape[1]

    # calculate Normalized Difference Modified Water Index (NIR - G)
    im_wi = self.ndwi(nir[:,:,0], im_ms[:,:,1], cloud_mask)

    #export index
    lens = im_wi

    # stack indices together
    im_ind = np.stack((im_wi, im_mwi), axis=-1)
    vec_ind = im_ind.reshape(nrows*ncols,2)

    # reshape labels into vectors
    vec_sand = im_labels[:,:,0].reshape(ncols*nrows)
    vec_water = im_labels[:,:,1].reshape(ncols*nrows)

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
    contours_wi = self.process_contours(contours_wi)

    # only return MNDWI contours and threshold
    return contours_wi, t_ave,lens

  def save_refined_shoreline(self):
    #save the refined shoreline to a csv file
    self.refined_filepath = self.shoreline_path.replace('_sl','_rl')
    np.savetxt(self.refined_filepath, self.refined_boundary, delimiter=",", fmt="%f")

################################################################################
#######################     static methods     #################################
################################################################################

  def rolling_highest_slope(self, seg_slopes, segmentation_transects,wz=3):
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

  def cluster_transects(self, sampled):
    segmentation_transects = []
    boundary_pts = []
    print(len(sampled))
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

      #get the point at the cluster boundary
      boundary_val = find_cluster_boundary(seg_array[:,0],seg_array[:,2],t_s)

      segmentation_transects.append(seg_array)
      boundary_pts.append(boundary_val)
    return segmentation_transects,boundary_pts

  def find_cluster_boundary(self, values,labels,pts):

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

  #from coastsat
  def ndwi(im1, im2):
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

  #from coastsat
  def process_contours(contours):
      # initialise variable
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
  
  # # this is the dumb version of thresholding, perhaps helpful to show as comparison
  # def simple_cubes(self, im_ms, ave, im_labels, im_ref_buffer):
  #   np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

  #   nrows = im_ref_buffer.shape[0]
  #   ncols = im_ref_buffer.shape[1]

  #   im_wi = ndwi(im_ms[:,:,1], ave) # NDWI G - NIR / G + NIR
  #   norm_ave = (ave - np.min(ave)) / (np.max(ave) - np.min(ave))
  #   im_wi_buffer = blend_masks(im_wi,im_ref_buffer,im_labels[:,:,0])

  #   #apply gaussian blur to ndwi_buffer
  #   blurred_image = cv2.GaussianBlur(im_wi_buffer, (101,101), cv2.BORDER_DEFAULT)

  #   # Find contours at a constant value
  #   contours = measure.find_contours(im_wi_buffer, 0.5)

  #   nir_buffer = blend_masks(norm_ave,im_ref_buffer,im_labels[:,:,0])
  #   nir_contours = measure.find_contours(nir_buffer, 0.5)

  #   return blurred_image,contours,nir_contours,nir_buffer



#create an RGB false color vis of a ndwi raster.  
def false_color_vis(ndwi_raster):
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