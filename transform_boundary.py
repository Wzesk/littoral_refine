 
from PIL import Image
import json
import numpy as np

class boundary_transform:
    def __init__(self,json_path,img_path,scale=1):
        self.starting_bounds = None
        self.target_bounds = None

        img = Image.open(img_path)
        width, height = img.size
        self.starting_bounds = [[0,0],[width*scale,height*scale]]

        # Parse JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
            if "aoi" in data:
                self.target_bounds = data["aoi"]
            else:
                print("Error: 'aoi' key not found in JSON file.")
                return None, None

        return self.starting_bounds, self.target_bounds
    

    def transform_to_world(self, xy_list,):

        transformed_xy_list = np.zeros_like(xy_list)
        i=0
        for x, y in xy_list:
            # flip y
            y = self.starting_bounds[1][1] - y

            # Normalize the x, y values to the range [0, 1] within start_bounds
            norm_x = (x - self.starting_bounds[0][0]) / (self.starting_bounds[1][0] - self.starting_bounds[0][0])
            norm_y = (y - self.starting_bounds[0][1]) / (self.starting_bounds[1][1] - self.starting_bounds[0][1])

            # Scale the normalized values to the target bounds
            transformed_xy_list[i][0] = norm_x * (self.target_bounds[1][0] - self.target_bounds[0][0]) + self.target_bounds[0][0]
            transformed_xy_list[i][1] = norm_y * (self.target_bounds[1][1] - self.target_bounds[0][1]) + self.target_bounds[0][1]
            i+=1

        return transformed_xy_list