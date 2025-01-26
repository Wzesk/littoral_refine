import matplotlib.pyplot as plt
import numpy as np
from cog import BasePredictor, Input, Path
import extract_boundary
import refine_boundary
import csv

def read_csv(file_path):
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return str(data)



def identify_periodic(input_shoreline_array, threshold=0.25):
    """
    Identify if the shoreline is periodic by comparing the distance between the first and last points
    with the total length of the shoreline.

    Parameters
    ----------
    input_shoreline_array : np.ndarray
        2D numpy array of the input shoreline.
    threshold : float
        Threshold for periodicity.

    Returns
    -------
    bool
        True if the shoreline is periodic, False otherwise.
    """
    periodic = False
    # Calculate the length of the shoreline
    length = np.sum(np.sqrt(np.sum(np.diff(input_shoreline_array, axis=0)**2, axis=1)))
    # Calculate the distance between the first and last points
    distance = np.linalg.norm(input_shoreline_array[0] - input_shoreline_array[-1])
    if distance / length < threshold:
        periodic = True

    return periodic



class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load any models or resources into memory for reuse across predictions"""
        pass

    def predict(
        self,
        input_shoreline: str = Input(
            description="polyline as an array of coordinates: [[x,y],[x,y] ...]"
        ),
        img_path: Path = Input(
            description="Path to the input image for shoreline refinement"
        ),
        # simplification: float = Input(
        #     description="Factor for boundary simplification", ge=0.0, le=10.0, default=0.5
        # ),
        # smoothing: int = Input(
        #     description="Factor for boundary smoothing", ge=0, le=10, default=2
        # ),
    ) -> Path:
        """
        Run the shoreline refinement process and save the visualization as an image.
        """
        submitted_path = "/tmp/input_shoreline.csv"
        refined_path = "/tmp/refined_shoreline.csv"
        output_path="/tmp/refined_shoreline_visualization.png"

        # # ==========================
        # # 1. Extract shoreline input
        # # ==========================
        # shoreline, buffer_array, shoreline_filepath = extract_boundary.get_shoreline(
        #     str(mask_filepath),
        #     simplification=simplification,
        #     smoothing=smoothing,
        # )

        # save a string to a txt file

        input_shoreline_array = np.array(eval(input_shoreline))
        likely_periodic = identify_periodic(input_shoreline_array)

        np.savetxt(submitted_path, input_shoreline_array, delimiter=",", fmt="%f")

        # ==========================
        # 2. Refine shoreline input
        # ==========================
        # Initialize refiner using the generated shoreline_filepath
        refiner = refine_boundary.boundary_refine(submitted_path, str(img_path), periodic=likely_periodic)

        # Run shore-normal refinement
        refiner.normal_thresholding()

        # ==========================
        # 3. Prepare for visualization
        # ==========================
        # # Original shoreline
        # original_shoreline = refiner.shoreline
        # # Points from NURBS curve
        # cp_arr = refiner.crv_pts

        # Refined shoreline
        bd_arr = refiner.refined_boundary

        # # Image used for sampling
        # img_arr = np.array(refiner.img)
        # # Sampled values
        # sampled_nir = refiner.sample_values

        # # ==========================
        # # 4. Visualization and Saving
        # # ==========================
        # plt.axis('equal')
        # plt.rcParams['figure.figsize'] = [25, 25]
        # plt.grid(linestyle=':', color='0.5')
        # plt.gca().invert_yaxis()

        # # Plot the sampled color points
        # for t_s in sampled_nir:
        #     for pt in t_s:
        #         # Scale pixel from [0..255] to [0..1]
        #         pixel = np.array([
        #             pt[2][0] / 255.0,
        #             pt[2][1] / 255.0,
        #             pt[2][2] / 255.0,
        #         ])
        #         plt.plot(pt[0], pt[1], '.', ms=5, color=pixel)

        # # Plot shorelines
        # plt.plot(original_shoreline[:, 0], original_shoreline[:, 1], color='blue')
        # plt.plot(cp_arr[:, 0], cp_arr[:, 1], color='green')
        # plt.plot(bd_arr[:, 0], bd_arr[:, 1], color='red')

        # plt.legend()
        # plt.title('Refined Shoreline Visualization')

        # # Save the plot as an image
        # plt.savefig(output_path)
        # print(f"Visualization saved to {output_path}")

        np.savetxt(refined_path, bd_arr, delimiter=",", fmt="%f")

        return Path(refined_path)
    
#### setting it up to respond with two outputs
# class Output(BaseModel):
#     cropped_table: Path
#     json: Path
#
# class Predictor(BasePredictor):
#     def predict(self) -> Output:
#          ...
#         return Output(cropped_table=..., json=...)