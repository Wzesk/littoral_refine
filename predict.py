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
        simplification: float = Input(
            description="Factor for boundary simplification", ge=0.0, le=10.0, default=0.5
        ),
        smoothing: int = Input(
            description="Factor for boundary smoothing", ge=0, le=10, default=2
        ),
    ) -> Path:
        """
        Run the shoreline refinement process and save the visualization as an image.
        """
        submitted_path = "/tmp/input_shoreline.csv"
        output_path="/tmp/refined_shoreline_visualization.png"

        # # ==========================
        # # 1. Extract shoreline input
        # # ==========================
        # shoreline, buffer_array, shoreline_filepath = extract_boundary.get_shoreline(
        #     str(mask_filepath),
        #     simplification=simplification,
        #     smoothing=smoothing,
        # )

        np.savetxt(submitted_path, input_shoreline, delimiter=",", fmt="%f")

        # ==========================
        # 2. Refine shoreline input
        # ==========================
        # Initialize refiner using the generated shoreline_filepath
        refiner = refine_boundary.boundary_refine(submitted_path, str(img_path))

        # Run shore-normal refinement
        refiner.normal_thresholding()

        # ==========================
        # 3. Prepare for visualization
        # ==========================
        # Original shoreline
        original_shoreline = refiner.shoreline
        # Points from NURBS curve
        cp_arr = refiner.crv_pts
        # Refined shoreline
        bd_arr = refiner.refined_boundary
        # Image used for sampling
        img_arr = np.array(refiner.img)
        # Sampled values
        sampled_nir = refiner.sample_values

        # ==========================
        # 4. Visualization and Saving
        # ==========================
        plt.axis('equal')
        plt.rcParams['figure.figsize'] = [25, 25]
        plt.grid(linestyle=':', color='0.5')
        plt.gca().invert_yaxis()

        # Plot the sampled color points
        for t_s in sampled_nir:
            for pt in t_s:
                # Scale pixel from [0..255] to [0..1]
                pixel = np.array([
                    pt[2][0] / 255.0,
                    pt[2][1] / 255.0,
                    pt[2][2] / 255.0,
                ])
                plt.plot(pt[0], pt[1], '.', ms=5, color=pixel)

        # Plot shorelines
        plt.plot(original_shoreline[:, 0], original_shoreline[:, 1], color='blue')
        plt.plot(cp_arr[:, 0], cp_arr[:, 1], color='green')
        plt.plot(bd_arr[:, 0], bd_arr[:, 1], color='red')

        plt.legend()
        plt.title('Refined Shoreline Visualization')

        # Save the plot as an image
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")

        return Path(output_path)