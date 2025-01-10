import matplotlib.pyplot as plt
import numpy as np
import extract_boundary
import refine_boundary

def main():
    """
    Main function that demonstrates:
      1. Extracting shoreline from a mask image.
      2. Initial boundary simplification and smoothing.
      3. Refining the shoreline using normal thresholding.
      4. Saving the results (intermediate points and final boundary) as an image.
    """

    # ==========================
    # 1. Extract shoreline input
    # ==========================
    mask_filepath = 'sample/20241211T052119_20241211T052515_T43NCE_mask.png'

    # ==========================
    # 2. Get initial shoreline
    # ==========================
    shoreline, buffer_array, shoreline_filepath = extract_boundary.get_shoreline(
        mask_filepath,
        simplification=0.5,
        smoothing=2
    )

    # ==========================
    # 3. Refine shoreline input
    # ==========================
    img_path = 'sample/20241211T052119_20241211T052515_T43NCE_sr.png'
    boundary_path = 'sample/20241211T052119_20241211T052515_T43NCE_mask_sl.csv'

    # ==========================
    # 4. Refine shoreline (slope)
    # ==========================
    # Initialize refiner
    refiner = refine_boundary.boundary_refine(boundary_path, img_path)
    print(refiner)

    # Run shore-normal refinement
    refiner.normal_thresholding()

    # ==========================
    # 5. Prepare for visualization
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
    # 6. Visualization and Saving
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
                pt[2][0]/255.0,
                pt[2][1]/255.0,
                pt[2][2]/255.0
            ])
            plt.plot(pt[0], pt[1], '.', ms=5, color=pixel)

    # Plot shorelines
    plt.plot(original_shoreline[:, 0], original_shoreline[:, 1], color='blue')
    plt.plot(cp_arr[:, 0], cp_arr[:, 1], color='green')
    plt.plot(bd_arr[:, 0], bd_arr[:, 1], color='red')

    #plt.legend()
    #plt.title('Refined Shoreline Visualization')

    # Save the plot as an image
    output_image_path = '/output/refined_shoreline_visualization.png'
    plt.savefig(output_image_path)
    print(f"Visualization saved to {output_image_path}")

if __name__ == '__main__':
    main()
