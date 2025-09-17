<div align="center">
  <h2>Littoral Shoreline Refinement</h2>
  <a target="_blank" href="https://colab.research.google.com/github/Wzesk/littoral_refine/blob/main/sample_shoreline_refinement.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</div>

## Overview

The `littoral_refine` module provides advanced shoreline refinement and boundary extraction for the Littoral shoreline analysis pipeline. This module converts segmentation masks into precise, smooth shoreline polylines and performs quality filtering to ensure accurate coastal boundary representation.

## Pipeline Integration

This module handles **Steps 9-10** in the Littoral pipeline, processing binary segmentation masks to extract and refine shoreline boundaries before geospatial transformation.

### Pipeline Context
```
Binary Masks → [Boundary Extraction] → [Boundary Refinement] → Raw Polylines → Geotransform → ...
```

### Interface
- **Step 9 Input**: Binary masks from segmentation
- **Step 9 Output**: Raw pixel coordinate polylines  
- **Step 9 Function**: `extract_boundary.get_shorelines_from_folder()`
- **Step 10 Input**: Raw polylines with artifacts
- **Step 10 Output**: Refined, smooth boundary representations
- **Step 10 Function**: `refine_boundary.refine_shorelines()`
- **Technology**: B-spline curve fitting, edge detection, and geometric filtering

## Processing Steps

### 1. Boundary Extraction (`extract_boundary.py`)
- **Purpose**: Extract initial shoreline coordinates from segmentation masks
- **Process**: Mask analysis → Edge detection → Polyline extraction → Artifact removal
- **Output**: Raw pixel coordinates as CSV files

### 2. Boundary Refinement (`refine_boundary.py`) 
- **Purpose**: Refine coarse shorelines to match high-resolution imagery details
- **Process**: B-spline fitting → Normal sampling → Slope analysis → Precision enhancement
- **Technology**: Advanced curve fitting with normal vector sampling for sub-pixel accuracy

## Usage in Pipeline

```python
import sys
sys.path.append('/path/to/littoral_refine')
from littoral_refine import extract_boundary, refine_boundary

# Step 9: Extract boundaries from segmentation masks
mask_folder = "/path/to/segmentation/masks"
shoreline_paths = extract_boundary.get_shorelines_from_folder(mask_folder)

# Step 10: Refine extracted boundaries  
site_folder = "/path/to/site/data"
refine_boundary.refine_shorelines(site_folder)
```

## Features

- **Sub-pixel Accuracy**: B-spline curve fitting for smooth, precise boundaries
- **Artifact Removal**: Automated detection and removal of segmentation artifacts
- **Quality Assessment**: Statistical filtering to identify and remove outliers
- **Batch Processing**: Efficient processing of multiple shorelines
- **Interactive Demo**: Live web interface for testing and visualization
- **Flexible Input**: Works with various mask formats and resolutions

## Output

The refinement process produces:
- **Extracted Boundaries**: Initial polyline coordinates from mask edge detection
- **Refined Shorelines**: Smooth, accurate boundary representations using B-spline curves
- **Quality Metrics**: Assessment of refinement accuracy and reliability
- **Processed Datasets**: Clean, analysis-ready shoreline coordinates

## Installation

### Environment Setup
```bash
# Create and activate conda environment
conda create --name littoral_pipeline python=3.10
conda activate littoral_pipeline
conda env update --file environment.yml --prune
```

### Interactive Usage
To experiment with the shoreline refiner, open the example notebook in Colab: 
<a target="_blank" href="https://colab.research.google.com/github/Wzesk/littoral_refine/blob/main/sample_shoreline_refinement.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Live Demo
To draw your own shorelines and test refinement, try the interactive demo: [Interactive Shoreline Refinement](https://refine.labs.littor.al/)

## Technical Approach

### B-Spline Curve Fitting
- Creates smooth mathematical representations of shoreline boundaries
- Enables sub-pixel precision through continuous curve interpolation
- Reduces noise and artifacts from discrete pixel-based extraction

### Normal Vector Sampling
- Samples imagery perpendicular to the smoothed shoreline
- Analyzes intensity gradients to locate precise land/water boundaries
- Uses slope detection and k-means clustering for boundary identification

### Quality Filtering
- Statistical analysis to identify and remove outlier boundaries
- Spatial clustering to group consistent shoreline measurements
- Automated quality assessment and reporting

## Contributors
This module and the larger project it is a part of has had numerous contributors, including:

**Core Development**: Walter Zesk, Tishya Chhabra, Leandra Tejedor, Philip Ndikum

**Project Leadership**: Sarah Dole, Skylar Tibbits, Peter Stempel

## Reference
This project draws extensive inspiration from the [CoastSat Project](https://github.com/kvos/CoastSat) described in detail here:

Vos K., Splinter K.D., Harley M.D., Simmons J.A., Turner I.L. (2019). CoastSat: a Google Earth Engine-enabled Python toolkit to extract shorelines from publicly available satellite imagery. Environmental Modelling and Software. 122, 104528. https://doi.org/10.1016/j.envsoft.2019.104528 (Open Access)
