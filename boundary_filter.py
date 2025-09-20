"""
Shoreline Quality Filter Module

This module provides functions to filter shoreline CSV files based on multiple quality metrics:
- Geometric validity (self-intersection, minimum points)
- Statistical outliers (length, area, compactness)  
- Spatial clustering (location consistency)
- Shape similarity (optional)

Author: Coastal Geotools Pipeline
Usage:
    from shoreline_filter import filter_shorelines, analyze_rejection_reasons
    
    good_files, bad_files, reasons = filter_shorelines(csv_files, path)
    analyze_rejection_reasons(reasons)
"""

import pandas as pd
import os
import numpy as np
from shapely.geometry import LineString, Polygon
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional


class ShorelineMetrics:
    """Container for shoreline geometric and statistical metrics"""
    
    def __init__(self, file: str, coords: np.ndarray, length: float, 
                 area: float, centroid_x: float, centroid_y: float, compactness: float):
        self.file = file
        self.coords = coords
        self.length = length
        self.area = area
        self.centroid_x = centroid_x
        self.centroid_y = centroid_y
        self.compactness = compactness


def is_self_intersecting(x: np.ndarray, y: np.ndarray) -> bool:
    """Check if shoreline coordinates form a self-intersecting line"""
    try:
        line = LineString(list(zip(x, y)))
        return not line.is_simple
    except:
        return True


def calculate_polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate area of polygon using Shapely"""
    try:
        polygon = Polygon(list(zip(x, y)))
        return polygon.area
    except:
        return 0.0


def calculate_centroid(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Calculate centroid of polygon"""
    try:
        polygon = Polygon(list(zip(x, y)))
        centroid = polygon.centroid
        return float(centroid.x), float(centroid.y)
    except:
        return float(np.mean(x)), float(np.mean(y))


def calculate_perimeter_length(coords: np.ndarray) -> float:
    """Calculate perimeter length of coordinate sequence"""
    return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))


def calculate_shape_compactness(length: float, area: float) -> float:
    """Calculate shape compactness (circularity ratio): 4πA/P²"""
    if area <= 0 or length <= 0:
        return 0.0
    return float((4 * np.pi * area) / (length ** 2))


def check_self_intersection(boundary_points: np.ndarray) -> bool:
    """
    Check if a boundary (shoreline) has self-intersections.
    
    Args:
        boundary_points: Array of boundary coordinates (N x 2)
        
    Returns:
        bool: True if self-intersections exist, False otherwise
    """
    if boundary_points is None or len(boundary_points) < 3:
        return False
        
    try:
        # Handle both string paths and numpy arrays
        if isinstance(boundary_points, str):
            coords = np.genfromtxt(boundary_points, delimiter=',')
        else:
            coords = np.array(boundary_points)
            
        if coords.ndim != 2 or coords.shape[1] != 2:
            return True  # Invalid format
            
        x, y = coords[:, 0], coords[:, 1]
        return is_self_intersecting(x, y)
    except Exception:
        return True


def remove_self_intersections(boundary_points: np.ndarray, 
                             max_iterations: int = 10,
                             tolerance: float = 1e-6) -> np.ndarray:
    """
    Remove self-intersections from a boundary by simplifying the geometry.
    
    This function uses multiple strategies to clean up self-intersecting boundaries:
    1. First attempts to fix with Shapely's buffer(0) operation
    2. Falls back to Douglas-Peucker simplification with increasing tolerance
    3. As a last resort, removes duplicate/very close points
    
    Args:
        boundary_points: Array of boundary coordinates (N x 2) 
        max_iterations: Maximum number of simplification attempts
        tolerance: Initial tolerance for simplification
        
    Returns:
        np.ndarray: Cleaned boundary coordinates without self-intersections
    """
    if boundary_points is None or len(boundary_points) < 3:
        return boundary_points
        
    try:
        # Handle both string paths and numpy arrays
        if isinstance(boundary_points, str):
            coords = np.genfromtxt(boundary_points, delimiter=',')
        else:
            coords = np.array(boundary_points)
            
        if coords.ndim != 2 or coords.shape[1] != 2:
            return coords
            
        # Check if already clean
        line = LineString(coords)
        if line.is_simple:
            return coords
            
        # Strategy 1: Try buffer(0) operation to fix topology
        try:
            # For closed polygons, try polygon approach
            if np.allclose(coords[0], coords[-1], atol=1e-8):
                poly = Polygon(coords[:-1])  # Remove duplicate last point
                if poly.is_valid:
                    fixed_poly = poly.buffer(0)
                    if hasattr(fixed_poly, 'exterior') and fixed_poly.exterior is not None:
                        fixed_coords = np.array(fixed_poly.exterior.coords)
                        # Ensure closure
                        if not np.allclose(fixed_coords[0], fixed_coords[-1], atol=1e-8):
                            fixed_coords = np.vstack([fixed_coords, fixed_coords[0:1]])
                        return fixed_coords
            else:
                # For open lines, try line approach
                fixed_line = line.buffer(0).boundary
                if hasattr(fixed_line, 'coords'):
                    return np.array(list(fixed_line.coords))
        except Exception:
            pass
            
        # Strategy 2: Douglas-Peucker simplification with increasing tolerance
        current_tolerance = tolerance
        for iteration in range(max_iterations):
            try:
                simplified = line.simplify(current_tolerance, preserve_topology=True)
                if simplified.is_simple and len(simplified.coords) >= 3:
                    result = np.array(list(simplified.coords))
                    
                    # For closed boundaries, ensure closure
                    if (np.allclose(coords[0], coords[-1], atol=1e-8) and 
                        not np.allclose(result[0], result[-1], atol=1e-8)):
                        result = np.vstack([result, result[0:1]])
                    
                    return result
                    
                current_tolerance *= 2  # Increase tolerance for next iteration
                
            except Exception:
                current_tolerance *= 2
                continue
                
        # Strategy 3: Manual cleanup - remove very close points
        cleaned_coords = [coords[0]]
        min_distance = np.percentile(np.linalg.norm(np.diff(coords, axis=0), axis=1), 10)
        min_distance = max(min_distance, tolerance * 10)
        
        for i in range(1, len(coords)):
            distance = np.linalg.norm(coords[i] - cleaned_coords[-1])
            if distance > min_distance:
                cleaned_coords.append(coords[i])
                
        cleaned_coords = np.array(cleaned_coords)
        
        # Ensure minimum number of points
        if len(cleaned_coords) < 3:
            # If too few points remain, sample from original
            step = max(1, len(coords) // 10)
            cleaned_coords = coords[::step]
            
        # For closed boundaries, ensure closure
        if (np.allclose(coords[0], coords[-1], atol=1e-8) and 
            not np.allclose(cleaned_coords[0], cleaned_coords[-1], atol=1e-8)):
            cleaned_coords = np.vstack([cleaned_coords, cleaned_coords[0:1]])
            
        return cleaned_coords
        
    except Exception as e:
        print(f"Warning: Could not remove self-intersections: {e}")
        return boundary_points


def load_shoreline_metrics(csv_files: List[str], shoreline_path: str, 
                          min_points: int = 3) -> Tuple[List[ShorelineMetrics], Dict[str, List[str]]]:
    """
    Load and calculate metrics for all shoreline CSV files
    
    Args:
        csv_files: List of CSV filenames to process
        shoreline_path: Directory containing the CSV files
        min_points: Minimum number of points required for valid shoreline
        
    Returns:
        Tuple of (valid_metrics_list, rejection_reasons_dict)
    """
    valid_metrics = []
    rejection_reasons = {}
    
    for csv_file in csv_files:
        csv_path = os.path.join(shoreline_path, csv_file)
        
        try:
            df = pd.read_csv(csv_path)
            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values
            
            # Validate minimum requirements
            if len(x) < min_points:
                rejection_reasons[csv_file] = ['too few points']
                continue
                
            if is_self_intersecting(x, y):
                rejection_reasons[csv_file] = ['self-intersecting']
                continue
            
            # Calculate all metrics
            coords = np.column_stack((x, y))
            length = calculate_perimeter_length(coords)
            area = calculate_polygon_area(x, y)
            centroid_x, centroid_y = calculate_centroid(x, y)
            compactness = calculate_shape_compactness(length, area)
            
            metrics = ShorelineMetrics(
                file=csv_file,
                coords=coords,
                length=length,
                area=area,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                compactness=compactness
            )
            valid_metrics.append(metrics)
            
        except Exception as e:
            rejection_reasons[csv_file] = [f'processing error: {str(e)}']
            continue
    
    return valid_metrics, rejection_reasons


def calculate_statistical_thresholds(metrics: List[ShorelineMetrics], 
                                   iqr_multiplier: float = 2.0) -> Dict[str, Tuple[float, float]]:
    """
    Calculate statistical filtering thresholds using IQR method
    
    Args:
        metrics: List of shoreline metrics
        iqr_multiplier: Multiplier for IQR-based outlier detection
        
    Returns:
        Dictionary with (lower, upper) thresholds for each metric
    """
    if not metrics:
        return {}
    
    # Extract metric arrays
    lengths = np.array([m.length for m in metrics])
    areas = np.array([m.area for m in metrics])
    compactnesses = np.array([m.compactness for m in metrics])
    
    thresholds = {}
    
    # Length thresholds
    length_median = np.median(lengths)
    length_iqr = np.subtract(*np.percentile(lengths, [75, 25]))
    thresholds['length'] = (
        length_median - iqr_multiplier * length_iqr,
        length_median + iqr_multiplier * length_iqr
    )
    
    # Area thresholds  
    area_median = np.median(areas)
    area_iqr = np.subtract(*np.percentile(areas, [75, 25]))
    thresholds['area'] = (
        area_median - iqr_multiplier * area_iqr,
        area_median + iqr_multiplier * area_iqr
    )
    
    # Compactness thresholds
    if len(compactnesses) > 0 and np.std(compactnesses) > 0:
        compactness_median = np.median(compactnesses)
        compactness_iqr = np.subtract(*np.percentile(compactnesses, [75, 25]))
        thresholds['compactness'] = (
            max(0, compactness_median - iqr_multiplier * compactness_iqr),
            compactness_median + iqr_multiplier * compactness_iqr
        )
    else:
        thresholds['compactness'] = (0.0, 1.0)
    
    return thresholds


def calculate_location_outliers(metrics: List[ShorelineMetrics], 
                              eps: float = 1.5, min_samples_ratio: float = 0.1) -> np.ndarray:
    """
    Identify location outliers using DBSCAN clustering
    
    Args:
        metrics: List of shoreline metrics
        eps: DBSCAN epsilon parameter for clustering sensitivity
        min_samples_ratio: Minimum samples as ratio of total points
        
    Returns:
        Boolean mask where True indicates shoreline is in main cluster
    """
    if len(metrics) <= 3:
        return np.ones(len(metrics), dtype=bool)
    
    # Extract centroids
    centroids = np.array([[m.centroid_x, m.centroid_y] for m in metrics])
    
    # Normalize centroids for clustering
    centroid_std = np.std(centroids, axis=0)
    if centroid_std[0] > 0 and centroid_std[1] > 0:
        centroids_norm = (centroids - np.mean(centroids, axis=0)) / centroid_std
        
        # Apply DBSCAN clustering
        min_samples = max(2, int(len(centroids) * min_samples_ratio))
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids_norm)
        
        # Find main cluster (largest cluster)
        valid_labels = clustering.labels_[clustering.labels_ >= 0]
        if len(valid_labels) > 0:
            main_cluster = np.bincount(valid_labels).argmax()
            return clustering.labels_ == main_cluster
    
    return np.ones(len(metrics), dtype=bool)


def apply_statistical_filters(metrics: List[ShorelineMetrics], 
                            thresholds: Dict[str, Tuple[float, float]],
                            location_mask: np.ndarray) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Apply statistical filters and return filtered files with rejection reasons
    
    Args:
        metrics: List of shoreline metrics
        thresholds: Statistical thresholds for each metric
        location_mask: Boolean mask for location filtering
        
    Returns:
        Tuple of (filtered_files_list, rejection_reasons_dict)
    """
    filtered_files = []
    rejection_reasons = {}
    
    length_lower, length_upper = thresholds['length']
    area_lower, area_upper = thresholds['area']
    compactness_lower, compactness_upper = thresholds['compactness']
    
    for i, metric in enumerate(metrics):
        reasons = []
        
        # Length filter
        if metric.length < length_lower or metric.length > length_upper:
            reasons.append(f"length ({metric.length:.1f})")
            
        # Area filter
        if metric.area < area_lower or metric.area > area_upper:
            reasons.append(f"area ({metric.area:.1f})")
            
        # Location filter
        if not location_mask[i]:
            reasons.append("location outlier")
            
        # Compactness filter
        if metric.compactness < compactness_lower or metric.compactness > compactness_upper:
            reasons.append(f"compactness ({metric.compactness:.3f})")
        
        if reasons:
            rejection_reasons[metric.file] = reasons
        else:
            filtered_files.append(metric.file)
    
    return filtered_files, rejection_reasons


def get_representative_points(coords: np.ndarray, n_points: int = 20) -> np.ndarray:
    """Get n evenly spaced points along the shoreline"""
    if len(coords) < n_points:
        return coords
    
    # Calculate cumulative distance along the shoreline
    distances = np.cumsum(np.concatenate([[0], np.linalg.norm(np.diff(coords, axis=0), axis=1)]))
    total_distance = distances[-1]
    
    # Get evenly spaced points
    target_distances = np.linspace(0, total_distance, n_points)
    representative_points = np.zeros((n_points, 2))
    
    for i, target_dist in enumerate(target_distances):
        idx = np.searchsorted(distances, target_dist)
        if idx >= len(coords):
            representative_points[i] = coords[-1]
        else:
            representative_points[i] = coords[idx]
    
    return representative_points


def apply_shape_similarity_filter(metrics: List[ShorelineMetrics],
                                 filtered_files: List[str],
                                 similarity_threshold: float = 0.3,
                                 n_representative_points: int = 20) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Apply shape similarity filter based on correlation with median shape
    
    Args:
        metrics: List of all shoreline metrics
        filtered_files: Files that passed previous filters
        similarity_threshold: Minimum correlation threshold (0-1)
        n_representative_points: Number of points for shape comparison
        
    Returns:
        Tuple of (shape_filtered_files, additional_rejection_reasons)
    """
    if len(filtered_files) <= 10:
        return filtered_files, {}
    
    # Get metrics for filtered files only
    filtered_metrics = [m for m in metrics if m.file in filtered_files]
    
    # Generate representative shapes
    representative_shapes = []
    for metric in filtered_metrics:
        # Normalize coordinates relative to centroid
        centroid = np.array([metric.centroid_x, metric.centroid_y])
        normalized_coords = metric.coords - centroid
        
        # Scale by average distance from centroid
        avg_dist = np.mean(np.linalg.norm(normalized_coords, axis=1))
        if avg_dist > 0:
            normalized_coords = normalized_coords / avg_dist
        
        rep_points = get_representative_points(normalized_coords, n_representative_points)
        representative_shapes.append(rep_points.flatten())
    
    if not representative_shapes:
        return filtered_files, {}
    
    representative_shapes = np.array(representative_shapes)
    
    # Calculate median shape
    median_shape = np.median(representative_shapes, axis=0)
    
    # Calculate similarities
    similarities = []
    for shape in representative_shapes:
        try:
            similarity = np.corrcoef(shape, median_shape)[0, 1]
            if np.isnan(similarity):
                similarity = 0.0
        except:
            similarity = 0.0
        similarities.append(similarity)
    
    # Filter by similarity
    shape_filtered_files = []
    shape_rejection_reasons = {}
    
    for metric, similarity in zip(filtered_metrics, similarities):
        if similarity >= similarity_threshold:
            shape_filtered_files.append(metric.file)
        else:
            shape_rejection_reasons[metric.file] = [f"shape similarity ({similarity:.3f})"]
    
    return shape_filtered_files, shape_rejection_reasons


def filter_shorelines(csv_files: List[str], 
                     shoreline_path: str,
                     iqr_multiplier: float = 2.0,
                     location_eps: float = 1.5,
                     min_points: int = 3,
                     enable_shape_filter: bool = True,
                     shape_similarity_threshold: float = 0.3,
                     verbose: bool = True) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """
    Main function to filter shoreline CSV files based on quality metrics
    
    Args:
        csv_files: List of CSV filenames to process
        shoreline_path: Directory containing CSV files
        iqr_multiplier: Multiplier for IQR-based outlier detection (default: 2.0)
        location_eps: DBSCAN epsilon for location clustering (default: 1.5)
        min_points: Minimum points required for valid shoreline (default: 3)
        enable_shape_filter: Whether to apply shape similarity filter (default: True)
        shape_similarity_threshold: Minimum shape similarity (default: 0.3)
        verbose: Whether to print progress information (default: True)
        
    Returns:
        Tuple of (good_files, bad_files, rejection_reasons)
    """
    if verbose:
        print(f"Filtering {len(csv_files)} shoreline files...")
    
    # Step 1: Load metrics and initial validation
    valid_metrics, initial_rejections = load_shoreline_metrics(
        csv_files, shoreline_path, min_points
    )
    
    if verbose:
        print(f"Valid shorelines after initial filtering: {len(valid_metrics)} out of {len(csv_files)}")
        print(f"Initially rejected: {len(initial_rejections)}")
    
    if not valid_metrics:
        return [], csv_files, initial_rejections
    
    # Step 2: Calculate statistical thresholds
    thresholds = calculate_statistical_thresholds(valid_metrics, iqr_multiplier)
    
    # Step 3: Calculate location outliers
    location_mask = calculate_location_outliers(valid_metrics, location_eps)
    
    # Step 4: Apply statistical filters
    filtered_files, stat_rejections = apply_statistical_filters(
        valid_metrics, thresholds, location_mask
    )
    
    # Combine rejection reasons
    all_rejections = {**initial_rejections, **stat_rejections}
    
    if verbose:
        print(f"Applied filters:")
        print(f"  Length range: {thresholds['length'][0]:.1f} - {thresholds['length'][1]:.1f}")
        print(f"  Area range: {thresholds['area'][0]:.1f} - {thresholds['area'][1]:.1f}")
        print(f"  Location clustering: {np.sum(location_mask)} of {len(location_mask)} in main cluster")
        print(f"  Compactness range: {thresholds['compactness'][0]:.3f} - {thresholds['compactness'][1]:.3f}")
        print(f"Filtered shorelines: {len(filtered_files)} out of {len(csv_files)}")
    
    # Step 5: Optional shape similarity filter
    if enable_shape_filter and len(filtered_files) > 10:
        if verbose:
            print("Applying shape similarity filter...")
        
        filtered_files, shape_rejections = apply_shape_similarity_filter(
            valid_metrics, filtered_files, shape_similarity_threshold
        )
        
        # Update rejection reasons
        for file, reasons in shape_rejections.items():
            if file in all_rejections:
                all_rejections[file].extend(reasons)
            else:
                all_rejections[file] = reasons
        
        if verbose:
            print(f"Shape similarity filter: {len(filtered_files)} files passed")
    
    # Generate final results
    rejected_files = [f for f in csv_files if f not in filtered_files]
    
    if verbose:
        print(f"\n=== FINAL RESULTS ===")
        print(f"Good files: {len(filtered_files)}")
        print(f"Bad files: {len(rejected_files)}")
        print(f"Success rate: {len(filtered_files)/len(csv_files)*100:.1f}%")
    
    return filtered_files, rejected_files, all_rejections


def analyze_rejection_reasons(rejection_details: Dict[str, List[str]], verbose: bool = True) -> Dict[str, int]:
    """
    Analyze and summarize rejection reasons
    
    Args:
        rejection_details: Dictionary with rejection reasons for each file
        verbose: Whether to print detailed analysis
        
    Returns:
        Dictionary with count of each rejection reason
    """
    reason_counts = {}
    
    for file, reasons in rejection_details.items():
        for reason in reasons:
            # Extract main reason type (before parentheses)
            reason_type = reason.split('(')[0].strip()
            reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
    
    if verbose:
        print("=== REJECTION REASONS ANALYSIS ===")
        print(f"Total rejected files: {len(rejection_details)}")
        print("\nRejection frequency:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count} files")
        
        print("\nExample rejected files:")
        for i, (file, reasons) in enumerate(list(rejection_details.items())[:5]):
            reasons_str = ", ".join(reasons)
            print(f"  {file}: {reasons_str}")
        
        if len(rejection_details) > 5:
            print(f"  ... and {len(rejection_details) - 5} more")
    
    return reason_counts


def save_filtering_results(good_files: List[str], bad_files: List[str], 
                          rejection_details: Dict[str, List[str]], 
                          output_dir: str, prefix: str = "shoreline_filter"):
    """
    Save filtering results to CSV files
    
    Args:
        good_files: List of files that passed filters
        bad_files: List of files that were rejected
        rejection_details: Dictionary with rejection reasons
        output_dir: Directory to save results
        prefix: Prefix for output filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save good files
    good_df = pd.DataFrame({'filename': good_files})
    good_df.to_csv(f"{output_dir}/{prefix}_good_files.csv", index=False)
    
    # Save bad files with reasons
    bad_data = []
    for file in bad_files:
        reasons = rejection_details.get(file, ['unknown'])
        bad_data.append({
            'filename': file,
            'rejection_reasons': '; '.join(reasons),
            'num_reasons': len(reasons)
        })
    
    bad_df = pd.DataFrame(bad_data)
    bad_df.to_csv(f"{output_dir}/{prefix}_rejected_files.csv", index=False)
    
    print(f"Results saved to {output_dir}/")
    print(f"  - {prefix}_good_files.csv ({len(good_files)} files)")
    print(f"  - {prefix}_rejected_files.csv ({len(bad_files)} files)")


if __name__ == "__main__":
    # Example usage
    print("Shoreline Filter Module")
    print("Import this module and use filter_shorelines() function")
    print("Example:")
    print("  from shoreline_filter import filter_shorelines")
    print("  good, bad, reasons = filter_shorelines(csv_files, path)")
