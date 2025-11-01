from typing import List, Tuple, Set, Dict
import math
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import rgb2hex, rgb_to_hsv, hsv_to_rgb

# Import Paul Tol's color schemes
sys.path.append(str(Path(__file__).resolve().parent))
from paul_tol_schemes import tol_cmap

# ========================================
# CONFIGURATION PARAMETERS
# ========================================
class ColorConfig:
    """Centralized configuration for color mapping"""
    # Color palette from Paul Tol's rainbow_discrete
    COLORMAP_NAME = 'rainbow_discrete'

    # Brightness adjustment range (applied to Paul Tol colors)
    BRIGHTNESS_MIN = 0.6    # Darkest (60% of original)
    BRIGHTNESS_MAX = 1.1    # Lightest (110% of original) - reduced to avoid white

    # Saturation adjustment range (applied to Paul Tol colors)
    SATURATION_MIN = 0.6    # Most desaturated (60% of original)
    SATURATION_MAX = 1.0    # Original saturation

    DENSITY_SAT_REDUCTION = 0.3  # Max saturation reduction from density (30%)

    # Conflict resolution offsets (in color index space)
    COLOR_OFFSET_STRONG = 3      # Jump 3 colors in palette
    COLOR_OFFSET_WEAK = 1        # Jump 1 color in palette
    BRIGHTNESS_OFFSET_STRONG = 0.15    # ±15% brightness
    BRIGHTNESS_OFFSET_WEAK = 0.08      # ±8% brightness
    SAT_OFFSET = 0.1                    # ±10% saturation

    SIMILARITY_THRESHOLD = 0.15   # Perceptual distance threshold (normalized RGB space)


# ========================================
# GEOMETRIC UTILITIES (FIXED)
# ========================================

def get_geometric_centroid(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Calculate the geometric centroid (center of mass) of a polygon.
    Uses the shoelace formula for area-weighted centroid.
    
    This is NOT the average of vertices - it's the true center of the shape's area.
    """
    n = len(polygon)
    if n < 3:
        # Degenerate case - fall back to vertex average
        return (sum(p[0] for p in polygon) / n, sum(p[1] for p in polygon) / n)
    
    area = 0.0
    cx = 0.0
    cy = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        cross = polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1]
        area += cross
        cx += (polygon[i][0] + polygon[j][0]) * cross
        cy += (polygon[i][1] + polygon[j][1]) * cross
    
    area *= 0.5
    
    if abs(area) < 1e-10:
        # Degenerate polygon - fall back to vertex average
        return (sum(p[0] for p in polygon) / n, sum(p[1] for p in polygon) / n)
    
    cx /= (6.0 * area)
    cy /= (6.0 * area)
    
    return (cx, cy)


def polygons_overlap_sat(poly1: List[Tuple[float, float]], 
                         poly2: List[Tuple[float, float]]) -> bool:
    """
    Check if two CONVEX polygons overlap using Separating Axis Theorem (SAT).
    
    This is exact for convex polygons. For concave polygons, this can give
    false positives, but is still much better than bounding box checks.
    
    Returns True if polygons overlap, False otherwise.
    """
    # Quick bounding box reject first (fast path)
    def get_bbox(poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return (min(xs), min(ys), max(xs), max(ys))
    
    bbox1 = get_bbox(poly1)
    bbox2 = get_bbox(poly2)
    
    if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or 
        bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
        return False
    
    # SAT: Check all edge normals as potential separating axes
    def get_edges(poly):
        edges = []
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            # Normal (perpendicular) to edge
            normal = (-edge[1], edge[0])
            # Normalize
            length = math.sqrt(normal[0]**2 + normal[1]**2)
            if length > 1e-10:
                edges.append((normal[0] / length, normal[1] / length))
        return edges
    
    def project_polygon(poly, axis):
        """Project polygon onto axis and return min/max"""
        projections = [p[0] * axis[0] + p[1] * axis[1] for p in poly]
        return min(projections), max(projections)
    
    # Test all axes from both polygons
    for axis in get_edges(poly1) + get_edges(poly2):
        min1, max1 = project_polygon(poly1, axis)
        min2, max2 = project_polygon(poly2, axis)
        
        # Check if projections are separated
        if max1 < min2 - 1e-10 or max2 < min1 - 1e-10:
            return False  # Found separating axis - no overlap
    
    return True  # No separating axis found - polygons overlap


# ========================================
# COLOR UTILITIES
# ========================================

def convert_rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to HSV. RGB values should be in [0, 1]."""
    return rgb_to_hsv(rgb.reshape(1, 1, 3)).reshape(3)

def convert_hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV to RGB. Returns values in [0, 1]."""
    return hsv_to_rgb(hsv.reshape(1, 1, 3)).reshape(3)

def color_distance_rgb(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    """
    Calculate perceptual distance between two RGB colors.

    Uses Euclidean distance in RGB space (simple but effective).
    RGB values should be in [0, 1].
    """
    return np.sqrt(np.sum((rgb1 - rgb2) ** 2))


# ========================================
# MAIN ALGORITHM (IMPROVED)
# ========================================

def color_polygons_improved(polygons: List[List[Tuple[float, float]]],
                           config: ColorConfig = None) -> List[str]:
    """
    Assign colors to polygons using Paul Tol's rainbow_discrete palette with
    spatial coherence and conflict-aware adjustments.

    Strategy:
    - Base palette: Paul Tol's rainbow_discrete (colorblind-safe)
    - Hue: X position (mapped to palette progression)
    - Brightness: Y position (top = light, bottom = dark)
    - Saturation: Density (isolated = vivid, crowded = muted) + conflict resolution

    Args:
        polygons: List of polygons, each as list of (x, y) coordinate tuples
        config: ColorConfig object with tuning parameters

    Returns:
        List of hex color strings, one per polygon
    """
    if config is None:
        config = ColorConfig()

    n = len(polygons)
    if n == 0:
        return []

    # Step 1: Calculate geometric centroids
    centroids = [get_geometric_centroid(poly) for poly in polygons]

    # Step 2: Build conflict graph using SAT
    print(f"Building conflict graph for {n} polygons...")
    overlaps: List[Set[int]] = [set() for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap_sat(polygons[i], polygons[j]):
                overlaps[i].add(j)
                overlaps[j].add(i)

    # Step 3: Calculate overlap counts (density)
    densities = [len(overlaps[i]) for i in range(n)]
    max_density = max(densities) if densities else 1

    # Step 4: Get Paul Tol's rainbow_discrete colormap
    # Use appropriate LUT based on number of polygons
    lut = min(23, max(1, n))
    cmap = tol_cmap(config.COLORMAP_NAME, lut=lut)

    # Step 5: Global spatial ranking
    x_sorted = sorted(range(n), key=lambda i: centroids[i][0])
    y_sorted = sorted(range(n), key=lambda i: centroids[i][1])

    x_rank_map = {idx: rank for rank, idx in enumerate(x_sorted)}
    y_rank_map = {idx: rank for rank, idx in enumerate(y_sorted)}

    # Step 6: Process in spatial order
    process_order = sorted(range(n), key=lambda i: (centroids[i][0], centroids[i][1]))

    colors_assigned: Dict[int, np.ndarray] = {}  # Store as RGB arrays

    for idx in process_order:
        # Base color from Paul Tol palette (based on X position)
        x_position = x_rank_map[idx] / max(n - 1, 1)
        base_rgba = cmap(x_position)
        base_rgb = np.array(base_rgba[:3])

        # Convert to HSV for adjustments
        base_hsv = convert_rgb_to_hsv(base_rgb)

        # Brightness from Y position (top = light, bottom = dark)
        y_position = y_rank_map[idx] / max(n - 1, 1)
        brightness_factor = config.BRIGHTNESS_MAX - (config.BRIGHTNESS_MAX - config.BRIGHTNESS_MIN) * y_position

        # Saturation from density (high density = lower saturation)
        density_factor = densities[idx] / max_density
        saturation_factor = config.SATURATION_MAX - config.DENSITY_SAT_REDUCTION * density_factor

        # Apply base adjustments
        base_hsv[2] = np.clip(base_hsv[2] * brightness_factor, 0, 0.85)  # Brightness (max 0.85 to avoid white)
        base_hsv[1] = np.clip(base_hsv[1] * saturation_factor, 0.3, 1)  # Saturation (min 0.3 to avoid gray/white)

        # Resolve conflicts with already-assigned neighbors
        conflicting_neighbors = [n for n in overlaps[idx] if n in colors_assigned]

        if not conflicting_neighbors:
            # No conflicts - use base color
            final_rgb = convert_hsv_to_rgb(base_hsv)
        else:
            # Resolve conflicts
            final_rgb = resolve_conflict_improved(
                idx, base_rgb, base_hsv,
                conflicting_neighbors, colors_assigned,
                centroids, x_rank_map, y_rank_map, n, config, cmap
            )

        colors_assigned[idx] = final_rgb

    # Step 7: Convert to hex strings
    result = []
    for idx in range(n):
        rgb = colors_assigned[idx]
        hex_color = rgb2hex(rgb)
        result.append(hex_color)

    return result


def resolve_conflict_improved(idx: int,
                              base_rgb: np.ndarray,
                              base_hsv: np.ndarray,
                              conflicting_neighbors: List[int],
                              colors_assigned: Dict[int, np.ndarray],
                              centroids: List[Tuple[float, float]],
                              x_rank_map: Dict[int, int],
                              y_rank_map: Dict[int, int],
                              n: int,
                              config: ColorConfig,
                              cmap) -> np.ndarray:
    """
    Resolve color conflicts using smart adjustments while preserving Paul Tol palette.

    Strategy:
    - Try shifting to adjacent palette colors if very similar
    - Adjust brightness and saturation to create distinction
    - Maintain spatial coherence where possible
    """
    my_pos = centroids[idx]
    my_x_rank = x_rank_map[idx]
    my_y_rank = y_rank_map[idx]

    # Start with base HSV
    working_hsv = base_hsv.copy()

    # Track maximum offsets needed
    color_shift = 0  # Shift in palette index
    brightness_offset = 0.0
    sat_offset = 0.0

    for neighbor_idx in conflicting_neighbors:
        neighbor_rgb = colors_assigned[neighbor_idx]

        # Check if too similar
        distance = color_distance_rgb(convert_hsv_to_rgb(working_hsv), neighbor_rgb)

        if distance < config.SIMILARITY_THRESHOLD:
            neighbor_pos = centroids[neighbor_idx]
            neighbor_x_rank = x_rank_map[neighbor_idx]
            neighbor_y_rank = y_rank_map[neighbor_idx]

            x_rank_diff = abs(my_x_rank - neighbor_x_rank)
            y_rank_diff = abs(my_y_rank - neighbor_y_rank)

            # Case 1: Close in BOTH dimensions
            if x_rank_diff <= 2 and y_rank_diff <= 2:
                # Try to shift palette color
                if my_pos[0] > neighbor_pos[0]:
                    color_shift = max(color_shift, config.COLOR_OFFSET_STRONG)
                else:
                    color_shift = min(color_shift, -config.COLOR_OFFSET_STRONG)

                # Adjust brightness
                if my_pos[1] > neighbor_pos[1]:
                    brightness_offset = max(brightness_offset, config.BRIGHTNESS_OFFSET_STRONG)
                else:
                    brightness_offset = min(brightness_offset, -config.BRIGHTNESS_OFFSET_STRONG)

                # Adjust saturation
                if working_hsv[1] > convert_rgb_to_hsv(neighbor_rgb)[1]:
                    sat_offset = max(sat_offset, config.SAT_OFFSET)
                else:
                    sat_offset = min(sat_offset, -config.SAT_OFFSET)

            # Case 2: Close in X only
            elif x_rank_diff <= 2:
                if my_x_rank > neighbor_x_rank:
                    color_shift = max(color_shift, config.COLOR_OFFSET_WEAK)
                else:
                    color_shift = min(color_shift, -config.COLOR_OFFSET_WEAK)

                # Reduce saturation slightly
                sat_offset = min(sat_offset, -config.SAT_OFFSET * 0.7)

            # Case 3: Close in Y only
            elif y_rank_diff <= 2:
                if my_y_rank > neighbor_y_rank:
                    brightness_offset = max(brightness_offset, config.BRIGHTNESS_OFFSET_WEAK)
                else:
                    brightness_offset = min(brightness_offset, -config.BRIGHTNESS_OFFSET_WEAK)

                # Reduce saturation slightly
                sat_offset = min(sat_offset, -config.SAT_OFFSET * 0.7)

            else:
                # Not very close spatially - just tweak saturation
                sat_offset = min(sat_offset, -config.SAT_OFFSET * 0.5)

    # Apply color shift if needed
    if color_shift != 0:
        # Get new base color from shifted position in palette
        x_position = x_rank_map[idx] / max(n - 1, 1)
        # Shift by a small amount in palette space
        shifted_position = np.clip(x_position + color_shift * 0.05, 0, 1)
        shifted_rgba = cmap(shifted_position)
        shifted_rgb = np.array(shifted_rgba[:3])
        working_hsv = convert_rgb_to_hsv(shifted_rgb)

    # Apply brightness and saturation offsets
    working_hsv[2] = np.clip(working_hsv[2] + brightness_offset, 0, 0.85)  # Max brightness 0.85 to avoid white
    working_hsv[1] = np.clip(working_hsv[1] + sat_offset, 0.3, 1)  # Min saturation 0.3 to stay colorful

    # Convert back to RGB
    final_rgb = convert_hsv_to_rgb(working_hsv)

    return final_rgb


# ========================================
# EXAMPLE USAGE
# ========================================

if __name__ == "__main__":
    # Example: 5 overlapping squares
    polygons = [
        [(0, 0), (10, 0), (10, 10), (0, 10)],      # Bottom-left
        [(8, 0), (18, 0), (18, 10), (8, 10)],      # Bottom-mid (overlaps with 0)
        [(0, 8), (10, 8), (10, 18), (0, 18)],      # Top-left (overlaps with 0)
        [(8, 8), (18, 8), (18, 18), (8, 18)],      # Top-mid (overlaps all!)
        [(16, 8), (26, 8), (26, 18), (16, 18)],    # Top-right
    ]

    print("Coloring polygons with Paul Tol's rainbow_discrete palette...")
    colors = color_polygons_improved(polygons)

    for i, (poly, color) in enumerate(zip(polygons, colors)):
        centroid = get_geometric_centroid(poly)
        print(f"Polygon {i} at {centroid}: {color}")