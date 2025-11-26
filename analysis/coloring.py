from typing import List, Tuple
import sys
from pathlib import Path

# Import Paul Tol's color schemes
sys.path.append(str(Path(__file__).resolve().parent))
from paul_tol_schemes import tol_cmap


# ========================================
# COLOR ASSIGNMENT
# ========================================

def assign_colors_by_position(centroids: List[Tuple[float, float]]) -> List[str]:
    """
    Assign colors to points based on their X position.

    Strategy:
    - Use 26 discrete colors from Paul Tol's rainbow_PuBr (colorblind-safe)
    - Assign based on X position (left to right)

    Args:
        centroids: List of (x, y) centroid tuples

    Returns:
        List of hex color strings, one per centroid
    """
    n = len(centroids)
    if n == 0:
        return []

    # Rainbow_PuBr discrete colors (26 colors)
    rainbow_pubr_colors = [
        '#6F4C9B', '#6059A9', '#5568B8', '#4E79C5', '#4D8AC6',
        '#4E96BC', '#549EB3', '#59A5A9', '#60AB9E', '#69B190',
        '#77B77D', '#8CBC68', '#A6BE54', '#BEBC48', '#D1B541',
        '#DDAA3C', '#E49C39', '#E78C35', '#E67932', '#E4632D',
        '#DF4828', '#DA2222', '#B8221E', '#95211B', '#721E17',
        '#521A13'
    ]

    # Sort by X position (left to right)
    x_sorted = sorted(range(n), key=lambda i: centroids[i][0])
    x_rank_map = {idx: rank for rank, idx in enumerate(x_sorted)}

    # Assign colors based on X rank
    result = [''] * n
    for idx in range(n):
        x_rank = x_rank_map[idx]
        # Map rank to color index (0 to 25)
        color_index = int((x_rank / max(n - 1, 1)) * 25)
        result[idx] = rainbow_pubr_colors[color_index]

    return result