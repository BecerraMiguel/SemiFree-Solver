"""
X-Point Boundary Transformation Module
=======================================

Transforms a D-shaped plasma boundary to incorporate magnetic X-points
using hybrid B-spline and linear interpolation methods with adaptive
loop prevention.

Two adaptive methods are provided:
    1. Multi-parameter adaptive: adjusts influence_radius_factor,
       cusp_window_size, and smoothness.
    2. Dual-parameter adaptive: adjusts influence_radius_factor and
       cusp_window_size only (smoothness held constant).

Usage (command-line):
    python3 xpoint_transform.py <input_file> <output_file> <method> <xpoints_file>

Arguments:
    input_file   : Boundary points file (R Z per line, in meters)
    output_file  : Output transformed boundary (R Z per line, in meters)
    method       : 1 (multi-parameter) or 2 (dual-parameter)
    xpoints_file : X-point coordinates (R Z per line, in meters)

Dependencies:
    numpy, scipy
"""

import sys
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from typing import List, Tuple, Dict


# =============================================================================
# Geometry Utilities
# =============================================================================

def _ccw(a: Tuple[float, float],
         b: Tuple[float, float],
         c: Tuple[float, float]) -> bool:
    """Check if three points are in counter-clockwise order."""
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(seg1: Tuple, seg2: Tuple) -> bool:
    """Check if two line segments intersect using the CCW algorithm."""
    a, b = seg1
    c, d = seg2
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


def _detect_loops(rb: np.ndarray,
                  zb: np.ndarray) -> Tuple[bool, int, int]:
    """
    Detect self-intersections in a closed curve.

    Parameters
    ----------
    rb, zb : np.ndarray
        Coordinates of the curve.

    Returns
    -------
    has_loop : bool
        True if a self-intersection was detected.
    idx1, idx2 : int
        Indices of the intersecting segments (-1 if none).
    """
    n = len(rb)
    for i in range(n - 1):
        seg1 = ((rb[i], zb[i]), (rb[i + 1], zb[i + 1]))
        for j in range(i + 2, n - 1):
            if j == n - 2 and i == 0:
                continue
            seg2 = ((rb[j], zb[j]), (rb[j + 1], zb[j + 1]))
            if _segments_intersect(seg1, seg2):
                return True, i, j
    return False, -1, -1


def _calculate_arc_length_parameter(rb: np.ndarray,
                                    zb: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculate normalized arc-length parameter s in [0, 1].

    Parameters
    ----------
    rb, zb : np.ndarray
        Boundary coordinates.

    Returns
    -------
    s : np.ndarray
        Normalized arc-length parameter.
    s_total : float
        Total perimeter length.
    """
    dx = np.diff(rb, append=rb[0])
    dy = np.diff(zb, append=zb[0])
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate([[0], np.cumsum(ds[:-1])])
    s_total = s[-1] + ds[-1]
    s_normalized = s / s_total
    return s_normalized, s_total


# =============================================================================
# Gaussian Transformation
# =============================================================================

def _apply_gaussian_transformation(
    rb: np.ndarray,
    zb: np.ndarray,
    x_points: List[Tuple[float, float]],
    influence_radius: float,
    combination_method: str = 'nearest'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Gaussian transformation moving boundary points toward X-points.

    Parameters
    ----------
    rb, zb : np.ndarray
        Original boundary coordinates.
    x_points : list of (float, float)
        X-point coordinates.
    influence_radius : float
        Radius of Gaussian influence.
    combination_method : str
        Method for combining multiple X-point influences:
        'nearest', 'weighted', or 'dominant'.

    Returns
    -------
    rb_transformed, zb_transformed : np.ndarray
        Transformed coordinates.
    nearest_xpoint_idx : np.ndarray
        Index of nearest X-point for each boundary point.
    """
    n_points = len(rb)
    n_xpoints = len(x_points)

    distances = np.zeros((n_points, n_xpoints))
    for i, (rx, zx) in enumerate(x_points):
        distances[:, i] = np.sqrt((rb - rx)**2 + (zb - zx)**2)

    nearest_xpoint_idx = np.argmin(distances, axis=1)

    rb_t = rb.copy()
    zb_t = zb.copy()

    if combination_method == 'nearest':
        for i in range(n_points):
            idx = nearest_xpoint_idx[i]
            rx, zx = x_points[idx]
            w = np.exp(-(distances[i, idx] / influence_radius)**2)
            rb_t[i] = rb[i] * (1 - w) + rx * w
            zb_t[i] = zb[i] * (1 - w) + zx * w

    elif combination_method == 'weighted':
        for i in range(n_points):
            weights = np.exp(-(distances[i, :] / influence_radius)**2)
            total_weight = np.sum(weights)
            if total_weight > 1e-10:
                r_targets = np.array([xp[0] for xp in x_points])
                z_targets = np.array([xp[1] for xp in x_points])
                ew = min(total_weight, 1.0)
                r_target = np.sum(weights * r_targets) / total_weight
                z_target = np.sum(weights * z_targets) / total_weight
                rb_t[i] = rb[i] * (1 - ew) + r_target * ew
                zb_t[i] = zb[i] * (1 - ew) + z_target * ew

    elif combination_method == 'dominant':
        for i in range(n_points):
            idx = nearest_xpoint_idx[i]
            rx_n, zx_n = x_points[idx]
            w_nearest = np.exp(-(distances[i, idx] / influence_radius)**2)

            other_mask = np.arange(n_xpoints) != idx
            if np.any(other_mask):
                other_w = np.exp(-(distances[i, other_mask] / influence_radius)**2)
                total_other = np.sum(other_w)
                if total_other > 1e-10:
                    r_others = np.array([x_points[j][0] for j in range(n_xpoints) if j != idx])
                    z_others = np.array([x_points[j][1] for j in range(n_xpoints) if j != idx])
                    r_other = np.sum(other_w * r_others) / total_other
                    z_other = np.sum(other_w * z_others) / total_other
                    r_target = 0.8 * rx_n + 0.2 * r_other
                    z_target = 0.8 * zx_n + 0.2 * z_other
                else:
                    r_target, z_target = rx_n, zx_n
            else:
                r_target, z_target = rx_n, zx_n

            rb_t[i] = rb[i] * (1 - w_nearest) + r_target * w_nearest
            zb_t[i] = zb[i] * (1 - w_nearest) + z_target * w_nearest

    else:
        raise ValueError(f"Unknown combination_method: {combination_method}")

    for idx_xp, (rx, zx) in enumerate(x_points):
        dist_to_xp = np.sqrt((rb_t - rx)**2 + (zb_t - zx)**2)
        idx_closest = np.argmin(dist_to_xp)
        rb_t[idx_closest] = rx
        zb_t[idx_closest] = zx

    return rb_t, zb_t, nearest_xpoint_idx


# =============================================================================
# Cusp Region Definition
# =============================================================================

def _define_cusp_regions(
    s: np.ndarray,
    rb_transformed: np.ndarray,
    zb_transformed: np.ndarray,
    x_points: List[Tuple[float, float]],
    cusp_window_size: float
) -> Tuple[List[Dict], np.ndarray]:
    """
    Define cusp regions around each X-point in arc-length space.

    Parameters
    ----------
    s : np.ndarray
        Normalized arc-length parameter.
    rb_transformed, zb_transformed : np.ndarray
        Transformed boundary coordinates.
    x_points : list of (float, float)
        X-point coordinates.
    cusp_window_size : float
        Size of cusp window as fraction of perimeter.

    Returns
    -------
    cusp_regions : list of dict
        Information about each cusp region.
    mask_cusp_combined : np.ndarray
        Boolean mask for all cusp regions combined.
    """
    n_points = len(s)
    half_window = cusp_window_size / 2

    cusp_regions = []
    for idx_xp, (rx, zx) in enumerate(x_points):
        dist_to_xp = np.sqrt((rb_transformed - rx)**2 + (zb_transformed - zx)**2)
        idx_xpoint = np.argmin(dist_to_xp)
        s_xpoint = s[idx_xpoint]

        s_start = s_xpoint - half_window
        s_end = s_xpoint + half_window

        if s_start < 0:
            s_start += 1.0
        if s_end > 1.0:
            s_end -= 1.0

        cusp_regions.append({
            'idx_xpoint': idx_xp,
            'rx': rx,
            'zx': zx,
            'idx_boundary': idx_xpoint,
            's_xpoint': s_xpoint,
            's_start': s_start,
            's_end': s_end,
        })

    mask_cusp_combined = np.zeros(n_points, dtype=bool)
    for region in cusp_regions:
        s_start = region['s_start']
        s_end = region['s_end']
        if s_start < s_end:
            mask_cusp_combined |= (s >= s_start) & (s <= s_end)
        else:
            mask_cusp_combined |= (s >= s_start) | (s <= s_end)

    return cusp_regions, mask_cusp_combined


# =============================================================================
# Core Hybrid Method
# =============================================================================

def _hybrid_bspline_linear(
    rb: np.ndarray,
    zb: np.ndarray,
    x_points: List[Tuple[float, float]],
    smoothness: float = 0.01,
    influence_radius_factor: float = 0.25,
    cusp_window_size: float = 0.05,
    n_output_points: int = 500,
    cusp_points_density: float = 1.0,
    combination_method: str = 'nearest'
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Hybrid B-spline + linear interpolation method for multiple X-points.

    Smooth regions of the boundary are fitted with cubic B-splines, while
    cusp regions near X-points use linear interpolation to preserve the
    sharp geometry.

    Parameters
    ----------
    rb, zb : np.ndarray
        Original boundary coordinates.
    x_points : list of (float, float)
        X-point coordinates.
    smoothness : float
        B-spline smoothing factor.
    influence_radius_factor : float
        Gaussian influence radius relative to estimated minor radius.
    cusp_window_size : float
        Size of linear region around each X-point as fraction of perimeter.
    n_output_points : int
        Number of output boundary points.
    cusp_points_density : float
        Density multiplier for points in cusp regions.
    combination_method : str
        Gaussian combination method ('nearest', 'weighted', 'dominant').

    Returns
    -------
    rb_new, zb_new : np.ndarray
        Transformed boundary coordinates.
    info : dict
        Diagnostic information.
    """
    n_xpoints = len(x_points)

    r_max, r_min = rb.max(), rb.min()
    a_estimate = (r_max - r_min) / 2
    influence_radius = a_estimate * influence_radius_factor

    rb_t, zb_t, nearest_idx = _apply_gaussian_transformation(
        rb, zb, x_points, influence_radius, combination_method
    )

    s, s_total = _calculate_arc_length_parameter(rb_t, zb_t)

    cusp_regions, mask_cusp = _define_cusp_regions(
        s, rb_t, zb_t, x_points, cusp_window_size
    )

    total_cusp_fraction = cusp_window_size * n_xpoints
    smooth_fraction = max(1.0 - total_cusp_fraction, 0.3)

    # Extract smooth region data
    mask_smooth = ~mask_cusp
    s_smooth = s[mask_smooth]
    rb_smooth = rb_t[mask_smooth]
    zb_smooth = zb_t[mask_smooth]

    # Add boundary transition points from each cusp region
    for region in cusp_regions:
        s_start = region['s_start']
        s_end = region['s_end']

        if s_start < s_end:
            idx_cusp = np.where((s >= s_start) & (s <= s_end))[0]
        else:
            idx_cusp = np.where((s >= s_start) | (s <= s_end))[0]

        if len(idx_cusp) > 0:
            idx_sorted = idx_cusp[np.argsort(
                s[idx_cusp] if s_start < s_end
                else np.where(s[idx_cusp] >= s_start, s[idx_cusp], s[idx_cusp] + 1)
            )]
            idx_first = idx_sorted[0]
            idx_last = idx_sorted[-1]

            s_smooth = np.concatenate([s_smooth, [s[idx_first], s[idx_last]]])
            rb_smooth = np.concatenate([rb_smooth, [rb_t[idx_first], rb_t[idx_last]]])
            zb_smooth = np.concatenate([zb_smooth, [zb_t[idx_first], zb_t[idx_last]]])

    sort_idx = np.argsort(s_smooth)
    s_smooth = s_smooth[sort_idx]
    rb_smooth = rb_smooth[sort_idx]
    zb_smooth = zb_smooth[sort_idx]

    _, unique_idx = np.unique(np.round(s_smooth, decimals=8), return_index=True)
    s_smooth = s_smooth[unique_idx]
    rb_smooth = rb_smooth[unique_idx]
    zb_smooth = zb_smooth[unique_idx]

    # B-spline for smooth regions
    if len(s_smooth) >= 4:
        spline_r = UnivariateSpline(s_smooth, rb_smooth, s=smoothness, k=3)
        spline_z = UnivariateSpline(s_smooth, zb_smooth, s=smoothness, k=3)
    else:
        spline_r = interp1d(s_smooth, rb_smooth, kind='linear', fill_value='extrapolate')
        spline_z = interp1d(s_smooth, zb_smooth, kind='linear', fill_value='extrapolate')

    # Point distribution
    n_cusp_total = int(n_output_points * total_cusp_fraction * cusp_points_density)
    n_cusp_per_xpoint = max(n_cusp_total // n_xpoints, 3)
    n_smooth_total = n_output_points - n_cusp_per_xpoint * n_xpoints

    # Generate smooth region points
    regions_sorted = sorted(cusp_regions, key=lambda r: r['s_xpoint'])
    s_smooth_fine_list = []

    for i in range(n_xpoints):
        current_region = regions_sorted[i]
        next_region = regions_sorted[(i + 1) % n_xpoints]

        s_seg_start = current_region['s_end']
        s_seg_end = next_region['s_start']

        if s_seg_start > s_seg_end:
            s_seg_end += 1.0

        seg_length = s_seg_end - s_seg_start
        if seg_length < 0:
            seg_length += 1.0

        n_seg_points = max(int(n_smooth_total * seg_length / smooth_fraction), 2)

        if seg_length > 0.01:
            s_seg = np.linspace(s_seg_start, s_seg_end, n_seg_points)
            s_seg = s_seg % 1.0
            s_smooth_fine_list.append(s_seg)

    if s_smooth_fine_list:
        s_smooth_fine = np.concatenate(s_smooth_fine_list)
    else:
        s_smooth_fine = np.array([])

    if len(s_smooth_fine) > 0:
        rb_smooth_fine = spline_r(s_smooth_fine)
        zb_smooth_fine = spline_z(s_smooth_fine)
    else:
        rb_smooth_fine = np.array([])
        zb_smooth_fine = np.array([])

    # Generate cusp region points
    all_s_cusp = []
    all_rb_cusp = []
    all_zb_cusp = []

    for region in cusp_regions:
        rx = region['rx']
        zx = region['zx']
        s_xpoint = region['s_xpoint']
        s_start = region['s_start']
        s_end = region['s_end']

        r_boundary_start = float(spline_r(s_start % 1.0))
        z_boundary_start = float(spline_z(s_start % 1.0))
        r_boundary_end = float(spline_r(s_end % 1.0))
        z_boundary_end = float(spline_z(s_end % 1.0))

        dist_to_start = np.sqrt((rx - r_boundary_start)**2 + (zx - z_boundary_start)**2)
        dist_to_end = np.sqrt((rx - r_boundary_end)**2 + (zx - z_boundary_end)**2)
        total_dist = dist_to_start + dist_to_end

        if total_dist < 1e-10:
            all_s_cusp.append(np.array([s_xpoint]))
            all_rb_cusp.append(np.array([rx]))
            all_zb_cusp.append(np.array([zx]))
        else:
            n_seg1 = max(int(n_cusp_per_xpoint * dist_to_start / total_dist), 2)
            t1 = np.linspace(0, 1, n_seg1)
            rb_seg1 = r_boundary_start * (1 - t1) + rx * t1
            zb_seg1 = z_boundary_start * (1 - t1) + zx * t1

            if s_start > s_xpoint:
                s_seg1 = np.linspace(s_start - 1.0, s_xpoint, n_seg1) % 1.0
            else:
                s_seg1 = np.linspace(s_start, s_xpoint, n_seg1)

            n_seg2 = n_cusp_per_xpoint - n_seg1 + 1
            t2 = np.linspace(0, 1, n_seg2)
            rb_seg2 = rx * (1 - t2) + r_boundary_end * t2
            zb_seg2 = zx * (1 - t2) + z_boundary_end * t2

            if s_end < s_xpoint:
                s_seg2 = np.linspace(s_xpoint, s_end + 1.0, n_seg2) % 1.0
            else:
                s_seg2 = np.linspace(s_xpoint, s_end, n_seg2)

            all_s_cusp.append(np.concatenate([s_seg1, s_seg2[1:]]))
            all_rb_cusp.append(np.concatenate([rb_seg1, rb_seg2[1:]]))
            all_zb_cusp.append(np.concatenate([zb_seg1, zb_seg2[1:]]))

    if all_s_cusp:
        s_cusp_fine = np.concatenate(all_s_cusp)
        rb_cusp_fine = np.concatenate(all_rb_cusp)
        zb_cusp_fine = np.concatenate(all_zb_cusp)
    else:
        s_cusp_fine = np.array([])
        rb_cusp_fine = np.array([])
        zb_cusp_fine = np.array([])

    # Combine and sort
    s_combined = np.concatenate([s_smooth_fine, s_cusp_fine])
    rb_combined = np.concatenate([rb_smooth_fine, rb_cusp_fine])
    zb_combined = np.concatenate([zb_smooth_fine, zb_cusp_fine])

    sort_idx = np.argsort(s_combined % 1.0)
    rb_new = rb_combined[sort_idx]
    zb_new = zb_combined[sort_idx]

    info = {
        'n_xpoints': n_xpoints,
        'cusp_window_size': cusp_window_size,
        'influence_radius': influence_radius,
        'influence_radius_factor': influence_radius_factor,
        'smoothness': smoothness,
        'combination_method': combination_method,
        'iterations': 1,
    }

    return rb_new, zb_new, info


# =============================================================================
# Adaptive Methods
# =============================================================================

def adaptive_multiparameter(
    rb: np.ndarray,
    zb: np.ndarray,
    x_points: List[Tuple[float, float]],
    smoothness: float = 0.01,
    influence_radius_factor: float = 0.25,
    cusp_window_size: float = 0.05,
    n_output_points: int = 500,
    cusp_points_density: float = 1.0,
    combination_method: str = 'nearest',
    max_iterations: int = 10
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Multi-parameter adaptive boundary transformation for multiple X-points.

    Iteratively adjusts three parameters (influence_radius_factor,
    cusp_window_size, smoothness) to eliminate self-intersections in the
    transformed boundary.

    Parameters
    ----------
    rb, zb : np.ndarray
        Original boundary coordinates.
    x_points : list of (float, float)
        X-point coordinates.
    smoothness : float
        Initial B-spline smoothing factor.
    influence_radius_factor : float
        Initial Gaussian influence radius relative to minor radius.
    cusp_window_size : float
        Initial cusp window size as fraction of perimeter.
    n_output_points : int
        Number of output boundary points.
    cusp_points_density : float
        Point density multiplier in cusp regions.
    combination_method : str
        Gaussian combination method ('nearest', 'weighted', 'dominant').
    max_iterations : int
        Maximum number of adaptive iterations.

    Returns
    -------
    rb_new, zb_new : np.ndarray
        Transformed boundary coordinates.
    info : dict
        Diagnostic information including convergence status.
    """
    inf_factor = influence_radius_factor
    cusp_win = cusp_window_size
    smooth = smoothness

    n_xpoints = len(x_points)
    max_cusp_win = 0.20 / n_xpoints

    for iteration in range(max_iterations):
        rb_new, zb_new, info = _hybrid_bspline_linear(
            rb, zb, x_points,
            smoothness=smooth,
            influence_radius_factor=inf_factor,
            cusp_window_size=cusp_win,
            n_output_points=n_output_points,
            cusp_points_density=cusp_points_density,
            combination_method=combination_method,
        )

        has_loop, _, _ = _detect_loops(rb_new, zb_new)

        if not has_loop:
            info['iterations'] = iteration + 1
            info['final_influence_radius_factor'] = inf_factor
            info['final_cusp_window_size'] = cusp_win
            info['final_smoothness'] = smooth
            info['converged'] = True
            return rb_new, zb_new, info

        inf_factor *= 1.3
        cusp_win = min(cusp_win * 1.2, max_cusp_win)
        smooth *= 1.5

    info['iterations'] = max_iterations
    info['final_influence_radius_factor'] = inf_factor
    info['final_cusp_window_size'] = cusp_win
    info['final_smoothness'] = smooth
    info['converged'] = False
    return rb_new, zb_new, info


def adaptive_dual_parameter(
    rb: np.ndarray,
    zb: np.ndarray,
    x_points: List[Tuple[float, float]],
    smoothness: float = 0.01,
    influence_radius_factor: float = 0.25,
    cusp_window_size: float = 0.05,
    n_output_points: int = 500,
    cusp_points_density: float = 1.0,
    combination_method: str = 'nearest',
    max_iterations: int = 10
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Dual-parameter adaptive boundary transformation for multiple X-points.

    Iteratively adjusts two parameters (influence_radius_factor and
    cusp_window_size) while keeping smoothness constant. This typically
    produces better curve quality than the multi-parameter method.

    Parameters
    ----------
    rb, zb : np.ndarray
        Original boundary coordinates.
    x_points : list of (float, float)
        X-point coordinates.
    smoothness : float
        B-spline smoothing factor (held constant).
    influence_radius_factor : float
        Initial Gaussian influence radius relative to minor radius.
    cusp_window_size : float
        Initial cusp window size as fraction of perimeter.
    n_output_points : int
        Number of output boundary points.
    cusp_points_density : float
        Point density multiplier in cusp regions.
    combination_method : str
        Gaussian combination method ('nearest', 'weighted', 'dominant').
    max_iterations : int
        Maximum number of adaptive iterations.

    Returns
    -------
    rb_new, zb_new : np.ndarray
        Transformed boundary coordinates.
    info : dict
        Diagnostic information including convergence status.
    """
    inf_factor = influence_radius_factor
    cusp_win = cusp_window_size

    n_xpoints = len(x_points)
    max_cusp_win = 0.20 / n_xpoints

    for iteration in range(max_iterations):
        rb_new, zb_new, info = _hybrid_bspline_linear(
            rb, zb, x_points,
            smoothness=smoothness,
            influence_radius_factor=inf_factor,
            cusp_window_size=cusp_win,
            n_output_points=n_output_points,
            cusp_points_density=cusp_points_density,
            combination_method=combination_method,
        )

        has_loop, _, _ = _detect_loops(rb_new, zb_new)

        if not has_loop:
            info['iterations'] = iteration + 1
            info['final_influence_radius_factor'] = inf_factor
            info['final_cusp_window_size'] = cusp_win
            info['final_smoothness'] = smoothness
            info['converged'] = True
            return rb_new, zb_new, info

        inf_factor *= 1.4
        cusp_win = min(cusp_win * 1.3, max_cusp_win)

    info['iterations'] = max_iterations
    info['final_influence_radius_factor'] = inf_factor
    info['final_cusp_window_size'] = cusp_win
    info['final_smoothness'] = smoothness
    info['converged'] = False
    return rb_new, zb_new, info


# =============================================================================
# Command-Line Interface
# =============================================================================

def main():
    if len(sys.argv) != 5:
        sys.stderr.write(
            "Usage: python3 xpoint_transform.py "
            "<input_file> <output_file> <method> <xpoints_file>\n"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    method = int(sys.argv[3])
    xpoints_file = sys.argv[4]

    if method not in (1, 2):
        sys.stderr.write("Error: method must be 1 or 2.\n")
        sys.exit(1)

    boundary = np.loadtxt(input_file)
    rb = boundary[:, 0]
    zb = boundary[:, 1]

    xpoints_data = np.loadtxt(xpoints_file)
    if xpoints_data.ndim == 1:
        xpoints_data = xpoints_data.reshape(1, 2)
    x_points = [(row[0], row[1]) for row in xpoints_data]

    n_output_points = len(rb)

    if method == 1:
        rb_new, zb_new, info = adaptive_multiparameter(
            rb, zb, x_points, n_output_points=n_output_points
        )
    else:
        rb_new, zb_new, info = adaptive_dual_parameter(
            rb, zb, x_points, n_output_points=n_output_points
        )

    if not info.get('converged', True):
        sys.stderr.write(
            "Warning: adaptive method did not converge after "
            f"{info['iterations']} iterations.\n"
        )

    output = np.column_stack([rb_new, zb_new])
    np.savetxt(output_file, output, fmt='%.10e')


if __name__ == '__main__':
    main()
