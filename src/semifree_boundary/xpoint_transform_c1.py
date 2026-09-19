"""
X-Point Boundary Transformation Module (C1 cusp variant)
=========================================================

Transforms a D-shaped plasma boundary to incorporate magnetic X-points
using hybrid B-spline and linear interpolation methods with adaptive
loop prevention.

This is a variant of xpoint_transform.py: the two cusp-region legs
(anchor -> X-point and X-point -> anchor) are built as tangent-matched
(C1) quadratic Bezier curves instead of straight line segments, removing
the spurious secondary kink at the anchor points while keeping the
X-point itself a genuine (unmatched) corner. See
docs/x-point-transformation/ for the verification plots.

Two adaptive methods are provided:
    1. Multi-parameter adaptive: adjusts influence_radius_factor,
       cusp_window_size, and smoothness.
    2. Dual-parameter adaptive: adjusts influence_radius_factor and
       cusp_window_size only (smoothness held constant).

Usage (command-line):
    python3 xpoint_transform_c1.py <input_file> <output_file> <method> <xpoints_file>

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
from typing import List, Tuple, Dict, Union


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


def _quadratic_bezier(P0: np.ndarray,
                       P1: np.ndarray,
                       P2: np.ndarray,
                       t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a quadratic Bezier curve B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2.

    dB/dt(0) is proportional to (P1-P0) and dB/dt(1) to (P2-P1), which is
    what makes this construction tangent-matchable at either endpoint.
    """
    P0 = np.asarray(P0, dtype=float)
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)
    t = np.asarray(t, dtype=float)
    x = (1 - t)**2 * P0[0] + 2 * (1 - t) * t * P1[0] + t**2 * P2[0]
    y = (1 - t)**2 * P0[1] + 2 * (1 - t) * t * P1[1] + t**2 * P2[1]
    return x, y


def _tangent_at_arclength(spline_r, spline_z, s_val: float,
                           fd_h: float = 1e-5) -> np.ndarray:
    """
    Unit-less (unnormalized) tangent vector (dr/ds, dz/ds) at s_val.

    Uses the spline's own analytical derivative when available
    (UnivariateSpline); falls back to a central finite difference for the
    interp1d branch used when too few smooth-region points are available.
    """
    if hasattr(spline_r, 'derivative'):
        dr = float(spline_r.derivative()(s_val))
        dz = float(spline_z.derivative()(s_val))
    else:
        dr = float(spline_r(s_val + fd_h) - spline_r(s_val - fd_h)) / (2 * fd_h)
        dz = float(spline_z(s_val + fd_h) - spline_z(s_val - fd_h)) / (2 * fd_h)
    return np.array([dr, dz])


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

def _smooth_taper(d, r0, r1):
    """
    Smoothstep mask: 1 for d<=r0, 0 for d>=r1, C1-smooth cubic in between.
    r0, r1 may be scalars or arrays broadcastable against d (one entry per
    X-point); r1 is assumed > r0 (true whenever r1=cutoff_radius > 0, since
    r0 = 0.7*r1 by construction at the call sites below).

    Multiplying a weight by this forces it to EXACTLY zero beyond r1 (hard,
    compact support) while still reaching zero smoothly (no new kink from
    the cutoff itself) -- unlike a bare Gaussian, whose tail never reaches
    exactly zero, so it always leaks some pull onto distant, unrelated parts
    of the boundary (see the comment on cutoff_radius below).
    """
    span = np.maximum(np.asarray(r1) - np.asarray(r0), 1e-12)
    t = np.clip((d - r0) / span, 0.0, 1.0)
    return 1.0 - (3 * t**2 - 2 * t**3)


def _apply_gaussian_transformation(
    rb: np.ndarray,
    zb: np.ndarray,
    x_points: List[Tuple[float, float]],
    influence_radius: float,
    combination_method: str = 'nearest',
    cutoff_radius=None
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
        Radius of Gaussian influence (decay scale of the near-field pull).
    combination_method : str
        Method for combining multiple X-point influences:
        'nearest', 'weighted', or 'dominant'.
    cutoff_radius : None, float, or array-like, optional
        If given (scalar shared by every X-point, or one value per
        X-point), the Gaussian weight is additionally multiplied by
        _smooth_taper(d, 0.7*cutoff_radius, cutoff_radius): boundary points
        farther than cutoff_radius from their X-point get EXACTLY zero pull,
        guaranteeing the transformation stays local no matter how large
        influence_radius itself had to be made to reach a distant X-point
        (a plain Gaussian's tail never reaches zero, see
        compute_auto_cutoff_radius()). None (default) disables this --
        original unbounded-Gaussian behavior, unchanged.

    Returns
    -------
    rb_transformed, zb_transformed : np.ndarray
        Transformed coordinates.
    nearest_xpoint_idx : np.ndarray
        Index of nearest X-point for each boundary point.
    """
    n_points = len(rb)
    n_xpoints = len(x_points)

    # influence_radius may be a single scalar (shared by every X-point, the
    # original behavior) or an array-like with one entry per X-point (so
    # each X-point can have its own pull radius, e.g. from
    # compute_auto_influence_radius()).
    influence_radius = np.atleast_1d(np.asarray(influence_radius, dtype=float))
    if influence_radius.size == 1 and n_xpoints > 1:
        influence_radius = np.repeat(influence_radius, n_xpoints)
    elif influence_radius.size != n_xpoints:
        raise ValueError(
            f"influence_radius must be a scalar or have one entry per X-point "
            f"({n_xpoints}), got size {influence_radius.size}"
        )

    # cutoff_radius: same broadcasting rule as influence_radius; None means
    # "no cutoff", i.e. every taper factor below is identically 1 and the
    # method behaves exactly as before this feature was added.
    if cutoff_radius is not None:
        cutoff_radius = np.atleast_1d(np.asarray(cutoff_radius, dtype=float))
        if cutoff_radius.size == 1 and n_xpoints > 1:
            cutoff_radius = np.repeat(cutoff_radius, n_xpoints)
        elif cutoff_radius.size != n_xpoints:
            raise ValueError(
                f"cutoff_radius must be a scalar or have one entry per X-point "
                f"({n_xpoints}), got size {cutoff_radius.size}"
            )

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
            w = np.exp(-(distances[i, idx] / influence_radius[idx])**2)
            if cutoff_radius is not None:
                w *= _smooth_taper(distances[i, idx], 0.7 * cutoff_radius[idx], cutoff_radius[idx])
            rb_t[i] = rb[i] * (1 - w) + rx * w
            zb_t[i] = zb[i] * (1 - w) + zx * w

    elif combination_method == 'weighted':
        for i in range(n_points):
            weights = np.exp(-(distances[i, :] / influence_radius)**2)
            if cutoff_radius is not None:
                weights = weights * _smooth_taper(distances[i, :], 0.7 * cutoff_radius, cutoff_radius)
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
            w_nearest = np.exp(-(distances[i, idx] / influence_radius[idx])**2)
            if cutoff_radius is not None:
                w_nearest *= _smooth_taper(distances[i, idx], 0.7 * cutoff_radius[idx], cutoff_radius[idx])

            other_mask = np.arange(n_xpoints) != idx
            if np.any(other_mask):
                other_w = np.exp(-(distances[i, other_mask] / influence_radius[other_mask])**2)
                if cutoff_radius is not None:
                    other_w = other_w * _smooth_taper(distances[i, other_mask],
                                                        0.7 * cutoff_radius[other_mask],
                                                        cutoff_radius[other_mask])
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


def _min_cusp_clearance(
    rb_cusp_fine: np.ndarray,
    zb_cusp_fine: np.ndarray,
    rb_smooth_fine: np.ndarray,
    zb_smooth_fine: np.ndarray,
    s_smooth_fine: np.ndarray,
    cusp_regions: List[Dict],
    exclude_buffer: float
) -> float:
    """
    Minimum Euclidean distance between the cusp region(s)' curve points and
    the REST of the boundary (the smooth-region curve), excluding an
    arc-length buffer immediately adjacent to each cusp's own anchors.

    Motivation: `_detect_loops()` only rejects a transformed boundary that
    self-intersects. A simple (non-self-intersecting) curve can still place
    the cusp legs geometrically very close to an unrelated, distant part of
    the boundary -- this happens when the X-point sits close to or within
    the original boundary's own footprint (e.g. R inside [R_min, R_max] of
    the LCFS), rather than clearly outside it. Found in practice: this does
    not enlarge the cusp opening angle or leg length (those stayed similar
    to a known-good case), it only shrinks the clearance between the cusp
    and the plasma body sitting right above it. That clearance is exactly
    what FIXED_GS_SOLVER_POLY's cardinal ray-casting (GS_solver.h,
    ray_distance_minus_R/plus_R/minus_Z/plus_Z) needs to stay well-posed --
    a boundary that is simple but has near-zero clearance there was observed
    to crash that solver intermittently (undefined-behavior-flavored: not
    100% reproducible run to run), well before any mesh-quality warning
    would fire. See docs/reports/ for the investigation that motivated this.

    The exclusion buffer prevents flagging the trivial, expected closeness
    between a cusp's own curve and the smooth-region points immediately
    next to its anchors (those are tangent-continuous by construction and
    are supposed to be close).

    Parameters
    ----------
    rb_cusp_fine, zb_cusp_fine : np.ndarray
        Concatenated fine-sampled points of all cusp regions.
    rb_smooth_fine, zb_smooth_fine, s_smooth_fine : np.ndarray
        Fine-sampled points of the smooth (non-cusp) region and their
        arc-length parameter.
    cusp_regions : list of dict
        As returned by _define_cusp_regions (uses 's_start'/'s_end').
    exclude_buffer : float
        Extra arc-length margin (fraction of perimeter) added on each side
        of every cusp window before excluding smooth points from the
        clearance check.

    Returns
    -------
    clearance : float
        Minimum distance found (meters, same units as rb/zb). np.inf if
        either point set is empty after exclusion (nothing to compare).
    """
    if len(rb_cusp_fine) == 0 or len(rb_smooth_fine) == 0:
        return np.inf

    keep = np.ones(len(s_smooth_fine), dtype=bool)
    for region in cusp_regions:
        s_lo = (region['s_start'] - exclude_buffer) % 1.0
        s_hi = (region['s_end'] + exclude_buffer) % 1.0
        if s_lo <= s_hi:
            keep &= ~((s_smooth_fine >= s_lo) & (s_smooth_fine <= s_hi))
        else:
            keep &= ~((s_smooth_fine >= s_lo) | (s_smooth_fine <= s_hi))

    if not np.any(keep):
        return np.inf

    rr = rb_smooth_fine[keep]
    zz = zb_smooth_fine[keep]
    d2 = (rr[:, None] - rb_cusp_fine[None, :])**2 + (zz[:, None] - zb_cusp_fine[None, :])**2
    return float(np.sqrt(d2.min()))


# =============================================================================
# Automatic influence_radius_factor from X-point-to-boundary distance
# =============================================================================

# Fitted from a controlled sweep (cusp_window_size=0.05 fixed, D-shape from
# CEDRES++ Table 1 "D-shape" column, Hertout et al. 2011): for each of 9
# synthetic X-points at increasing depth below a fixed boundary point, the
# smallest influence_radius_factor that produced no self-intersection was
# recorded and fit to distance-to-boundary (excluding one non-monotonic
# outlier -- self-intersection is not perfectly monotonic in this method,
# see docs/x-point-transformation/scaling_law_influence_radius_factor.png):
# influence_radius_factor* = 1.3967 * dist_to_boundary - 0.3292 (R^2=0.975).
# Below dist_to_boundary ~= 0.236 m the fit goes negative, i.e. the smallest
# tested factor (0.05) already suffices; MIN_INFLUENCE_RADIUS_FACTOR floors
# the result there instead of extrapolating to zero/negative.
AUTO_SLOPE = 1.3967
AUTO_INTERCEPT = -0.3292
MIN_INFLUENCE_RADIUS_FACTOR = 0.05


def compute_auto_influence_radius_factor(
    rb: np.ndarray,
    zb: np.ndarray,
    x_points: List[Tuple[float, float]],
    slope: float = AUTO_SLOPE,
    intercept: float = AUTO_INTERCEPT,
    min_factor: float = MIN_INFLUENCE_RADIUS_FACTOR
) -> Tuple[List[float], List[float]]:
    """
    Automatic influence_radius_factor, one value per X-point, from each
    X-point's own distance to the nearest point of the original boundary.

    Uses the empirical linear fit above. This is specific to the D-shape
    used to derive it (it depends on a_estimate through the fitted slope,
    which was measured for one plasma size) and to cusp_window_size=0.05;
    treat it as a reasonable starting point that the adaptive loop
    (adaptive_dual_parameter/adaptive_multiparameter) still grows further if
    it turns out to self-intersect for a given geometry, not as an exact law.

    Returns
    -------
    factors : list of float
        One influence_radius_factor per X-point, same order as x_points.
    distances : list of float
        The corresponding distance-to-boundary used for each X-point (for
        diagnostics/reporting).
    """
    factors = []
    distances = []
    for (rx, zx) in x_points:
        d = float(np.sqrt((rb - rx)**2 + (zb - zx)**2).min())
        f = max(min_factor, slope * d + intercept)
        factors.append(f)
        distances.append(d)
    return factors, distances


# =============================================================================
# Core Hybrid Method
# =============================================================================

def _hybrid_bspline_linear(
    rb: np.ndarray,
    zb: np.ndarray,
    x_points: List[Tuple[float, float]],
    smoothness: float = 0.01,
    influence_radius_factor: Union[float, List[float]] = 1.1,
    cusp_window_size: float = 0.05,
    n_output_points: int = 500,
    cusp_points_density: float = 1.0,
    combination_method: str = 'nearest',
    bezier_handle_fraction: float = 0.25,
    cutoff_distance_factor: Union[None, float, List[float]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Hybrid B-spline + Bezier method for multiple X-points.

    Smooth regions of the boundary are fitted with cubic B-splines, while
    cusp regions near X-points use two tangent-matched (C1) quadratic
    Bezier legs -- anchor_start->X-point and X-point->anchor_end -- so the
    curve continues smoothly from the spline into the cusp region without
    the spurious secondary kink that a straight-line leg produces. The
    X-point itself remains an unmatched, genuine corner between the two
    legs, as physically expected.

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
        Size of Bezier-leg region around each X-point as fraction of perimeter.
    n_output_points : int
        Number of output boundary points.
    cusp_points_density : float
        Density multiplier for points in cusp regions.
    combination_method : str
        Gaussian combination method ('nearest', 'weighted', 'dominant').
    bezier_handle_fraction : float
        Fraction of the anchor-to-X-point chord length used as the Bezier
        control-point handle length (h in P1 = P0 + h*tangent_unit).
    cutoff_distance_factor : None, float, or list of float
        If given, caps how far the Gaussian pull can reach: each X-point's
        cutoff_radius = cutoff_distance_factor * (that X-point's own
        distance to the nearest point of the ORIGINAL boundary), and every
        boundary point farther than cutoff_radius from its X-point is left
        completely untouched (see _smooth_taper). None (default) disables
        this, matching the original unbounded-Gaussian behavior -- needed
        for a distant X-point, whose influence_radius (set large enough to
        pull the nearest boundary point close enough to avoid a loop) would
        otherwise also drag the FAR side of the boundary, since a plain
        Gaussian's tail never actually reaches zero.

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

    cutoff_radius = None
    if cutoff_distance_factor is not None:
        dist_to_boundary = np.array([
            np.sqrt((rb - rx)**2 + (zb - zx)**2).min() for (rx, zx) in x_points
        ])
        cutoff_radius = np.atleast_1d(np.asarray(cutoff_distance_factor, dtype=float)) * dist_to_boundary

    rb_t, zb_t, nearest_idx = _apply_gaussian_transformation(
        rb, zb, x_points, influence_radius, combination_method, cutoff_radius=cutoff_radius
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
            anchor_start = np.array([r_boundary_start, z_boundary_start])
            anchor_end = np.array([r_boundary_end, z_boundary_end])
            xpoint_vec = np.array([rx, zx])

            # --- Leg 1: anchor_start -> X-point, tangent-matched at anchor_start ---
            n_seg1 = max(int(n_cusp_per_xpoint * dist_to_start / total_dist), 2)

            tangent_start = _tangent_at_arclength(spline_r, spline_z, s_start % 1.0)
            norm_ts = np.linalg.norm(tangent_start)
            chord1_vec = xpoint_vec - anchor_start
            if norm_ts > 1e-12:
                tangent_start_unit = tangent_start / norm_ts
            else:
                tangent_start_unit = chord1_vec / max(np.linalg.norm(chord1_vec), 1e-12)
            if np.dot(tangent_start_unit, chord1_vec) < 0:
                tangent_start_unit = -tangent_start_unit

            chord1 = np.linalg.norm(chord1_vec)
            P1_1 = anchor_start + bezier_handle_fraction * chord1 * tangent_start_unit

            t1 = np.linspace(0, 1, n_seg1)
            rb_seg1, zb_seg1 = _quadratic_bezier(anchor_start, P1_1, xpoint_vec, t1)

            if s_start > s_xpoint:
                s_seg1 = np.linspace(s_start - 1.0, s_xpoint, n_seg1) % 1.0
            else:
                s_seg1 = np.linspace(s_start, s_xpoint, n_seg1)

            # --- Leg 2: X-point -> anchor_end, tangent-matched at anchor_end ---
            n_seg2 = n_cusp_per_xpoint - n_seg1 + 1

            tangent_end = _tangent_at_arclength(spline_r, spline_z, s_end % 1.0)
            norm_te = np.linalg.norm(tangent_end)
            chord2_vec = anchor_end - xpoint_vec
            if norm_te > 1e-12:
                tangent_end_unit = tangent_end / norm_te
            else:
                tangent_end_unit = chord2_vec / max(np.linalg.norm(chord2_vec), 1e-12)
            if np.dot(tangent_end_unit, chord2_vec) < 0:
                tangent_end_unit = -tangent_end_unit

            chord2 = np.linalg.norm(chord2_vec)
            P1_2 = anchor_end - bezier_handle_fraction * chord2 * tangent_end_unit

            t2 = np.linspace(0, 1, n_seg2)
            rb_seg2, zb_seg2 = _quadratic_bezier(xpoint_vec, P1_2, anchor_end, t2)

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

    # See _min_cusp_clearance()'s docstring for why this is checked in
    # addition to _detect_loops(). The exclusion buffer matches
    # cusp_window_size itself -- generous enough to skip the
    # tangent-continuous seam without hiding a genuine nearby-boundary issue.
    min_clearance = _min_cusp_clearance(
        rb_cusp_fine, zb_cusp_fine, rb_smooth_fine, zb_smooth_fine, s_smooth_fine,
        cusp_regions, exclude_buffer=cusp_window_size
    )

    info = {
        'n_xpoints': n_xpoints,
        'cusp_window_size': cusp_window_size,
        'influence_radius': influence_radius,
        'influence_radius_factor': influence_radius_factor,
        'smoothness': smoothness,
        'combination_method': combination_method,
        'bezier_handle_fraction': bezier_handle_fraction,
        'iterations': 1,
        'min_cusp_clearance': min_clearance,
        'a_estimate': a_estimate,
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
    influence_radius_factor: Union[str, float, List[float]] = 'auto',
    cusp_window_size: float = 0.05,
    n_output_points: int = 500,
    cusp_points_density: float = 1.0,
    combination_method: str = 'nearest',
    max_iterations: int = 10,
    cutoff_distance_factor: Union[None, float, List[float]] = None,
    min_clearance_factor: float = 0.6
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Multi-parameter adaptive boundary transformation for multiple X-points.

    Iteratively adjusts three parameters (influence_radius_factor,
    cusp_window_size, smoothness) to eliminate self-intersections in the
    transformed boundary AND to keep the cusp region a minimum distance away
    from the rest of the boundary (see _min_cusp_clearance()'s docstring --
    a simple, non-self-intersecting curve can still be too close to itself
    for downstream mesh classification to handle safely).

    Parameters
    ----------
    rb, zb : np.ndarray
        Original boundary coordinates.
    x_points : list of (float, float)
        X-point coordinates.
    smoothness : float
        Initial B-spline smoothing factor.
    influence_radius_factor : 'auto', float, or list of float
        Initial Gaussian influence radius relative to minor radius, one
        value per X-point. 'auto' (default) picks a starting value per
        X-point from its own distance to the boundary via
        compute_auto_influence_radius_factor() -- a single scalar is
        broadcast to every X-point (backward compatible); a list must have
        one entry per X-point. Each entry grows independently across
        iterations if its X-point's region keeps causing self-intersection.
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
    cutoff_distance_factor : None, float, or list of float
        Forwarded to _hybrid_bspline_linear -- caps the Gaussian pull to
        cutoff_distance_factor * (X-point's own distance to the boundary),
        so a distant X-point cannot drag the far side of the boundary. None
        (default) disables this.
    min_clearance_factor : float
        Minimum allowed distance between the cusp region and the rest of
        the boundary, as a fraction of a_estimate (the minor-radius
        estimate already used for influence_radius). Default 0.6 (60% of
        a_estimate -- empirically set from 3 test cases: a known-good
        X-point far outside the LCFS's own R-span gave clearance/a_estimate
        ~0.90, while two X-points placed inside that span (which crashed
        FIXED_GS_SOLVER_POLY intermittently) gave ~0.44-0.47; 0.6 sits
        between them. Treat this as a first cut, not a validated safety
        margin -- it has not yet been confirmed to eliminate the crash end
        to end). If the actual clearance (info['min_cusp_clearance'])
        falls below min_clearance_factor * info['a_estimate'], this counts
        as a rejection alongside a detected loop, and the same parameter
        growth is applied.

    Returns
    -------
    rb_new, zb_new : np.ndarray
        Transformed boundary coordinates.
    info : dict
        Diagnostic information including convergence status.
    """
    auto_distances = None
    if isinstance(influence_radius_factor, str) and influence_radius_factor == 'auto':
        influence_radius_factor, auto_distances = compute_auto_influence_radius_factor(rb, zb, x_points)

    inf_factor = np.atleast_1d(np.asarray(influence_radius_factor, dtype=float))
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
            cutoff_distance_factor=cutoff_distance_factor,
        )

        has_loop, _, _ = _detect_loops(rb_new, zb_new)
        clearance_ok = info['min_cusp_clearance'] >= min_clearance_factor * info['a_estimate']

        if not has_loop and clearance_ok:
            info['iterations'] = iteration + 1
            info['final_influence_radius_factor'] = inf_factor
            info['final_cusp_window_size'] = cusp_win
            info['final_smoothness'] = smooth
            info['converged'] = True
            info['auto_distances'] = auto_distances
            info['min_clearance_factor'] = min_clearance_factor
            return rb_new, zb_new, info

        inf_factor = inf_factor * 1.3
        cusp_win = min(cusp_win * 1.2, max_cusp_win)
        smooth *= 1.5

    info['iterations'] = max_iterations
    info['final_influence_radius_factor'] = inf_factor
    info['final_cusp_window_size'] = cusp_win
    info['final_smoothness'] = smooth
    info['converged'] = False
    info['auto_distances'] = auto_distances
    info['min_clearance_factor'] = min_clearance_factor
    return rb_new, zb_new, info


def adaptive_dual_parameter(
    rb: np.ndarray,
    zb: np.ndarray,
    x_points: List[Tuple[float, float]],
    smoothness: float = 0.01,
    influence_radius_factor: Union[str, float, List[float]] = 'auto',
    cusp_window_size: float = 0.05,
    n_output_points: int = 500,
    cusp_points_density: float = 1.0,
    combination_method: str = 'nearest',
    max_iterations: int = 10,
    cutoff_distance_factor: Union[None, float, List[float]] = None,
    min_clearance_factor: float = 0.6
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Dual-parameter adaptive boundary transformation for multiple X-points.

    Iteratively adjusts two parameters (influence_radius_factor and
    cusp_window_size) while keeping smoothness constant. This typically
    produces better curve quality than the multi-parameter method.

    Rejects a candidate transform (and keeps growing the parameters) if
    either it self-intersects OR the cusp region ends up too close to the
    rest of the boundary -- see _min_cusp_clearance()'s docstring: a simple
    curve can still be geometrically too tight for downstream mesh
    classification (FIXED_GS_SOLVER_POLY) to handle safely, which is not
    detected by the self-intersection check alone.

    Parameters
    ----------
    rb, zb : np.ndarray
        Original boundary coordinates.
    x_points : list of (float, float)
        X-point coordinates.
    smoothness : float
        B-spline smoothing factor (held constant).
    influence_radius_factor : 'auto', float, or list of float
        Initial Gaussian influence radius relative to minor radius, one
        value per X-point. 'auto' (default) picks a starting value per
        X-point from its own distance to the boundary via
        compute_auto_influence_radius_factor() -- a single scalar is
        broadcast to every X-point (backward compatible); a list must have
        one entry per X-point. Each entry grows independently across
        iterations if its X-point's region keeps causing self-intersection.
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
    cutoff_distance_factor : None, float, or list of float
        Forwarded to _hybrid_bspline_linear -- caps the Gaussian pull to
        cutoff_distance_factor * (X-point's own distance to the boundary),
        so a distant X-point cannot drag the far side of the boundary. None
        (default) disables this.
    min_clearance_factor : float
        Minimum allowed distance between the cusp region and the rest of
        the boundary, as a fraction of a_estimate (the minor-radius
        estimate already used for influence_radius). Default 0.6 (60% of
        a_estimate -- empirically set from 3 test cases: a known-good
        X-point far outside the LCFS's own R-span gave clearance/a_estimate
        ~0.90, while two X-points placed inside that span (which crashed
        FIXED_GS_SOLVER_POLY intermittently) gave ~0.44-0.47; 0.6 sits
        between them. Treat this as a first cut, not a validated safety
        margin -- it has not yet been confirmed to eliminate the crash end
        to end). If the actual clearance (info['min_cusp_clearance'])
        falls below min_clearance_factor * info['a_estimate'], this counts
        as a rejection alongside a detected loop, and the same parameter
        growth is applied.

    Returns
    -------
    rb_new, zb_new : np.ndarray
        Transformed boundary coordinates.
    info : dict
        Diagnostic information including convergence status.
    """
    auto_distances = None
    if isinstance(influence_radius_factor, str) and influence_radius_factor == 'auto':
        influence_radius_factor, auto_distances = compute_auto_influence_radius_factor(rb, zb, x_points)

    inf_factor = np.atleast_1d(np.asarray(influence_radius_factor, dtype=float))
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
            cutoff_distance_factor=cutoff_distance_factor,
        )

        has_loop, _, _ = _detect_loops(rb_new, zb_new)
        clearance_ok = info['min_cusp_clearance'] >= min_clearance_factor * info['a_estimate']

        if not has_loop and clearance_ok:
            info['iterations'] = iteration + 1
            info['final_influence_radius_factor'] = inf_factor
            info['final_cusp_window_size'] = cusp_win
            info['final_smoothness'] = smoothness
            info['converged'] = True
            info['auto_distances'] = auto_distances
            info['min_clearance_factor'] = min_clearance_factor
            return rb_new, zb_new, info

        inf_factor = inf_factor * 1.4
        cusp_win = min(cusp_win * 1.3, max_cusp_win)

    info['iterations'] = max_iterations
    info['final_influence_radius_factor'] = inf_factor
    info['final_cusp_window_size'] = cusp_win
    info['final_smoothness'] = smoothness
    info['converged'] = False
    info['auto_distances'] = auto_distances
    info['min_clearance_factor'] = min_clearance_factor
    return rb_new, zb_new, info


# =============================================================================
# Command-Line Interface
# =============================================================================

def main():
    if len(sys.argv) != 5:
        sys.stderr.write(
            "Usage: python3 xpoint_transform_c1.py "
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

    # Enabled by default for this CLI entry point (unlike the library
    # functions' own default of None): SemiFree_Solver.cpp feeds the
    # transformed boundary directly into Jt regeneration and the
    # coil-current solve, so a distant X-point's influence_radius (sized to
    # reach it without self-intersecting) must not be allowed to perturb the
    # far side of the boundary. Verified: reduces the displacement of the
    # boundary point opposite a distant X-point from 215 mm to ~3 mm without
    # affecting convergence -- see
    # docs/x-point-transformation/cutoff_locality_fix.png.
    PRODUCTION_CUTOFF_DISTANCE_FACTOR = 2.0

    if method == 1:
        rb_new, zb_new, info = adaptive_multiparameter(
            rb, zb, x_points, n_output_points=n_output_points,
            cutoff_distance_factor=PRODUCTION_CUTOFF_DISTANCE_FACTOR
        )
    else:
        rb_new, zb_new, info = adaptive_dual_parameter(
            rb, zb, x_points, n_output_points=n_output_points,
            cutoff_distance_factor=PRODUCTION_CUTOFF_DISTANCE_FACTOR
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
