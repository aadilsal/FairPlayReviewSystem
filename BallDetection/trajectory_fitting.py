"""Coordinate system (image-space, pixels):
- Origin at top-left.
- +x to the right, +y down.
- Gravity acts in +y.
- Delivery direction (bowler -> batsman) is +x by convention.
"""

import numpy as np
import scipy.interpolate
from sklearn.linear_model import RANSACRegressor
from typing import List, Tuple, Optional

def fit_trajectory(positions: List[Tuple[float, float]]) -> Optional[Tuple[List[Tuple[float, float]], List[int]]]:
    """
    Fit a quadratic curve for Y and a linear curve for X using RANSAC.
    Args:
        positions: List of (x, y) coordinates.
    Returns:
        tuple of (fitted_points, inlier_indices)
    """
    if len(positions) < 5:
        return None
    
    pos_arr = np.array(positions)
    t = np.arange(len(positions)).reshape(-1, 1)
    
    # 1. Fit X (Linear)
    x = pos_arr[:, 0]
    
    from sklearn.linear_model import LinearRegression
    
    try:
        ransac_x = RANSACRegressor(min_samples=min(5, len(positions)), max_trials=100)
        ransac_x.fit(t, x)
        fitted_x = ransac_x.predict(t)
        inliers_x = ransac_x.inlier_mask_
    except Exception:
        # Fallback to standard Least Squares
        lr_x = LinearRegression().fit(t, x)
        fitted_x = lr_x.predict(t)
        inliers_x = np.ones(len(positions), dtype=bool)
    
    # 2. Fit Y (Quadratic)
    y = pos_arr[:, 1]
    T = np.column_stack([t, t**2])
    
    try:
        ransac_y = RANSACRegressor(min_samples=min(5, len(positions)), max_trials=100)
        ransac_y.fit(T, y)
        fitted_y = ransac_y.predict(T)
        inliers_y = ransac_y.inlier_mask_
    except Exception:
        # Fallback to standard Least Squares
        lr_y = LinearRegression().fit(T, y)
        fitted_y = lr_y.predict(T)
        inliers_y = np.ones(len(positions), dtype=bool)
    
    # Combined inliers
    combined_inliers = inliers_x & inliers_y
    
    fitted_points = [(float(fx), float(fy)) for fx, fy in zip(fitted_x, fitted_y)]
    inlier_indices = np.where(combined_inliers)[0].tolist()
    
    return fitted_points, inlier_indices

def detect_events_from_trajectory(positions: List[Tuple[float, float]], velocities: List[Tuple[float, float]]) -> dict:
    """
    Detect bounce and impact events.
    """
    events = {'bounce': [], 'impact': []}
    
    if len(velocities) < 3:
        return events
        
    for i in range(1, len(velocities) - 1):
        vy_prev = velocities[i-1][1]
        vy_curr = velocities[i][1]
        
        # Bounce: vertical velocity sign change (Down to Up)
        if vy_prev > 2.0 and vy_curr < -1.0:
            events['bounce'].append(i)
            
        # Impact: sudden velocity magnitude drop
        v_mag_prev = np.hypot(velocities[i-1][0], velocities[i-1][1])
        v_mag_curr = np.hypot(velocities[i][0], velocities[i][1])
        if v_mag_prev > 10.0 and v_mag_curr < v_mag_prev * 0.5:
             events['impact'].append(i)
             
    return events
