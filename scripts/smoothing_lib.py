#!/usr/bin/env python3
"""
Trajectory smoothing utilities using SciPy filters

Provides multiple smoothing methods for camera trajectory stabilization:
- Moving Average: Simple averaging over a window
- Savitzky-Golay: Preserves peaks and valleys while smoothing
- Gaussian: Smooth blur-like smoothing
- Spline: Very smooth cubic spline interpolation
"""

import inspect

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline


def smooth_moving_average(trajectory, window):
    """
    Simple moving average smoothing

    Args:
        trajectory: Nx3 array of [dx, dy, da] transforms
        window: Window size for averaging (frames)

    Returns:
        Smoothed trajectory array
    """
    # Ensure window is odd and at least 1
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1

    smoothed = np.copy(trajectory).astype(np.float64)
    half_window = window // 2

    for i in range(len(trajectory)):
        # Compute bounds for averaging window
        start = max(0, i - half_window)
        end = min(len(trajectory), i + half_window + 1)

        # Average over window
        smoothed[i] = np.mean(trajectory[start:end], axis=0)

    return smoothed


def smooth_savitzky_golay(trajectory, window, polyorder=3):
    """
    Savitzky-Golay filter - preserves peaks and valleys

    Best for trajectories with intentional motion that should be preserved.

    Args:
        trajectory: Nx3 array of [dx, dy, da] transforms
        window: Window size for the filter (must be odd)
        polyorder: Polynomial order (default 3, must be less than window)

    Returns:
        Smoothed trajectory array
    """
    # Ensure window is odd and valid
    window = max(5, int(window))
    if window % 2 == 0:
        window += 1

    # Ensure polyorder is valid
    polyorder = min(polyorder, window - 1)

    smoothed = np.copy(trajectory).astype(np.float64)

    # Apply filter to each column
    for col in range(trajectory.shape[1]):
        smoothed[:, col] = savgol_filter(
            trajectory[:, col],
            window_length=window,
            polyorder=polyorder,
            mode='nearest'
        )

    return smoothed


def smooth_gaussian(trajectory, sigma):
    """
    Gaussian filter - smooth blur-like smoothing

    Provides very smooth results, good for removing all jitter.

    Args:
        trajectory: Nx3 array of [dx, dy, da] transforms
        sigma: Standard deviation for Gaussian kernel (higher = smoother)

    Returns:
        Smoothed trajectory array
    """
    sigma = max(0.1, float(sigma))
    smoothed = np.copy(trajectory).astype(np.float64)

    # Apply Gaussian filter to each column
    for col in range(trajectory.shape[1]):
        smoothed[:, col] = gaussian_filter1d(
            trajectory[:, col],
            sigma=sigma,
            mode='nearest'
        )

    return smoothed


def smooth_spline(trajectory, smoothing_factor=None):
    """
    Cubic spline interpolation - very smooth

    Produces the smoothest results, good for removing all shake.

    Args:
        trajectory: Nx3 array of [dx, dy, da] transforms
        smoothing_factor: Spline smoothing parameter (None for auto)
                         Higher values = smoother (try 1.0 to 100.0)

    Returns:
        Smoothed trajectory array
    """
    n_points = trajectory.shape[0]
    x = np.arange(n_points)
    smoothed = np.copy(trajectory).astype(np.float64)

    # Apply spline to each column
    for col in range(trajectory.shape[1]):
        spline = UnivariateSpline(x, trajectory[:, col], s=smoothing_factor)
        smoothed[:, col] = spline(x)

    return smoothed


# Method registry
SMOOTHING_METHODS = {
    'moving_average': smooth_moving_average,
    'savgol': smooth_savitzky_golay,
    'gaussian': smooth_gaussian,
    'spline': smooth_spline,
}


def apply_smoothing(trajectory, method='moving_average', **params):
    """
    Unified interface for applying trajectory smoothing

    Args:
        trajectory: Nx3 array of [dx, dy, da] transforms
        method: Smoothing method name (see SMOOTHING_METHODS)
        **params: Method-specific parameters

    Returns:
        Smoothed trajectory array

    Raises:
        ValueError: If method is unknown

    Examples:
        # Moving average with window of 30 frames
        smooth = apply_smoothing(traj, 'moving_average', window=30)

        # Savitzky-Golay with window of 51 and polynomial order 3
        smooth = apply_smoothing(traj, 'savgol', window=51, polyorder=3)

        # Gaussian with sigma of 10
        smooth = apply_smoothing(traj, 'gaussian', sigma=10)

        # Spline with smoothing factor of 50
        smooth = apply_smoothing(traj, 'spline', smoothing_factor=50)
    """
    if method not in SMOOTHING_METHODS:
        raise ValueError(f"Unknown smoothing method: {method}. "
                        f"Available: {list(SMOOTHING_METHODS.keys())}")

    # Filter params to only those accepted by the chosen method
    func = SMOOTHING_METHODS[method]
    sig = inspect.signature(func)
    accepted = set(sig.parameters.keys()) - {'trajectory'}
    filtered_params = {k: v for k, v in params.items() if k in accepted}

    return func(trajectory, **filtered_params)


if __name__ == '__main__':
    # Test the smoothing methods
    print("Testing trajectory smoothing library...")

    # Create a synthetic noisy trajectory
    np.random.seed(42)
    n_frames = 100

    # Smooth base trajectory with added noise
    t = np.linspace(0, 4 * np.pi, n_frames)
    base_x = np.sin(t) * 10
    base_y = np.cos(t) * 5
    base_a = np.sin(t * 0.5) * 0.1

    # Add noise
    noise_scale = 2.0
    trajectory = np.column_stack([
        base_x + np.random.randn(n_frames) * noise_scale,
        base_y + np.random.randn(n_frames) * noise_scale,
        base_a + np.random.randn(n_frames) * 0.02
    ])

    print(f"\nOriginal trajectory shape: {trajectory.shape}")
    print(f"Original trajectory mean: {trajectory.mean(axis=0)}")
    print(f"Original trajectory std: {trajectory.std(axis=0)}")

    # Test each method
    methods_to_test = [
        ('moving_average', {'window': 15}),
        ('savgol', {'window': 15, 'polyorder': 3}),
        ('gaussian', {'sigma': 5}),
        ('spline', {'smoothing_factor': 50}),
    ]

    for method, params in methods_to_test:
        smoothed = apply_smoothing(trajectory, method, **params)
        print(f"\n{method} with {params}:")
        print(f"  Shape: {smoothed.shape}")
        print(f"  Mean: {smoothed.mean(axis=0)}")
        print(f"  Std: {smoothed.std(axis=0)}")

        # Verify smoothing reduced variance
        assert smoothed.std(axis=0).mean() < trajectory.std(axis=0).mean(), \
            f"{method} did not reduce variance"

    print("\n✓ All smoothing methods working correctly!")
