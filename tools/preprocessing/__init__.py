"""Preprocessing helpers with lazy imports for optional dependencies.

Plotting is only needed by :func:`visualize_nii_slices`; keeping it lazy lets
resize and dimension utilities run in minimal training environments.
"""

__all__ = ['visualize_nii_slices', 'print_dimensions']


def visualize_nii_slices(*args, **kwargs):
    from .slices import visualize_nii_slices as _visualize
    return _visualize(*args, **kwargs)


def print_dimensions(*args, **kwargs):
    from .dimensions import print_dimensions as _print_dimensions
    return _print_dimensions(*args, **kwargs)
