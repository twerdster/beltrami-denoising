"""Non-local means with the SSVM 2011 comparator's patch weighting.

The implementation evaluates each search displacement over an image region.
It uses symmetric patch padding, candidates within the real image domain,
and self-weight equal to the largest neighboring weight.
"""

import numpy as np
from scipy.ndimage import correlate


def patch_kernel(radius):
    if int(radius) != radius or radius < 1:
        raise ValueError("similarity radius must be a positive integer")
    radius = int(radius)
    kernel = np.zeros((2*radius+1, 2*radius+1), dtype=np.float64)
    for distance in range(1, radius+1):
        kernel[radius-distance:radius+distance+1,
               radius-distance:radius+distance+1] += 1 / (2*distance+1)**2
    kernel /= radius
    return kernel / kernel.sum(axis=0).sum()


def nonlocal_means(image, search_radius=5, patch_radius=2, bandwidth=20.0):
    """Denoise a grayscale array using the historical comparison protocol."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or not image.size or not np.isfinite(image).all():
        raise ValueError("image must be a nonempty finite grayscale array")
    if int(search_radius) != search_radius or search_radius < 0:
        raise ValueError("search_radius must be a nonnegative integer")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be positive and finite")
    kernel = patch_kernel(patch_radius)
    search_radius, patch_radius = int(search_radius), int(patch_radius)
    height, width = image.shape
    padded = np.pad(image, patch_radius, mode="symmetric")
    averages = np.zeros_like(image)
    weights = np.zeros_like(image)
    largest = np.zeros_like(image)
    for dr in range(-search_radius, search_radius+1):
        for dc in range(-search_radius, search_radius+1):
            if dr == dc == 0:
                continue
            r0, r1 = max(0, -dr), min(height, height-dr)
            c0, c1 = max(0, -dc), min(width, width-dc)
            if r1 <= r0 or c1 <= c0:
                continue
            target = (slice(r0, r1), slice(c0, c1))
            a = padded[r0:r1+2*patch_radius, c0:c1+2*patch_radius]
            b = padded[r0+dr:r1+dr+2*patch_radius, c0+dc:c1+dc+2*patch_radius]
            differences = (a-b)**2
            distances = correlate(differences, kernel, mode="constant")
            distances = distances[patch_radius:-patch_radius, patch_radius:-patch_radius]
            weight = np.exp(-distances / bandwidth**2)
            np.maximum(largest[target], weight, out=largest[target])
            weights[target] += weight
            averages[target] += weight * image[r0+dr:r1+dr, c0+dc:c1+dc]
    averages += largest * image
    weights += largest
    return np.divide(averages, weights, out=image.copy(), where=weights > 0)
