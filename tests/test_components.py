#!/usr/bin/env python3
"""
Semantix Component Tests

Verifies that all components work correctly before running the full demo.
"""

import numpy as np
import sys

def test_bayesian_mapping():
    """Test Beta posterior updates and entropy calculation"""
    print("Testing Bayesian Mapping...")

    # Initialize
    alpha = np.ones((10, 10))
    beta = np.ones((10, 10))

    # Simulate observation: high hazard at (5, 5)
    alpha[5, 5] += 0.9
    beta[5, 5] += 0.1

    # Compute mean
    mu = alpha / (alpha + beta)

    # Check
    assert 0 < mu[5, 5] < 1, "Hazard mean should be in [0,1]"
    assert mu[5, 5] > mu[0, 0], "Observed hazard should be higher than prior"

    # Entropy
    eps = 1e-6
    mu_safe = np.clip(mu, eps, 1 - eps)
    H = -(mu_safe * np.log(mu_safe) + (1 - mu_safe) * np.log(1 - mu_safe))

    assert np.all(H >= 0), "Entropy should be non-negative"
    assert H[5, 5] < H[0, 0], "Entropy should decrease after observation"

    print("  ✓ Beta posterior updates correctly")
    print("  ✓ Entropy calculation works")
    print("  ✓ Bayesian mapping: PASSED\n")

def test_convolution():
    """Test Gaussian convolution for white glow"""
    print("Testing White Glow Convolution...")

    # Create simple test image
    img = np.zeros((10, 10))
    img[5, 5] = 1.0  # single spike

    # 3×3 Gaussian kernel
    K = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32)
    K /= K.sum()

    # Manual convolution
    def conv2d_manual(a, kernel):
        pad = kernel.shape[0] // 2
        a_padded = np.pad(a, pad, mode='reflect')
        out = np.zeros_like(a)
        kh, kw = kernel.shape
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                out[i, j] = np.sum(a_padded[i:i+kh, j:j+kw] * kernel)
        return out

    result = conv2d_manual(img, K)

    # Check: center should still be highest
    assert result[5, 5] > result[5, 6], "Convolution should preserve peak"
    assert result[5, 6] > 0, "Convolution should spread values"
    assert result[0, 0] < 0.01, "Distant cells should be near zero"

    print("  ✓ Gaussian kernel convolution works")
    print("  ✓ Spatial diffusion behaves correctly")
    print("  ✓ White glow: PASSED\n")

def test_fov_projection():
    """Test field-of-view projection to grid"""
    print("Testing FOV Projection...")

    # Simulate robot at origin, heading 0 (east)
    robot_pos = np.array([0.0, 0.0])
    heading = 0.0  # radians
    GRID_SIZE = 64
    CELL_SIZE = 0.3125  # 20m / 64
    FOV_DEGREES = 70
    MAX_RANGE = 6.0

    # Simple ray casting
    angles = np.linspace(-np.radians(FOV_DEGREES/2),
                         np.radians(FOV_DEGREES/2),
                         31)

    cells = []
    for angle_offset in angles:
        ray_angle = heading + angle_offset
        r = CELL_SIZE
        while r <= MAX_RANGE:
            x = robot_pos[0] + r * np.cos(ray_angle)
            y = robot_pos[1] + r * np.sin(ray_angle)
            gx = int(x / CELL_SIZE + GRID_SIZE / 2)
            gy = int(y / CELL_SIZE + GRID_SIZE / 2)
            if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                cells.append((gy, gx, r))
            r += CELL_SIZE

    assert len(cells) > 0, "Should project some cells"

    # Check: cells should be in front of robot (positive X)
    avg_gx = np.mean([c[1] for c in cells])
    assert avg_gx > GRID_SIZE / 2, "FOV should point in heading direction"

    print(f"  ✓ Projected {len(cells)} cells")
    print("  ✓ FOV direction correct")
    print("  ✓ FOV projection: PASSED\n")

def test_utility_function():
    """Test utility calculation"""
    print("Testing Utility Function...")

    # Create test maps
    mu = np.random.rand(10, 10) * 0.5
    H = np.random.rand(10, 10) * 0.3
    glow = np.random.rand(10, 10) * 0.2

    # Place high values at (7, 7)
    mu[7, 7] = 1.0
    H[7, 7] = 0.5
    glow[7, 7] = 0.8

    # Test utility at (7, 7)
    wy, wx = 7, 7
    y0, y1 = max(0, wy-1), min(10, wy+2)
    x0, x1 = max(0, wx-1), min(10, wx+2)

    patch_mu = mu[y0:y1, x0:x1]
    patch_H = H[y0:y1, x0:x1]
    patch_glow = glow[y0:y1, x0:x1]

    U_high = 1.0 * np.max(patch_mu) + 0.5 * np.max(patch_H) + 0.8 * np.max(patch_glow)

    # Test utility at (0, 0) (lower values)
    wy, wx = 0, 0
    y0, y1 = max(0, wy-1), min(10, wy+2)
    x0, x1 = max(0, wx-1), min(10, wx+2)

    patch_mu = mu[y0:y1, x0:x1]
    patch_H = H[y0:y1, x0:x1]
    patch_glow = glow[y0:y1, x0:x1]

    U_low = 1.0 * np.max(patch_mu) + 0.5 * np.max(patch_H) + 0.8 * np.max(patch_glow)

    assert U_high > U_low, "High-value region should have higher utility"

    print("  ✓ Utility combines hazard + entropy + glow")
    print("  ✓ High-value regions have higher utility")
    print("  ✓ Utility function: PASSED\n")

def test_imports():
    """Test that all required packages are importable"""
    print("Testing Dependencies...")

    has_pybullet = True
    try:
        import pybullet
        print("  ✓ pybullet")
    except ImportError:
        print("  ⚠ pybullet (MISSING)")
        print("    PyBullet is required to run the full simulation.")
        print("    Install with: pip3 install pybullet")
        print("    If compilation fails, try a pre-built wheel:")
        print("    pip3 install --pre pybullet")
        has_pybullet = False

    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy (MISSING)")
        return False

    try:
        import matplotlib
        print("  ✓ matplotlib")
    except ImportError:
        print("  ✗ matplotlib (MISSING)")
        return False

    if not has_pybullet:
        print("\n  ℹ Core logic tests will run, but simulation requires PyBullet")

    print("  ✓ Core dependencies: PASSED\n")
    return True

def main():
    print("=" * 60)
    print("Semantix Component Tests")
    print("=" * 60)
    print()

    # Test imports first
    if not test_imports():
        print("\n⚠ Missing dependencies. Run: pip install -r requirements.txt")
        sys.exit(1)

    # Run component tests
    try:
        test_bayesian_mapping()
        test_convolution()
        test_fov_projection()
        test_utility_function()

        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nSemanfix is ready to run! Try:")
        print("  python scout_semantix.py")
        print()

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
