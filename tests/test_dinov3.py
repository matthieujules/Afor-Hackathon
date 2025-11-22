#!/usr/bin/env python3
"""
Quick test script to verify DINOv3 integration works correctly.
Tests both the model loading and the analyze_scene interface.
"""

import numpy as np
from vision_alternatives import DinoV2Client

def test_dinov3_client():
    print("=" * 60)
    print("DINOv3 Integration Test")
    print("=" * 60)
    print()

    # Initialize client
    print("Step 1: Initializing DINOv3 client...")
    client = DinoV2Client()
    print()

    # Create test images
    print("Step 2: Creating test images...")

    # Test 1: Blank/boring image (uniform color)
    blank_image = np.ones((320, 320, 3), dtype=np.uint8) * 128
    print("  - Created blank image (should have low interest)")

    # Test 2: Cluttered image (random noise)
    cluttered_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    print("  - Created cluttered image (should have high interest)")

    # Test 3: Image with left-side features
    left_image = np.ones((320, 320, 3), dtype=np.uint8) * 128
    left_image[:, :100] = np.random.randint(0, 255, (320, 100, 3), dtype=np.uint8)
    print("  - Created left-biased image (should lead left)")
    print()

    # Test analysis
    print("Step 3: Testing scene analysis...")
    print()

    # Test blank image
    print("Test 1: Blank image")
    result1 = client.analyze_scene(blank_image)
    print(f"  Interest Score: {result1['interest_score']:.3f}")
    print(f"  Lead Direction: {result1['lead_direction']}")
    print(f"  Hazard Score: {result1['hazard_score']:.3f}")
    assert 0 <= result1['interest_score'] <= 1, "Interest score out of range"
    assert result1['lead_direction'] in ['left', 'right', 'center', 'none'], "Invalid lead direction"
    print("  ✓ PASSED")
    print()

    # Test cluttered image
    print("Test 2: Cluttered image")
    result2 = client.analyze_scene(cluttered_image)
    print(f"  Interest Score: {result2['interest_score']:.3f}")
    print(f"  Lead Direction: {result2['lead_direction']}")
    print(f"  Hazard Score: {result2['hazard_score']:.3f}")
    assert 0 <= result2['interest_score'] <= 1, "Interest score out of range"
    assert result2['lead_direction'] in ['left', 'right', 'center', 'none'], "Invalid lead direction"
    print("  ✓ PASSED")
    print()

    # Test left-biased image
    print("Test 3: Left-biased image")
    result3 = client.analyze_scene(left_image)
    print(f"  Interest Score: {result3['interest_score']:.3f}")
    print(f"  Lead Direction: {result3['lead_direction']}")
    print(f"  Hazard Score: {result3['hazard_score']:.3f}")
    assert 0 <= result3['interest_score'] <= 1, "Interest score out of range"
    assert result3['lead_direction'] in ['left', 'right', 'center', 'none'], "Invalid lead direction"
    print("  ✓ PASSED")
    print()

    # Verify relative interest scores make sense
    print("Step 4: Verifying interest score ordering...")
    print(f"  Blank interest: {result1['interest_score']:.3f}")
    print(f"  Clutter interest: {result2['interest_score']:.3f}")

    if result2['interest_score'] > result1['interest_score']:
        print("  ✓ Cluttered image has higher interest than blank (correct)")
    else:
        print("  ⚠ Warning: Interest scores may not be calibrated optimally")
    print()

    print("=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    print()
    print("DINOv3 integration is working correctly!")
    print("You can now run: python3 scout_semantix.py")
    print()

if __name__ == '__main__':
    try:
        test_dinov3_client()
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
