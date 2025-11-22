#!/usr/bin/env python3
"""
Quick test script to verify WebSocket communication works.
Tests the dashboard without running full PyBullet simulation.
"""

import asyncio
import websockets
import json
import base64
import numpy as np
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def numpy_to_base64_png(img_array):
    """Convert numpy array to base64 PNG string"""
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)

    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=100)
    ax.axis('off')

    if len(img_array.shape) == 2:
        ax.imshow(img_array, cmap='hot')
    else:
        ax.imshow(img_array)

    plt.tight_layout(pad=0)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

async def test_dashboard():
    """Send test data to dashboard"""
    print("Connecting to dashboard...")

    try:
        async with websockets.connect("ws://localhost:8080/ws/data") as websocket:
            print("Connected! Sending test frames...")

            for i in range(10):
                # Create test image with changing pattern
                test_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
                test_image[i*30:(i+1)*30, :] = [255, 0, 0]  # Red stripe

                # Create test data
                data = {
                    'timestamp': asyncio.get_event_loop().time(),
                    'live_feed': numpy_to_base64_png(test_image),
                    'snapshot': numpy_to_base64_png(test_image),
                    'metrics': {
                        'interest_score': 0.5 + i * 0.05,
                        'lead_direction': ['left', 'right', 'center'][i % 3],
                        'current_heading': i * 36.0,
                        'decision': f'Test Frame {i}',
                        'entropy_score': 10.0 + i,
                        'glow_score': 5.0 + i * 0.5
                    }
                }

                # Add test visualizations
                test_vis = np.random.rand(14, 14, 3)
                test_heatmap = np.random.rand(14, 14)

                data['pca_vis'] = numpy_to_base64_png(test_vis)
                data['attention_map'] = numpy_to_base64_png(test_heatmap)
                data['similarity_map'] = numpy_to_base64_png(test_heatmap)

                # Send data
                await websocket.send(json.dumps(data))
                print(f"  Sent frame {i+1}/10")

                await asyncio.sleep(0.5)

            print("\nTest complete! Check browser at http://localhost:8080")
            print("Press Ctrl+C to exit")

            # Keep connection alive
            while True:
                await asyncio.sleep(1)

    except websockets.exceptions.ConnectionRefused:
        print("\nERROR: Could not connect to dashboard!")
        print("Make sure the web server is running:")
        print("  python3 src/web_dashboard.py")
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    print("="*60)
    print("Dashboard Test Script")
    print("="*60)
    print("\nMake sure web server is running first:")
    print("  python3 src/web_dashboard.py")
    print("\nThen open browser to: http://localhost:8080")
    print("\nStarting in 3 seconds...")
    print("="*60 + "\n")

    import time
    time.sleep(3)

    asyncio.run(test_dashboard())
