#!/usr/bin/env python3
"""
Mock VLM Server for Semantix Testing

Provides a simple HTTP API that mimics a real VLM hazard scorer.
Uses basic image analysis (red color detection) to return hazard scores.

Usage:
    python mock_vlm_server.py

Then run Semantix with:
    USE_VLM=1 VLM_ENDPOINT=http://localhost:8000/score python scout_semantix.py
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import io
from PIL import Image
import numpy as np

PORT = 8000

class VLMHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/score':
            # Parse multipart form data (simplified)
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                # Extract image (rough parsing, assumes single image upload)
                # In production, use proper multipart parsing
                boundary_idx = post_data.find(b'\r\n\r\n')
                if boundary_idx != -1:
                    image_data = post_data[boundary_idx+4:]
                    # Find end boundary
                    end_idx = image_data.find(b'\r\n--')
                    if end_idx != -1:
                        image_data = image_data[:end_idx]

                    # Load image
                    img = Image.open(io.BytesIO(image_data))
                    img_array = np.array(img)

                    # Simple hazard detection: measure "redness"
                    # High red channel, low green/blue → hazard
                    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                        r = img_array[:, :, 0].astype(float)
                        g = img_array[:, :, 1].astype(float)
                        b = img_array[:, :, 2].astype(float)

                        # Redness metric: R - 0.5*(G+B)
                        redness = np.mean(r - 0.5 * (g + b))

                        # Normalize to [0, 1]
                        score = np.clip(redness / 128.0, 0.0, 1.0)

                        print(f"VLM Request: redness={redness:.1f} → score={score:.3f}")
                    else:
                        score = 0.1  # grayscale fallback

                    # Return JSON response
                    response = {'score': float(score)}
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
                    return

            except Exception as e:
                print(f"Error processing image: {e}")
                pass

            # Error fallback
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "invalid request"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging
        pass

def run_server():
    server = HTTPServer(('localhost', PORT), VLMHandler)
    print("=" * 60)
    print("Mock VLM Server Running")
    print("=" * 60)
    print(f"Listening on http://localhost:{PORT}/score")
    print("")
    print("This server simulates a VLM by detecting red pixels in images.")
    print("More red → higher hazard score")
    print("")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()

if __name__ == '__main__':
    run_server()
