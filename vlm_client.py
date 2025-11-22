import os
import time
import base64
import io
from PIL import Image
import numpy as np
import requests
import json
import re

class VLMClient:
    def __init__(self, provider="gemini", model="gemini-1.5-flash", api_key=None):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key and provider != "mock":
            print("WARNING: No API key found. Defaulting to MOCK mode.")
            self.provider = "mock"

    def analyze_scene(self, image_rgb):
        """
        Analyzes the image and returns a dictionary with:
        - hazard_score: 0.0 - 1.0 (Immediate danger)
        - interest_score: 0.0 - 1.0 (Visual complexity/clutter)
        - lead_direction: "left", "right", "center", "none" (Where does the interest go?)
        """
        if self.provider == "mock":
            return self._mock_analysis(image_rgb)
        elif self.provider == "gemini":
            return self._gemini_analysis(image_rgb)
        elif self.provider == "openai":
            return self._openai_analysis(image_rgb)
        else:
            return self._mock_analysis(image_rgb)

    def _mock_analysis(self, image_rgb):
        # Mock logic:
        # Red = Hazard
        # Clutter (high variance) = Interest
        
        r = image_rgb[:, :, 0]
        g = image_rgb[:, :, 1]
        b = image_rgb[:, :, 2]
        
        # Hazard (Red)
        is_red = (r > 150) & (r > g * 1.5) & (r > b * 1.5)
        red_ratio = np.sum(is_red) / image_rgb.size
        hazard_score = 0.9 if red_ratio > 0.005 else 0.1
        
        # Interest (Edge detection / Variance)
        gray = np.mean(image_rgb, axis=2)
        variance = np.var(gray)
        interest_score = min(variance / 1000.0, 1.0) # Normalize somewhat
        
        # Lead direction (Mock: random or based on brightness gradient)
        left_bright = np.mean(gray[:, :gray.shape[1]//3])
        right_bright = np.mean(gray[:, 2*gray.shape[1]//3:])
        
        if left_bright > right_bright + 20:
            lead = "left"
        elif right_bright > left_bright + 20:
            lead = "right"
        else:
            lead = "center"

        return {
            "hazard_score": hazard_score,
            "interest_score": interest_score,
            "lead_direction": lead
        }

    def _encode_image(self, image_rgb):
        img = Image.fromarray(image_rgb)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _gemini_analysis(self, image_rgb):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {'Content-Type': 'application/json'}
        
        b64_image = self._encode_image(image_rgb)
        
        prompt = """
        Analyze this image from a warehouse scout robot. Return a JSON object with:
        1. "hazard_score" (0.0 to 1.0): Probability of immediate danger (fire, chemical spill, sharp objects).
        2. "interest_score" (0.0 to 1.0): Visual complexity or "clutter" that implies more to see.
        3. "lead_direction" ("left", "right", "center", "none"): If a trail, cable, or object is cut off by the edge, which way does it go?
        
        Example: {"hazard_score": 0.1, "interest_score": 0.8, "lead_direction": "left"}
        Return ONLY the JSON.
        """
        
        data = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64_image
                    }}
                ]
            }],
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            text = result['candidates'][0]['content']['parts'][0]['text']
            return json.loads(text)
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return {"hazard_score": 0.1, "interest_score": 0.1, "lead_direction": "none"}

    def _openai_analysis(self, image_rgb):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        b64_image = self._encode_image(image_rgb)
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image. Return JSON with 'hazard_score' (0-1), 'interest_score' (0-1), and 'lead_direction' (left/right/center/none)."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            }
                        }
                    ]
                }
            ],
            "response_format": { "type": "json_object" },
            "max_tokens": 100
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            return json.loads(content)
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return {"hazard_score": 0.1, "interest_score": 0.1, "lead_direction": "none"}
