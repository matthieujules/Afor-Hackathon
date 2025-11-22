"""
Vision Alternatives for Semantix
================================

This file demonstrates how to integrate fast, local vision models (DinoV2, CLIP, Saliency)
into the Semantix "Search" logic.

The Core Interface:
The 'Search' logic (scout_semantix.py) only cares about two signals:
1. interest_score (0.0 - 1.0): "Is this view complex/important?"
2. lead_direction (left/right/center/none): "Does the interest continue off-screen?"

Any model that can output these two values can drive the Panopticon.
"""

import numpy as np

class BaseVisionClient:
    def analyze_scene(self, image_rgb):
        """
        Input: RGB Image (H, W, 3)
        Output: {
            'interest_score': float,
            'lead_direction': str
        }
        """
        raise NotImplementedError

# ==============================================================================
# OPTION 1: DINOv3 (The "Texture/Object" Expert)
# Best for: Finding objects in clutter without knowing what they are.
# Uses Meta's DINOv3 - latest self-supervised vision transformer
# ==============================================================================
class DinoV2Client(BaseVisionClient):
    """
    DINO Vision Client (supports both DINOv2 and DINOv3)

    By default uses DINOv2 (no authentication required).
    Set USE_DINOV3=1 environment variable to use DINOv3 (requires Hugging Face auth).
    Falls back to numpy-based edge detection if model loading fails.
    """
    def __init__(self):
        import os
        use_dinov3 = os.getenv('USE_DINOV3', '0') == '1'

        if use_dinov3:
            self._load_dinov3()
        else:
            self._load_dinov2()

    def _load_dinov2(self):
        """Load DINOv2 from torch.hub (no authentication required)"""
        print("Loading DINOv2 model (this may take a moment)...")
        self.model = None
        self.processor = None
        self.device = None
        self.model_type = 'dinov2'

        try:
            import torch
            import torchvision.transforms as T
            import ssl

            # Detect best available device
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')

            print(f"  Device: {self.device}")

            # Workaround for SSL certificate issues on macOS
            # This is safe for loading models from trusted sources like torch.hub
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context

            # Load DINOv2 from torch.hub
            print(f"  Loading: DINOv2 ViT-S/14 (torch.hub)")
            # Use skip_validation=True to avoid GitHub API rate limits
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                                       skip_validation=True, trust_repo=True)
            self.model = self.model.to(self.device)
            self.model.eval()

            # Create transform for DINOv2
            self.transform = T.Compose([
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            print(f"  Status: DINOv2 ViT-S/14 ready on {self.device}")
            print(f"  Features: 384-dim embeddings, 16x16 patch grid")

        except ImportError as e:
            print(f"  Warning: PyTorch not installed - {e}")
            print(f"  Falling back to numpy-based edge detection")
            print(f"  Install with: pip install torch torchvision")
        except Exception as e:
            print(f"  Warning: Failed to load DINOv2 model - {e}")
            print(f"  Falling back to numpy-based edge detection")
            print(f"  (Model downloads automatically on first run)")

        if self.model is None:
            print("  Mode: FALLBACK (numpy edge detection)")

    def _load_dinov3(self):
        """Load DINOv3 from Hugging Face (requires authentication)"""
        print("Loading DINOv3 model (this may take a moment)...")
        self.model = None
        self.processor = None
        self.device = None
        self.model_type = 'dinov3'

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            # Detect best available device
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')

            print(f"  Device: {self.device}")

            # Load DINOv3 model from Hugging Face
            model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"

            print(f"  Loading: {model_name}")
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

            # Enable eager attention to support attention map extraction
            self.model.set_attn_implementation('eager')

            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"  Status: DINOv3 ViT-S/16 ready on {self.device}")
            print(f"  Features: 384-dim embeddings per patch")

        except ImportError as e:
            print(f"  Warning: Required libraries not installed - {e}")
            print(f"  Falling back to DINOv2...")
            self._load_dinov2()
            return
        except Exception as e:
            print(f"  Warning: Failed to load DINOv3 model - {e}")
            print(f"  (Requires Hugging Face authentication - see DINOV3_SETUP.md)")
            print(f"  Falling back to DINOv2...")
            self._load_dinov2()
            return

        if self.model is None:
            print("  Mode: FALLBACK (numpy edge detection)")

    def analyze_scene(self, image_rgb):
        """
        Analyze scene using DINO (v2 or v3) or fallback to edge detection.

        Returns:
            dict: {
                'interest_score': float (0-1),
                'lead_direction': str ('left', 'right', 'center', 'none'),
                'hazard_score': float (0-1, always 0.1 for DINO)
            }
        """
        # Fallback if model failed to load (no PyTorch or no internet)
        if self.model is None:
            return self._fallback_analysis(image_rgb)

        if self.model_type == 'dinov3':
            return self._analyze_dinov3(image_rgb)
        else:
            return self._analyze_dinov2(image_rgb)

    def _analyze_dinov2(self, image_rgb):
        """Analyze using DINOv2 model"""
        import torch
        from PIL import Image

        # 1. Preprocess image
        img_pil = Image.fromarray(image_rgb)
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        # 2. Run DINOv2 inference
        with torch.no_grad():
            # DINOv2 forward_features returns dict with patch tokens
            features_dict = self.model.forward_features(img_tensor)
            patch_tokens = features_dict["x_norm_patchtokens"]  # Shape: (1, 256, 384)

            # For ViT-S/14: 224/14 = 16, so 16x16 = 256 patches

            # === INTEREST SCORE: Measure visual complexity ===
            variance = torch.var(patch_tokens, dim=1).mean().item()

            # Debug: Print variance to help with calibration
            print(f"    [DINOv2] Raw variance: {variance:.6f}")

            # Empirically calibrated thresholds for DINOv2 with PyBullet:
            # Based on observed values from actual PyBullet renders:
            # - Empty walls / boring scenes: ~2.90-2.95
            # - Moderate clutter: ~2.95-3.05
            # - High clutter / objects: >3.05

            if variance < 2.95:
                interest_score = 0.1
            elif variance > 3.05:
                interest_score = 1.0
            else:
                interest_score = (variance - 2.95) / (3.05 - 2.95)
                interest_score = min(max(interest_score, 0.1), 1.0)

            # === LEAD DIRECTION: Detect where visual interest continues ===
            # Reshape to 16x16 grid
            h, w = 16, 16
            patches = patch_tokens.reshape(1, h, w, -1)  # (1, 16, 16, 384)

            # Calculate feature magnitude (L2 norm) per patch
            energy = torch.norm(patches, dim=-1).squeeze()  # (16, 16)

            # Compare edge regions
            left_edge = energy[:, :2].mean().item()
            right_edge = energy[:, -2:].mean().item()
            center = energy[:, 4:-4].mean().item()

            # Determine lead direction
            lead = 'none'
            edge_threshold = center * 1.15

            if left_edge > edge_threshold and left_edge > right_edge * 1.2:
                lead = 'left'
            elif right_edge > edge_threshold and right_edge > left_edge * 1.2:
                lead = 'right'
            elif max(left_edge, right_edge) > edge_threshold:
                lead = 'center'

            # === VISUALIZATIONS (Added for DINOv2) ===
            import numpy as np
            from sklearn.decomposition import PCA

            # 1. PCA Rainbow Visualization
            patches_np = patch_tokens.squeeze(0).cpu().numpy()  # (256, 384)
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(patches_np)  # (256, 3)

            # Normalize to [0, 1]
            pca_min = pca_features.min(axis=0)
            pca_max = pca_features.max(axis=0)
            pca_rgb = (pca_features - pca_min) / (pca_max - pca_min + 1e-8)
            
            pca_vis = pca_rgb.reshape(h, w, 3)

            # 2. Similarity Map
            center_idx = (h // 2) * w + (w // 2)
            center_patch = patches_np[center_idx]
            similarities = np.dot(patches_np, center_patch) / (
                np.linalg.norm(patches_np, axis=1) * np.linalg.norm(center_patch) + 1e-8
            )
            similarity_map = similarities.reshape(h, w)

            # 3. Attention Map (Proxy using feature energy)
            # Normalize energy to 0-1
            energy_np = energy.cpu().numpy()
            att_min = energy_np.min()
            att_max = energy_np.max()
            attention_map = (energy_np - att_min) / (att_max - att_min + 1e-8)

        return {
            'interest_score': interest_score,
            'lead_direction': lead,
            'hazard_score': 0.1,
            'pca_vis': pca_vis,
            'attention_map': attention_map,
            'similarity_map': similarity_map
        }

    def _analyze_dinov3(self, image_rgb):
        """Analyze using DINOv3 model"""
        import torch
        from PIL import Image

        # 1. Preprocess image using DINOv3 processor
        img_pil = Image.fromarray(image_rgb)
        inputs = self.processor(images=img_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2. Run DINOv3 inference
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

            # DINOv3 outputs: [CLS, 4 register tokens, 196 patch tokens] = 201 total
            all_tokens = outputs.last_hidden_state  # Shape: (1, 201, 384)
            attentions = outputs.attentions  # Tuple of attention weights per layer

            # Extract only the patch tokens (skip CLS + 4 registers)
            patch_tokens = all_tokens[:, 5:, :]  # Shape: (1, 196, 384)

            # === INTEREST SCORE: Measure visual complexity ===
            variance = torch.var(patch_tokens, dim=1).mean().item()

            # Debug: Print variance to help with calibration
            print(f"    [DINOv3] Raw variance: {variance:.6f}")

            # Empirically calibrated thresholds for DINOv3 with PyBullet:
            # Based on observed values from actual PyBullet renders:
            # - Empty walls / boring scenes: ~0.027-0.028
            # - Objects / ducks visible: ~0.029-0.030
            # - High clutter: >0.031
            if variance < 0.028:
                interest_score = 0.1
            elif variance > 0.031:
                interest_score = 1.0
            else:
                interest_score = (variance - 0.028) / (0.031 - 0.028)
                interest_score = min(max(interest_score, 0.1), 1.0)

            # === LEAD DIRECTION: Detect where visual interest continues ===
            # Reshape to 14x14 grid (ViT-S/16: 224/16 = 14)
            h, w = 14, 14
            patches = patch_tokens.reshape(1, h, w, -1)  # (1, 14, 14, 384)

            # Calculate feature magnitude (L2 norm) per patch
            energy = torch.norm(patches, dim=-1).squeeze()  # (14, 14)

            # Compare edge regions
            left_edge = energy[:, :2].mean().item()
            right_edge = energy[:, -2:].mean().item()
            center = energy[:, 3:-3].mean().item()

            # Determine lead direction
            lead = 'none'
            edge_threshold = center * 1.15

            if left_edge > edge_threshold and left_edge > right_edge * 1.2:
                lead = 'left'
            elif right_edge > edge_threshold and right_edge > left_edge * 1.2:
                lead = 'right'
            elif max(left_edge, right_edge) > edge_threshold:
                lead = 'center'

            # === SEMANTIC VISUALIZATIONS (Meta DINOv3 style) ===
            import numpy as np
            from sklearn.decomposition import PCA

            # 1. PCA Rainbow Visualization (Meta's signature viz)
            patches_np = patch_tokens.squeeze(0).cpu().numpy()  # (196, 384)
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(patches_np)  # (196, 3)

            # Normalize to [0, 1] for RGB
            pca_min = pca_features.min(axis=0)
            pca_max = pca_features.max(axis=0)
            pca_rgb = (pca_features - pca_min) / (pca_max - pca_min + 1e-8)

            # Reshape to 14x14x3 image
            pca_vis = pca_rgb.reshape(h, w, 3)

            # 2. Attention Map (average across heads and layers)
            # Use last layer attention: (1, num_heads, 201, 201)
            last_attention = attentions[-1]  # Last transformer layer
            # Average across heads: (1, 201, 201) → (201, 201)
            avg_attention = last_attention.mean(dim=1).squeeze(0).cpu().numpy()

            # Extract attention to patch tokens from CLS token (row 0, cols 5:)
            cls_to_patches = avg_attention[0, 5:]  # (196,)
            attention_map = cls_to_patches.reshape(h, w)  # (14, 14)

            # 3. Patch Similarity Map (center patch vs all others)
            center_idx = (h // 2) * w + (w // 2)  # Middle patch
            center_patch = patches_np[center_idx]  # (384,)

            # Cosine similarity
            similarities = np.dot(patches_np, center_patch) / (
                np.linalg.norm(patches_np, axis=1) * np.linalg.norm(center_patch) + 1e-8
            )
            similarity_map = similarities.reshape(h, w)  # (14, 14)

        return {
            'interest_score': interest_score,
            'lead_direction': lead,
            'hazard_score': 0.1,
            # DINOv3 semantic visualizations
            'pca_vis': pca_vis,  # (14, 14, 3) rainbow semantic map
            'attention_map': attention_map,  # (14, 14) attention heatmap
            'similarity_map': similarity_map  # (14, 14) patch similarity
        }

    def _fallback_analysis(self, image_rgb):
        """
        Numpy-based fallback when DINOv3 model is not available.
        Uses gradient-based edge detection as a proxy for visual complexity.
        """
        # Convert to grayscale
        gray = np.mean(image_rgb, axis=2)

        # Compute image gradients (Sobel-like)
        gy, gx = np.gradient(gray)
        edge_energy = np.sqrt(gx**2 + gy**2)

        # Interest = average edge strength
        # Typical values: blank wall ~5-10, cluttered scene ~15-30
        avg_energy = np.mean(edge_energy)
        interest_score = min(max(avg_energy / 25.0, 0.1), 1.0)

        # Continuity: Compare left vs right edge regions
        h, w = gray.shape
        left_zone = edge_energy[:, :w//5].mean()
        right_zone = edge_energy[:, -w//5:].mean()
        center_zone = edge_energy[:, w//5:-w//5].mean()

        lead = 'none'
        if left_zone > center_zone * 1.2 and left_zone > right_zone * 1.3:
            lead = 'left'
        elif right_zone > center_zone * 1.2 and right_zone > left_zone * 1.3:
            lead = 'right'
        elif max(left_zone, right_zone) > center_zone * 1.2:
            lead = 'center'

        return {
            'interest_score': interest_score,
            'lead_direction': lead,
            'hazard_score': 0.1
        }

# ==============================================================================
# OPTION 2: CLIP (The "Semantic Vibe" Expert)
# Best for: Specific semantic hunting (e.g., "Find the red canister").
# ==============================================================================
class CLIPClient(BaseVisionClient):
    def __init__(self):
        # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        pass

    def analyze_scene(self, image_rgb):
        # 1. Define Prompts
        prompts = ["hazardous chemical object", "cluttered warehouse shelf", "empty concrete wall", "clean floor"]
        
        # 2. Run Inference (Image-Text Similarity)
        # probs = self.model(image_rgb, prompts)
        
        # 3. Calculate Interest
        # Interest = Probability of "hazardous" + "cluttered"
        # interest_score = probs[0] + probs[1]
        interest_score = 0.9 # Mock
        
        # 4. Calculate Continuity (Crop & Compare)
        # Crop the left 20% of the image.
        # Ask CLIP: "Is this a cut-off object?" vs "Is this empty space?"
        # left_crop = image_rgb[:, :width//5]
        # lead_score = self.model(left_crop, ["cut off object", "empty space"])
        
        lead = 'none'
        
        return {'interest_score': interest_score, 'lead_direction': lead}

# ==============================================================================
# OPTION 3: Saliency (The "Human Eye" Reflex)
# Best for: Pure speed (milliseconds).
# ==============================================================================
class SaliencyClient(BaseVisionClient):
    def analyze_scene(self, image_rgb):
        # 1. Run Saliency (e.g., OpenCV Saliency or U2Net)
        # saliency_map = cv2.saliency.StaticSaliencySpectralResidual_create().computeSaliency(image_rgb)
        
        # 2. Calculate Interest (Total Energy)
        # interest_score = np.mean(saliency_map)
        interest_score = 0.5 # Mock
        
        # 3. Calculate Continuity (Center of Mass)
        # M = cv2.moments(saliency_map)
        # cx = int(M["m10"] / M["m00"])
        # width = image_rgb.shape[1]
        
        # if cx < width * 0.3: # Center of mass is far left
        #     lead = 'left'
        lead = 'center'
        
        return {'interest_score': interest_score, 'lead_direction': lead}
