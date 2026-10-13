"""
Synthetic Reference Gallery & Spatial Score-Level Fusion.
Implements:
- 256-element Integer Lookup Table (LUT) Gamma Correction (gamma = 1.8)
- Automated HSV Modality Detection (threshold eps_S = 12.0)
- 7 Canonical Synthetic Occlusion Profiles
- Top-K Spatial Score Fusion
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple


class SyntheticGalleryManager:
    """
    Manages offline synthetic occlusion gallery instantiation and Top-K matching.
    """
    def __init__(self, gamma: float = 1.8, saturation_threshold: float = 12.0, k_fusion: int = 3):
        self.gamma = gamma
        self.saturation_threshold = saturation_threshold
        self.k_fusion = k_fusion
        
        # Precompute 256-element integer Lookup Table for O(1) runtime indexing
        inv_gamma = 1.0 / self.gamma
        self.gamma_lut = np.array([
            np.round(255.0 * ((i / 255.0) ** inv_gamma)) for i in range(256)
        ], dtype=np.uint8)
        
        self.gallery_features: List[np.ndarray] = []
        self.gallery_images: Dict[str, np.ndarray] = {}

    def apply_lut_gamma(self, img: np.ndarray) -> np.ndarray:
        """Applies O(1) memory lookup table gamma correction."""
        return cv2.LUT(img, self.gamma_lut)

    def detect_modality(self, aligned_bgr: np.ndarray) -> str:
        """
        Calculates mean saturation in HSV space to adaptively classify document modality.
        """
        hsv = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2HSV)
        mean_saturation = np.mean(hsv[:, :, 1])
        return "grayscale" if mean_saturation < self.saturation_threshold else "color"

    def generate_occlusions(self, aligned_bgr: np.ndarray, glasses_asset_path: str = None) -> Dict[str, np.ndarray]:
        """
        Generates 7 canonical occlusion profiles from aligned 112x112 facial patch.
        """
        h, w = aligned_bgr.shape[:2]
        assert (h, w) == (112, 112), f"Expected 112x112 input, got ({h}, {w})"
        
        occlusions = {}
        
        # 1. Unoccluded Reference
        occlusions["unoccluded"] = aligned_bgr.copy()
        
        # 2. Eyebrow & Ocular Band (y in [35, 65])
        eye_mask = aligned_bgr.copy()
        cv2.rectangle(eye_mask, (0, 35), (112, 65), (30, 30, 30), -1)
        occlusions["eyes"] = eye_mask
        
        # 3. Forehead & Headgear (y in [0, 35])
        hat_mask = aligned_bgr.copy()
        cv2.rectangle(hat_mask, (0, 0), (112, 35), (40, 40, 40), -1)
        occlusions["hat"] = hat_mask
        
        # 4. Medical Mask / Scarf (y in [70, 112])
        mouth_mask = aligned_bgr.copy()
        cv2.rectangle(mouth_mask, (0, 70), (112, 112), (50, 50, 50), -1)
        occlusions["mask"] = mouth_mask
        
        # 5. Headscarf Rectangular (lateral cheeks + forehead)
        scarf_rect = aligned_bgr.copy()
        cv2.rectangle(scarf_rect, (0, 0), (25, 112), (20, 20, 20), -1)
        cv2.rectangle(scarf_rect, (87, 0), (112, 112), (20, 20, 20), -1)
        cv2.rectangle(scarf_rect, (0, 0), (112, 20), (20, 20, 20), -1)
        occlusions["headscarf_rect"] = scarf_rect
        
        # 6. Headscarf Organic Oval
        scarf_oval = aligned_bgr.copy()
        ellipse_mask = np.zeros((112, 112), dtype=np.uint8)
        cv2.ellipse(ellipse_mask, (56, 55), (32, 42), 0, 0, 360, 255, -1)
        scarf_oval[ellipse_mask == 0] = (20, 20, 20)
        occlusions["headscarf_oval"] = scarf_oval
        
        # 7. Eyewear Alpha-blending
        glasses_img = aligned_bgr.copy()
        if glasses_asset_path and os.path.exists(glasses_asset_path):
            try:
                glasses_png = cv2.imread(glasses_asset_path, cv2.IMREAD_UNCHANGED)
                if glasses_png is not None and glasses_png.shape[2] == 4:
                    resized_glasses = cv2.resize(glasses_png, (90, 40))
                    b, g, r, a = cv2.split(resized_glasses)
                    glasses_rgb = cv2.merge((b, g, r))
                    
                    roi = glasses_img[34:74, 13:103]
                    alpha = a.astype(np.float32) / 255.0
                    alpha_3 = cv2.merge((alpha, alpha, alpha))
                    
                    blended = (alpha_3 * glasses_rgb + (1.0 - alpha_3) * roi).astype(np.uint8)
                    glasses_img[34:74, 13:103] = blended
            except Exception:
                pass
        occlusions["glasses"] = glasses_img
        
        return occlusions

    def register_reference(self, aligned_portrait: np.ndarray, face_pipeline, glasses_path: str = None):
        """
        Generates synthetic gallery and extracts 128-D feature embeddings for all profiles.
        """
        self.gallery_features = []
        self.gallery_images = self.generate_occlusions(aligned_portrait, glasses_asset_path=glasses_path)
        
        modality = self.detect_modality(aligned_portrait)
        
        for name, profile_img in self.gallery_images.items():
            if modality == "grayscale":
                gray = cv2.cvtColor(profile_img, cv2.COLOR_BGR2GRAY)
                gamma_corrected = self.apply_lut_gamma(gray)
                input_patch = cv2.cvtColor(gamma_corrected, cv2.COLOR_GRAY2BGR)
            else:
                input_patch = profile_img
                
            # Extract 128-D feature embedding using SFace
            feat = face_pipeline.recognizer.feature(input_patch)
            self.gallery_features.append(feat)

    def match_top_k(self, live_feature: np.ndarray, face_pipeline) -> Tuple[float, List[float]]:
        """
        Computes pairwise cosine similarities across gallery profiles and applies Top-K spatial fusion.
        """
        if not self.gallery_features:
            return 0.0, []
            
        scores = []
        for ref_feat in self.gallery_features:
            s = float(face_pipeline.recognizer.match(ref_feat, live_feature, cv2.FaceRecognizerSF_FR_COSINE))
            scores.append(s)
            
        scores.sort(reverse=True)
        k = min(self.k_fusion, len(scores))
        top_k_score = float(np.mean(scores[:k])) if k > 0 else 0.0
        return top_k_score, scores
