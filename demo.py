"""
Quickstart Demo: End-to-End Face Verification Pipeline.
Demonstrates:
1. Physical document degradation
2. Synthetic occlusion gallery generation (7 profiles)
3. Lookup Table (LUT) gamma correction
4. Top-3 spatial score fusion
"""

import os
import cv2
import numpy as np

from src.face_pipeline import FacePipeline
from src.passport_degradation import apply_passport_degradation
from src.synthetic_gallery import SyntheticGalleryManager


def main():
    print("=" * 60)
    print("Edge Face Verification on Smart Glasses - Quickstart Demo")
    print("=" * 60)

    # 1. Initialize Synthetic Gallery Manager
    gallery_mgr = SyntheticGalleryManager(gamma=1.8, saturation_threshold=12.0, k_fusion=3)

    # 2. Create a synthetic sample face patch (112x112) for demonstration
    sample_face = np.full((112, 112, 3), 180, dtype=np.uint8)
    # Add simple facial landmark geometries (eyes, nose, mouth)
    cv2.circle(sample_face, (35, 45), 8, (60, 40, 30), -1)   # Left eye
    cv2.circle(sample_face, (77, 45), 8, (60, 40, 30), -1)   # Right eye
    cv2.line(sample_face, (56, 50), (56, 68), (50, 40, 30), 3) # Nose bridge
    cv2.ellipse(sample_face, (56, 82), (18, 8), 0, 0, 180, (40, 30, 80), -1) # Mouth

    print("\n[Step 1] Applying physical document degradation model...")
    degraded_doc = apply_passport_degradation(sample_face, target_res=200.0, noise_sigma=8.0)
    print(f"  -> Generated degraded document portrait ({degraded_doc.shape[1]}x{degraded_doc.shape[0]})")

    print("\n[Step 2] Generating 7 canonical synthetic occlusion profiles...")
    glasses_asset = "assets/glasses.png" if os.path.exists("assets/glasses.png") else None
    occlusion_profiles = gallery_mgr.generate_occlusions(sample_face, glasses_asset_path=glasses_asset)
    for name, img in occlusion_profiles.items():
        print(f"  -> Generated profile '{name}': {img.shape}")

    print("\n[Step 3] Checking HSV modality detection...")
    modality = gallery_mgr.detect_modality(degraded_doc)
    print(f"  -> Detected modality for degraded document: '{modality}'")

    print("\n[Step 4] Applying integer LUT gamma correction (gamma=1.8)...")
    gamma_corrected = gallery_mgr.apply_lut_gamma(degraded_doc)
    print(f"  -> Gamma corrected intensity mean: {np.mean(gamma_corrected):.2f}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("To run benchmark evaluations on LFW or RFW:")
    print("  python src/evaluate_lfw.py --lfw /path/to/lfw")
    print("  python src/evaluate_rfw.py --rfw /path/to/rfw")
    print("=" * 60)


if __name__ == "__main__":
    main()
