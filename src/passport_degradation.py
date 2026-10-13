"""
Physical Passport Document Degradation Model.
Simulates real-world identity document artifacts:
- Monochrome BT.601 desaturation
- Printing press/scanner resolution downscaling
- Halftone Gaussian micro-noise (sigma = 8.0)
- Specular laminate radial glare
"""

import cv2
import numpy as np


def apply_passport_degradation(img: np.ndarray, target_res: float = 200.0, noise_sigma: float = 8.0) -> np.ndarray:
    """
    Applies calibrated physical passport degradation to an input image.
    
    Args:
        img (np.ndarray): Input portrait (BGR).
        target_res (float): Equivalent document height in pixels (default: 200.0).
        noise_sigma (float): Gaussian noise standard deviation (default: 8.0).
        
    Returns:
        np.ndarray: Degraded synthetic document portrait.
    """
    if img is None:
        raise ValueError("Input image is None.")
        
    h, w = img.shape[:2]
    
    # 1. Grayscale Conversion (Simulating laser-engraved travel document)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Resampling & Resolution Loss
    scale_down = max(0.1, target_res / float(h))
    small = cv2.resize(gray, (0, 0), fx=scale_down, fy=scale_down, interpolation=cv2.INTER_LINEAR)
    upscaled = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 3. Additive Halftone Gaussian Noise
    noise = np.random.normal(0, noise_sigma, upscaled.shape).astype(np.float32)
    noisy = cv2.add(upscaled.astype(np.float32), noise)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Convert back to 3-channel BGR
    degraded = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
    
    # 4. Specular Laminate Reflection Glare
    glare = np.zeros((h, w, 3), dtype=np.float32)
    center = (int(w * 0.8), int(h * 0.2))  # Centered at (0.8W, 0.2H)
    radius = int(min(w, h) * 0.6)
    
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if dist < radius:
                intensity = 1.0 - (dist / radius)
                val = 255.0 * intensity * 0.5  # 50% opacity
                glare[y, x] = (val, val, val)
                
    final = cv2.addWeighted(degraded, 1.0, glare.astype(np.uint8), 1.0, 0)
    return final
