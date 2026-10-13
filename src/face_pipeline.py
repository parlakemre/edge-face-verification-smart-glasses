"""
Face Detection and Embedding Extraction Pipeline using YuNet and SFace.
"""

import cv2
import numpy as np


class FacePipeline:
    """
    Wraps OpenCV YuNet detector and SFace recognizer with edge-optimized parameters.
    """
    def __init__(self, detection_model_path: str, recognition_model_path: str, score_threshold: float = 0.6):
        self.detector = cv2.FaceDetectorYN.create(
            model=detection_model_path,
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=recognition_model_path,
            config="",
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )

    def detect(self, image: np.ndarray, downscale_factor: float = 4.0):
        """
        Detects faces using aspect-ratio preserving dynamic downscaling.
        """
        h, w = image.shape[:2]
        target_w = int(w / downscale_factor)
        target_h = int(h / downscale_factor)
        
        small = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        self.detector.setInputSize((target_w, target_h))
        faces = self.detector.detect(small)
        
        if faces[1] is None or len(faces[1]) == 0:
            return None
        
        # Scale bounding box and landmarks back to native frame coordinates
        rescaled_faces = faces[1].copy()
        rescaled_faces[:, :14] *= downscale_factor
        return rescaled_faces

    def align_and_extract(self, image: np.ndarray, face_box: np.ndarray) -> np.ndarray:
        """
        Aligns the facial crop to 112x112 using 5 facial landmarks and extracts a 128-D embedding.
        """
        aligned_face = self.recognizer.alignCrop(image, face_box)
        feature_vector = self.recognizer.feature(aligned_face)
        return feature_vector

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Computes cosine similarity between two feature embeddings.
        """
        return float(self.recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE))
