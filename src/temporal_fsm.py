"""
3-Phase Noise-Resistant Temporal Finite State Machine (FSM).
Implements:
- Phase 1: Calibration (15 frames, Top-10 median aggregation)
- Phase 2: Optical Flow Tracking (Lucas-Kanade tracking, >30 FPS)
- Phase 3: Burst Verification (periodic burst consensus, INT8 drift mitigation)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class TemporalVerificationFSM:
    """
    Finite state machine managing live camera stream lifecycle on wearable smart glasses.
    """
    def __init__(self, threshold: float = 0.400, calibration_window: int = 15,
                 burst_interval: int = 30, burst_window: int = 5, consensus_window: int = 40):
        self.threshold = threshold
        self.calibration_window = calibration_window
        self.burst_interval = burst_interval
        self.burst_window = burst_window
        self.consensus_window = consensus_window
        
        self.state = "CALIBRATION"  # CALIBRATION, TRACKING, BURST_VERIFY
        self.calibration_scores: List[float] = []
        self.locked_score: float = 0.0
        
        self.prev_gray: Optional[np.ndarray] = None
        self.tracked_points: Optional[np.ndarray] = None
        self.tracking_frame_count: int = 0
        
        self.burst_scores: List[float] = []
        self.temporal_history: List[bool] = []

    def reset(self):
        """Resets the state machine back to calibration."""
        self.state = "CALIBRATION"
        self.calibration_scores = []
        self.locked_score = 0.0
        self.prev_gray = None
        self.tracked_points = None
        self.tracking_frame_count = 0
        self.burst_scores = []
        self.temporal_history = []

    def process_calibration_frame(self, spatial_score: float) -> Tuple[str, float]:
        """
        Phase 1: Collects 15 candidate frames and computes Top-10 median score.
        """
        self.calibration_scores.append(spatial_score)
        if len(self.calibration_scores) >= self.calibration_window:
            sorted_scores = sorted(self.calibration_scores, reverse=True)
            top_10 = sorted_scores[:10]
            self.locked_score = float(np.median(top_10))
            
            if self.locked_score >= self.threshold:
                self.state = "TRACKING"
                self.tracking_frame_count = 0
                return "VERIFIED_LOCKED", self.locked_score
            else:
                self.reset()
                return "REJECTED_RETRY", self.locked_score
                
        return "CALIBRATING", spatial_score

    def init_optical_flow(self, frame_gray: np.ndarray, face_box: np.ndarray):
        """Initializes Shi-Tomasi good features to track within facial bounding box."""
        x, y, w, h = [int(v) for v in face_box[:4]]
        roi_mask = np.zeros_like(frame_gray)
        roi_mask[max(0, y):min(frame_gray.shape[0], y+h), max(0, x):min(frame_gray.shape[1], x+w)] = 255
        
        pts = cv2.goodFeaturesToTrack(frame_gray, maxCorners=50, qualityLevel=0.01, minDistance=5, mask=roi_mask)
        self.tracked_points = pts
        self.prev_gray = frame_gray.copy()

    def track_optical_flow(self, curr_gray: np.ndarray) -> bool:
        """
        Phase 2: Lucas-Kanade optical flow feature tracking across consecutive frames.
        Returns True if tracking succeeds, False if tracking fails and re-detection is needed.
        """
        if self.tracked_points is None or len(self.tracked_points) < 5 or self.prev_gray is None:
            return False
            
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.tracked_points, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        good_pts = next_pts[status == 1] if next_pts is not None else []
        if len(good_pts) < 5:
            return False  # Motion displacement lost
            
        self.tracked_points = good_pts.reshape(-1, 1, 2)
        self.prev_gray = curr_gray.copy()
        self.tracking_frame_count += 1
        
        if self.tracking_frame_count >= self.burst_interval:
            self.state = "BURST_VERIFY"
            self.burst_scores = []
            
        return True

    def process_burst_frame(self, spatial_score: float) -> Tuple[str, float]:
        """
        Phase 3: Gathers a 5-frame burst window and computes median consensus score.
        """
        self.burst_scores.append(spatial_score)
        if len(self.burst_scores) >= self.burst_window:
            burst_median = float(np.median(self.burst_scores))
            passed = burst_median >= self.threshold
            
            # Update rolling temporal consensus buffer
            self.temporal_history.append(passed)
            if len(self.temporal_history) > self.consensus_window:
                self.temporal_history.pop(0)
                
            self.state = "TRACKING"
            self.tracking_frame_count = 0
            
            if passed:
                return "BURST_PASSED", burst_median
            else:
                self.reset()
                return "BURST_FAILED", burst_median
                
        return "BURST_COLLECTING", spatial_score

    def get_consensus_ratio(self) -> float:
        """Computes C_temporal = M_pass / N_window over rolling temporal history."""
        if not self.temporal_history:
            return 1.0 if self.state == "TRACKING" else 0.0
        return float(sum(self.temporal_history) / len(self.temporal_history))
