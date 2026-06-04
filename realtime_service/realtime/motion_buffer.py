"""
STEP6: Frontend ver2 - Real-time Service
 WebSocket Routes
 Feature Extraction
 Motion Buffering (THIS FILE)
 Real-time prediction and feedback

PURPOSE: Maintains a sliding window of recent pose data.
Buffers live pose landmarks for sequence-based analysis (emotional prediction).
Handles temporal synchronization between video stream and analysis.
"""

"""
Motion Buffer: Stores and manages recent pose frames in a sliding window.
Used by the real-time prediction pipeline to analyze motion patterns.
"""

from collections import deque
from typing import List, Dict, Any


class MotionBuffer:
    """
    A fixed-size buffer that stores recent pose frames.
    
    When full, adding a new frame removes the oldest one (FIFO).
    Provides methods to query the buffer and check readiness for prediction.
    """

    def __init__(self, max_length: int = 30):
        """
        Initialize the motion buffer.
        
        Args:
            max_length: Maximum number of frames to store (default 30 for ~0.5s at 60fps).
        """
        self.max_length = max_length
        self.frames: deque = deque(maxlen=max_length)

    def add_frame(self, frame: Dict[str, Any]) -> None:
        """
        Add a pose frame to the buffer.
        If the buffer is at max capacity, the oldest frame is automatically discarded.
        
        Args:
            frame: A dictionary containing:
                - timestamp: float (milliseconds)
                - landmarks: List of landmark dicts with {name, x, y, visibility}
        """
        self.frames.append(frame)

    def get_recent_frames(self, n: int = None) -> List[Dict[str, Any]]:
        """
        Get the most recent n frames from the buffer.
        
        Args:
            n: Number of frames to retrieve. If None, returns all frames.
        
        Returns:
            List of frame dictionaries, most recent last.
        """
        if n is None:
            return list(self.frames)
        
        return list(self.frames)[-n:] if n > 0 else []

    def is_ready(self, min_frames: int = 5) -> bool:
        """
        Check if the buffer has enough frames for prediction.
        
        Args:
            min_frames: Minimum number of frames required (default 5).
        
        Returns:
            True if buffer has at least min_frames frames.
        """
        return len(self.frames) >= min_frames

    def clear(self) -> None:
        """Clear all frames from the buffer."""
        self.frames.clear()

    def size(self) -> int:
        """Return the current number of frames in the buffer."""
        return len(self.frames)
