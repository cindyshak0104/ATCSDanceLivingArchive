"""
STEP6: Frontend ver2 - Real-time Service
UTILITY: Data Schemas (THIS FILE)

PURPOSE: Defines data structures for WebSocket communication.
Request/response schemas for real-time pose data and prediction results.
Ensures type safety and validation for real-time interactions.
"""

"""
Data validation schemas for real-time WebSocket messages.
Ensures incoming data from the frontend meets expected format.
"""

from typing import Dict, List, Any


def validate_pose_frame(data: Any) -> tuple[bool, str, Dict[str, Any]]:
    """
    Validate a pose frame message from the frontend.
    
    Args:
        data: The parsed JSON data to validate.
    
    Returns:
        (is_valid, error_message, cleaned_data)
        - is_valid: True if valid, False otherwise
        - error_message: Empty string if valid, error description if invalid
        - cleaned_data: The validated data dict if valid, empty dict if invalid
    """
    
    # Check message type
    if not isinstance(data, dict):
        return False, "Data must be a JSON object", {}
    
    msg_type = data.get("type")
    if msg_type != "pose_frame":
        return False, f"Expected type='pose_frame', got '{msg_type}'", {}
    
    # Check timestamp
    timestamp = data.get("timestamp")
    if timestamp is None:
        return False, "Missing 'timestamp' field", {}
    
    if not isinstance(timestamp, (int, float)):
        return False, f"timestamp must be numeric, got {type(timestamp)}", {}
    
    # Check landmarks
    landmarks = data.get("landmarks")
    if landmarks is None:
        return False, "Missing 'landmarks' field", {}
    
    if not isinstance(landmarks, list):
        return False, f"landmarks must be a list, got {type(landmarks)}", {}
    
    if len(landmarks) == 0:
        return False, "landmarks list cannot be empty", {}
    
    # Validate each landmark
    for i, landmark in enumerate(landmarks):
        if not isinstance(landmark, dict):
            return False, f"Landmark {i} must be a dict, got {type(landmark)}", {}
        
        required_keys = {"name", "x", "y", "visibility"}
        missing_keys = required_keys - set(landmark.keys())
        
        if missing_keys:
            return False, f"Landmark {i} missing keys: {missing_keys}", {}
        
        # Validate numeric fields
        for key in ["x", "y", "visibility"]:
            val = landmark.get(key)
            if not isinstance(val, (int, float)):
                return (
                    False,
                    f"Landmark {i}.{key} must be numeric, got {type(val)}",
                    {},
                )
    
    # Return cleaned data
    cleaned = {
        "type": msg_type,
        "timestamp": float(timestamp),
        "landmarks": landmarks,
    }
    
    return True, "", cleaned


def validate_ping(data: Any) -> tuple[bool, str]:
    """
    Validate a ping message (for keep-alive).
    
    Args:
        data: The parsed JSON data to validate.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Data must be a JSON object"
    
    msg_type = data.get("type")
    if msg_type != "ping":
        return False, f"Expected type='ping', got '{msg_type}'"
    
    return True, ""
