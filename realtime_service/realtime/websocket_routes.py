"""
STEP6: Frontend ver2 - Real-time Service
 WebSocket Routes (THIS FILE)
 Feature extraction from live poses
 Real-time prediction and feedback
 Motion buffering and sequence analysis

PURPOSE: FastAPI WebSocket router for real-time interaction.
Handles live pose streams from the frontend and provides real-time pose matching + emotion feedback.
Integrates STEP3 ML models for live inference.
"""

"""
WebSocket routes for real-time prediction mode.
Handles incoming pose streams and returns next movement predictions.
"""

import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from realtime_service.realtime.motion_buffer import MotionBuffer
from realtime_service.realtime.next_movement_predictor import NextMovementPredictor
from realtime_service.realtime.schemas import validate_pose_frame, validate_ping


logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/realtime-dance")
async def realtime_dance_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time pose streaming and prediction.
    
    Client sends pose frames in JSON format:
    {
        "type": "pose_frame",
        "timestamp": 12345.67,
        "landmarks": [...]
    }
    
    Server responds with predictions:
    {
        "type": "prediction",
        "status": "ok" | "warming_up",
        "current_matched_movement_id": "clip_003",
        "predicted_movement_id": "clip_004",
        ...
    }
    """
    
    await websocket.accept()
    logger.info("WebSocket connection accepted: /ws/realtime-dance")
    
    # Initialize buffer and predictor for this connection
    buffer = MotionBuffer(max_length=30)
    predictor = NextMovementPredictor()
    
    try:
        while True:
            # Receive raw JSON string from client
            raw_data = await websocket.receive_text()
            
            try:
                # Parse JSON
                data = json.loads(raw_data)
            except json.JSONDecodeError as e:
                # Send error response if JSON is malformed
                error_response = {
                    "type": "error",
                    "status": "error",
                    "message": f"Invalid JSON: {str(e)}",
                }
                await websocket.send_json(error_response)
                continue
            
            # Handle ping (keep-alive)
            if data.get("type") == "ping":
                is_valid, error_msg = validate_ping(data)
                if is_valid:
                    await websocket.send_json({"type": "pong"})
                else:
                    error_response = {
                        "type": "error",
                        "status": "error",
                        "message": error_msg,
                    }
                    await websocket.send_json(error_response)
                continue
            
            # Handle pose frame
            if data.get("type") == "pose_frame":
                is_valid, error_msg, cleaned_data = validate_pose_frame(data)
                
                if not is_valid:
                    # Send error response if validation fails
                    error_response = {
                        "type": "error",
                        "status": "error",
                        "message": error_msg,
                    }
                    await websocket.send_json(error_response)
                    continue
                
                # Add frame to buffer
                buffer.add_frame(cleaned_data)
                
                # Generate prediction
                prediction = predictor.predict(buffer)
                
                # Send prediction back to client
                await websocket.send_json(prediction)
                continue
            
            # Unknown message type
            error_response = {
                "type": "error",
                "status": "error",
                "message": f"Unknown message type: {data.get('type')}",
            }
            await websocket.send_json(error_response)
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: /ws/realtime-dance")
        buffer.clear()
    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        
        # Attempt to send error message before closing
        try:
            error_response = {
                "type": "error",
                "status": "error",
                "message": f"Server error: {str(e)}",
            }
            await websocket.send_json(error_response)
        except Exception:
            pass  # Connection may already be closed
        
        # Cleanup
        buffer.clear()
