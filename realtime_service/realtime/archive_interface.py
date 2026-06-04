"""
STEP6: Frontend ver2 - Real-time Service
UTILITY: Archive Interface (THIS FILE)

PURPOSE: Provides access to the pose archive database.
Loads and manages archive embeddings and metadata for real-time querying.
Supports fast retrieval of similar poses from the database during live interaction.
"""

"""
Archive Interface: Manage archive metadata and stick figure paths
Maps movement IDs to stick figure frame sequences
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)


class ArchiveInterface:
    """Interface to archive stick figures and metadata."""

    def __init__(self, stick_images_dir: str = "stick_images2"):
        """
        Args:
            stick_images_dir: Path to stick_images2 directory
        """
        self.stick_images_dir = Path(stick_images_dir)
        self._movement_cache = {}
        self._load_movements()

    def _load_movements(self):
        """Scan stick_images2 directory and build movement index."""
        if not self.stick_images_dir.exists():
            logger.warning(
                f"stick_images_dir not found at {self.stick_images_dir}"
            )
            return

        try:
            # List all subdirectories (each is a movement)
            dirs = [
                d
                for d in self.stick_images_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]

            for movement_dir in sorted(dirs):
                movement_id = movement_dir.name

                # Find PNG frames (more efficient than full iterdir scan)
                frames_dir = list(movement_dir.glob("*.png"))

                if frames_dir:
                    # Sort frame names numerically
                    frames = sorted([f.name for f in frames_dir])

                    # Build relative URLs for frontend
                    frame_urls = [
                        f"/stick_images2/{movement_id}/{frame}"
                        for frame in frames
                    ]

                    self._movement_cache[movement_id] = {
                        "movement_id": movement_id,
                        "stick_figure_path": f"/stick_images2/{movement_id}/",
                        "stick_figure_frames": frame_urls,
                        "frame_count": len(frame_urls),
                    }

            logger.info(
                f"Loaded {len(self._movement_cache)} movements from {self.stick_images_dir}"
            )
        except Exception as e:
            logger.error(f"Error loading movements: {e}")

    def get_movement_frames(self, movement_id: str) -> Optional[List[str]]:
        """
        Get stick figure frame URLs for a movement.

        Args:
            movement_id: The movement ID (e.g., "clip_001")

        Returns:
            List of frame URLs, or None if movement not found
        """
        if movement_id in self._movement_cache:
            return self._movement_cache[movement_id]["stick_figure_frames"]

        return None

    def get_movement_info(self, movement_id: str) -> Optional[Dict]:
        """
        Get full movement metadata.

        Args:
            movement_id: The movement ID

        Returns:
            Dict with movement metadata or None if not found
        """
        return self._movement_cache.get(movement_id)

    def list_movements(self) -> List[str]:
        """Get list of all available movement IDs."""
        return sorted(list(self._movement_cache.keys()))

    def get_all_movements(self) -> Dict[str, Dict]:
        """Get all movement metadata."""
        return self._movement_cache.copy()

    def has_movement(self, movement_id: str) -> bool:
        """Check if a movement exists in the archive."""
        return movement_id in self._movement_cache
