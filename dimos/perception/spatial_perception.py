# Copyright 2025 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Spatial Memory module for creating a semantic map of the environment.
"""

import logging
import uuid
import time
import uuid
import os
import math
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import cv2
from reactivex import Observable
from reactivex import operators as ops
from reactivex.subject import Subject
from datetime import datetime

from dimos.utils.logging_config import setup_logger
from dimos.agents.memory.spatial_vector_db import SpatialVectorDB
from dimos.agents.memory.image_embedding import ImageEmbeddingProvider
from dimos.agents.memory.visual_memory import VisualMemory
from dimos.types.vector import Vector

logger = setup_logger("dimos.perception.spatial_memory")

class SpatialMemory:
    """
    A class for building and querying Robot spatial memory.
    
    This class processes video frames from ROSControl, associates them with
    XY locations, and stores them in a vector database for later retrieval.
    """
    
    def __init__(
        self,
        collection_name: str = "spatial_memory",
        embedding_model: str = "clip", 
        embedding_dimensions: int = 512,
        min_distance_threshold: float = 0.01,  # Min distance in meters to store a new frame
        min_time_threshold: float = 1.0,  # Min time in seconds to store a new frame
        chroma_client = None,  # Optional ChromaDB client for persistence
        visual_memory = None,  # Optional VisualMemory instance for storing images
        output_dir: str = None,  # Directory for storing visual memory data
    ):
        """
        Initialize the spatial perception system.
        
        Args:
            collection_name: Name of the vector database collection
            embedding_model: Model to use for image embeddings ("clip", "resnet", etc.)
            embedding_dimensions: Dimensions of the embedding vectors
            min_distance_threshold: Minimum distance in meters to record a new frame
            min_time_threshold: Minimum time in seconds to record a new frame
            chroma_client: Optional ChromaDB client for persistent storage
            visual_memory: Optional VisualMemory instance for storing images
            output_dir: Directory for storing visual memory data if visual_memory is not provided
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.min_distance_threshold = min_distance_threshold
        self.min_time_threshold = min_time_threshold
        
        # Create visual memory if not provided
        if visual_memory is None and output_dir is not None:
            visual_memory = VisualMemory(output_dir=output_dir)
        
        # Pass the chroma_client and visual_memory to SpatialVectorDB
        self.vector_db = SpatialVectorDB(
            collection_name=collection_name,
            chroma_client=chroma_client,
            visual_memory=visual_memory
        )
        
        self.embedding_provider = ImageEmbeddingProvider(
            model_name=embedding_model,
            dimensions=embedding_dimensions
        )
        
        self.last_position: Optional[Vector] = None
        self.last_record_time: Optional[float] = None
        
        self.frame_count = 0
        self.stored_frame_count = 0
        
        logger.info(f"SpatialMemory initialized with model {embedding_model}")
    
    
    def query_by_location(self, x: float, y: float, radius: float = 2.0, limit: int = 5) -> List[Dict]:
        """
        Query the vector database for images near the specified location.
        
        Args:
            x: X coordinate
            y: Y coordinate
            radius: Search radius in meters
            limit: Maximum number of results to return
            
        Returns:
            List of results, each containing the image and its metadata
        """
        return self.vector_db.query_by_location(x, y, radius, limit)
    
    def process_stream(self, combined_stream: Observable) -> Observable:
        """
        Process a combined stream of video frames and positions.
        
        This method handles a stream where each item already contains both the frame and position,
        such as the stream created by combining video and transform streams with the 
        with_latest_from operator.
        
        Args:
            combined_stream: Observable stream of dictionaries containing 'frame' and 'position'
            
        Returns:
            Observable of processing results, including the stored frame and its metadata
        """
        self.last_position = None
        self.last_record_time = None
        
        def process_combined_data(data):
            self.frame_count += 1
            
            frame = data.get('frame')
            position_vec = data.get('position')  # Use .get() for consistency
            rotation_vec = data.get('rotation')  # Get rotation data if available
            
            
            if not position_vec or not rotation_vec:
                logger.info("No position or rotation data available, skipping frame")
                return None
            
            if self.last_position is not None and (self.last_position - position_vec).length() < self.min_distance_threshold:
                logger.debug("Position has not moved, skipping frame")
                return None

            if self.last_record_time is not None and (time.time() - self.last_record_time) < self.min_time_threshold:
                logger.debug("Time since last record too short, skipping frame")
                return None
            
            current_time = time.time()
                        
            frame_embedding = self.embedding_provider.get_embedding(frame)
            
            frame_id = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Create metadata dictionary with primitive types only
            metadata = {
                "pos_x": float(position_vec.x),
                "pos_y": float(position_vec.y),
                "pos_z": float(position_vec.z),
                "rot_x": float(rotation_vec.x),
                "rot_y": float(rotation_vec.y),
                "rot_z": float(rotation_vec.z),
                "timestamp": current_time,
                "frame_id": frame_id
            }
            
            self.vector_db.add_image_vector(
                vector_id=frame_id,
                image=frame,
                embedding=frame_embedding,
                metadata=metadata
            )
            
            self.last_position = position_vec
            self.last_record_time = current_time
            self.stored_frame_count += 1
            
            logger.info(f"Stored frame at position {position_vec}, rotation {rotation_vec})"
                        f" stored {self.stored_frame_count}/{self.frame_count} frames")
            
            # Create return dictionary with primitive-compatible values
            return {
                "frame": frame,
                "position": (position_vec.x, position_vec.y, position_vec.z),
                "rotation": (rotation_vec.x, rotation_vec.y, rotation_vec.z),
                "frame_id": frame_id,
                "timestamp": current_time
            }
                    
        return combined_stream.pipe(
            ops.map(process_combined_data),
            ops.filter(lambda result: result is not None)
        )

    def query_by_image(self, image: np.ndarray, limit: int = 5) -> List[Dict]:
        """
        Query the vector database for images similar to the provided image.
        
        Args:
            image: Query image
            limit: Maximum number of results to return
            
        Returns:
            List of results, each containing the image and its metadata
        """
        embedding = self.embedding_provider.get_embedding(image)
        return self.vector_db.query_by_embedding(embedding, limit)
    
    def query_by_text(self, text: str, limit: int = 5) -> List[Dict]:
        """
        Query the vector database for images matching the provided text description.
        
        This method uses CLIP's text-to-image matching capability to find images
        that semantically match the text query (e.g., "where is the kitchen").
        
        Args:
            text: Text query to search for
            limit: Maximum number of results to return
            
        Returns:
            List of results, each containing the image, its metadata, and similarity score
        """
        logger.info(f"Querying spatial memory with text: '{text}'")
        return self.vector_db.query_by_text(text, limit)
    
    def cleanup(self):
        """Clean up resources."""
        if self.vector_db:
            logger.info(f"Cleaning up SpatialMemory, stored {self.stored_frame_count} frames")
