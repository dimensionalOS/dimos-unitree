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

from vector_perception.segmentation import Sam2DSegmenter
from dimos.models.depth.metric3d import Metric3D
from dimos.hardware.camera import Camera
from reactivex import Observable
from reactivex import operators as ops
from dimos.types.segmentation import SegmentationType
import numpy as np
import cv2


class SemanticSegmentationStream:
    def __init__(
        self, 
        model_path: str = "FastSAM-s.pt",
        device: str = "cuda",
        enable_depth: bool = False,
        camera_params: dict = None
    ):
        """
        Initialize a semantic segmentation stream using Sam2DSegmenter.
        
        Args:
            model_path: Path to the FastSAM model file
            device: Computation device ("cuda" or "cpu")
            enable_depth: Whether to enable depth processing
            camera_params: Dictionary containing either:
                - Direct intrinsics: [fx, fy, cx, cy]
                - Physical parameters: resolution, focal_length, sensor_size
        """
        self.segmenter = Sam2DSegmenter(
            model_path=model_path,
            device=device,
            min_analysis_interval=5.0,
            use_tracker=True,
            use_analyzer=True
        )
        
        self.enable_depth = enable_depth
        if enable_depth:
            self.depth_model = Metric3D()
            
            if camera_params:
                # Check if direct intrinsics are provided
                if 'intrinsics' in camera_params:
                    intrinsics = camera_params['intrinsics']
                    if len(intrinsics) != 4:
                        raise ValueError("Intrinsics must be a list of 4 values: [fx, fy, cx, cy]")
                    self.depth_model.update_intrinsic(intrinsics)
                else:
                    # Create camera object and calculate intrinsics from physical parameters
                    self.camera = Camera(
                        resolution=camera_params.get('resolution'),
                        focal_length=camera_params.get('focal_length'),
                        sensor_size=camera_params.get('sensor_size')
                    )
                    print(f"resolution: {camera_params.get('resolution')}")
                    print(f"focal_length: {camera_params.get('focal_length')}")
                    print(f"sensor_size: {camera_params.get('sensor_size')}")
                    intrinsics = self.camera.calculate_intrinsics()
                    self.depth_model.update_intrinsic([
                        intrinsics['focal_length_x'],
                        intrinsics['focal_length_y'],
                        intrinsics['principal_point_x'],
                        intrinsics['principal_point_y']
                    ])
            else:
                # Use default camera parameters if none provided
                self.camera = Camera(
                    resolution=(1280, 720),
                    focal_length=3.04,  # mm
                    sensor_size=(4.8, 2.7)  # mm
                )
                intrinsics = self.camera.calculate_intrinsics()
                self.depth_model.update_intrinsic([
                    intrinsics['focal_length_x'],
                    intrinsics['focal_length_y'],
                    intrinsics['principal_point_x'],
                    intrinsics['principal_point_y']
                ])
        
    def create_stream(self, video_stream: Observable) -> Observable[SegmentationType]:
        """
        Create an Observable stream of segmentation results from a video stream.
        
        Args:
            video_stream: Observable that emits video frames
            
        Returns:
            Observable that emits SegmentationType objects containing masks and metadata
        """
        def process_frame(frame):
            # Process image and get results
            masks, bboxes, target_ids, probs, names = self.segmenter.process_image(frame)
            
            # Run analysis if enabled
            if self.segmenter.use_analyzer:
                self.segmenter.run_analysis(frame, bboxes, target_ids)
                names = self.segmenter.get_object_names(target_ids, names)

            viz_frame = self.segmenter.visualize_results(
                frame,
                masks,
                bboxes,
                target_ids,
                probs,
                names
            )
            
            # Create metadata dictionary
            metadata = {
                "viz_frame": viz_frame,
                "bboxes": bboxes,
                "target_ids": target_ids,
                "probs": probs,
                "names": names
            }
            
            # Process depth if enabled
            if self.enable_depth:
                # Get depth map
                depth_map = self.depth_model.infer_depth(frame)
                depth_map = np.array(depth_map)
                
                # Calculate average depth for each object
                object_depths = []
                for mask in masks:
                    # Convert mask to numpy if needed
                    mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
                    # Get depth values where mask is True
                    object_depth = depth_map[mask_np > 0.5]
                    # Calculate average depth (in meters)
                    avg_depth = np.mean(object_depth) if len(object_depth) > 0 else 0
                    object_depths.append(avg_depth)
                
                # Create colorized depth visualization
                depth_viz = self._create_depth_visualization(depth_map)
                
                # Overlay depth values on the visualization frame
                for bbox, depth, name in zip(bboxes, object_depths, names):
                    x1, y1, x2, y2 = map(int, bbox)
                    # Draw depth text at bottom left of bounding box
                    depth_text = f"{depth:.1f}m"
                    # Add black background for better visibility
                    text_size = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(viz_frame, 
                                (x1, y2 - text_size[1] - 5),
                                (x1 + text_size[0], y2),
                                (0, 0, 0), -1)
                    # Draw text in white
                    cv2.putText(viz_frame, depth_text,
                              (x1, y2 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                metadata["depths"] = object_depths
                metadata["depth_viz"] = depth_viz
            
            # Convert masks to numpy arrays if they aren't already
            numpy_masks = [mask.cpu().numpy() if hasattr(mask, 'cpu') else mask for mask in masks]
            
            return SegmentationType(masks=numpy_masks, metadata=metadata)
        
        return video_stream.pipe(
            ops.map(process_frame)
        )
    
    def _create_depth_visualization(self, depth_map):
        """
        Create a colorized visualization of the depth map.
        
        Args:
            depth_map: Raw depth map in meters
            
        Returns:
            Colorized depth map visualization
        """
        # Normalize depth map to 0-255 range for visualization
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        
        # Apply colormap (using JET colormap for better depth perception)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        # Add depth scale bar
        scale_height = 30
        scale_width = depth_map.shape[1]  # Match width with depth map
        scale_bar = np.zeros((scale_height, scale_width, 3), dtype=np.uint8)
        
        # Create gradient for scale bar
        for i in range(scale_width):
            color = cv2.applyColorMap(np.array([[i * 255 // scale_width]], dtype=np.uint8), cv2.COLORMAP_JET)
            scale_bar[:, i] = color[0, 0]
        
        # Add depth values to scale bar
        cv2.putText(scale_bar, f"{depth_min:.1f}m", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(scale_bar, f"{depth_max:.1f}m", (scale_width-60, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine depth map and scale bar
        combined_viz = np.vstack((depth_colored, scale_bar))
        
        return combined_viz
    
    def cleanup(self):
        """Clean up resources."""
        self.segmenter.cleanup()
        if self.enable_depth:
            del self.depth_model

