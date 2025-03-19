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

from reactivex import Observable
from reactivex import operators as ops
from dimos.types.segmentation import SegmentationType


class SemanticSegmentationStream:
    def __init__(self, model_path: str = "FastSAM-s.pt", device: str = "cuda"):
        """
        Initialize a semantic segmentation stream using Sam2DSegmenter.
        
        Args:
            model_path: Path to the FastSAM model file
            device: Computation device ("cuda" or "cpu")
        """
        self.segmenter = Sam2DSegmenter(
            model_path=model_path,
            device=device,
            min_analysis_interval=5.0,
            use_tracker=True,
            use_analyzer=True
        )
        
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
            
            # Convert masks to numpy arrays if they aren't already
            numpy_masks = [mask.cpu().numpy() if hasattr(mask, 'cpu') else mask for mask in masks]
            
            return SegmentationType(masks=numpy_masks, metadata=metadata)
        
        return video_stream.pipe(
            ops.map(process_frame)
        )
    
    def cleanup(self):
        """Clean up resources."""
        self.segmenter.cleanup()

