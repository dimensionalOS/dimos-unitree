from .depth import DepthProcessor
from .labels import LabelProcessor
from .pointcloud import PointCloudProcessor
from .segment import SegmentProcessor
from dimos.stream.videostream import VideoStream   # Lukas to implement
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque

class DataPipeline:
    def __init__(self, video_stream: VideoStream,
                 run_depth: bool = False,
                 run_labels: bool = False,
                 run_pointclouds: bool = False,
                 run_segmentations: bool = False,
                 max_workers: int = 4):
        """
        Initialize the DataPipeline with specified pipeline layers.

        Args:
            video_stream (VideoStream): The video stream to process.
            run_depth (bool): Whether to run the depth map generation.
            run_labels (bool): Whether to run the label/caption generation.
            run_pointclouds (bool): Whether to run the point cloud generation.
            run_segmentations (bool): Whether to run the segmentation generation.
            max_workers (int): Maximum number of worker processes for parallel processing.

        Raises:
            ValueError: If invalid pipeline configurations are provided.
        """
        self.video_stream = video_stream
        self.depth_processor = DepthProcessor(debug=True) if run_depth else None
        self.labels_processor = LabelProcessor(debug=True) if run_labels else None
        self.pointcloud_processor = PointCloudProcessor(debug=True) if run_pointclouds else None
        self.segmentation_processor = SegmentationProcessor(debug=True) if run_segmentations else None
        self.run_depth = run_depth
        self.run_labels = run_labels
        self.run_pointclouds = run_pointclouds
        self.run_segmentations = run_segmentations

        self.max_workers = max_workers

        # Validate pipeline configuration
        self._validate_pipeline()

        # Initialize the pipeline
        self._initialize_pipeline()

        # Storage for processed data
        self.generated_depth_maps = deque()
        self.generated_labels = deque()
        self.generated_pointclouds = deque()
        self.generated_segmentations = deque()

    def _validate_pipeline(self):
        """Validate the pipeline configuration based on dependencies."""
        if self.run_pointclouds and not self.run_depth:
            raise ValueError("PointClouds generation requires Depth maps. "
                             "Enable run_depth=True to use run_pointclouds=True.")
        
        if self.run_segmentations and not self.run_labels:
            raise ValueError("Segmentations generation requires Labels. "
                             "Enable run_labels=True to use run_segmentations=True.")
        
        if not any([self.run_depth, self.run_labels, self.run_pointclouds, self.run_segmentations]):
            warnings.warn("No pipeline layers selected to run. The DataPipeline will be initialized without any processing.")

    def _initialize_pipeline(self):
        """Initialize necessary components based on selected pipeline layers."""
        if self.run_depth:
            print("Depth map generation enabled.")
        
        if self.run_labels:
            print("Label generation enabled.")
        
        if self.run_pointclouds:
            print("PointCloud generation enabled.")
        
        if self.run_segmentations:
            print("Segmentation generation enabled.")

    def run(self):
        """Execute the selected pipeline layers in parallel."""
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_frame = {}
            for frame in self.video_stream:
                # Submit frame processing to the executor
                future = executor.submit(self._process_frame, frame)
                future_to_frame[future] = frame

            # Collect results as they become available
            for future in as_completed(future_to_frame):
                result = future.result()
                depth_map, label, pointcloud, segmentation = result

                if depth_map is not None:
                    self.generated_depth_maps.append(depth_map)
                if label is not None:
                    self.generated_labels.append(label)
                if pointcloud is not None:
                    self.generated_pointclouds.append(pointcloud)
                if segmentation is not None:
                    self.generated_segmentations.append(segmentation)

    def _process_frame(self, frame):
        """Process a single frame and return results."""
        depth_map = None
        label = None
        pointcloud = None
        segmentation = None

        if self.run_depth:
            depth_map = self.depth_processor.process(frame)

        if self.run_labels:
            label = self.labels_processor.caption_image_data(frame)

        if self.run_pointclouds and depth_map is not None:
            pointcloud = self.pointcloud_processor.process_frame(frame, depth_map)

        if self.run_segmentations and label is not None:
            segmentation = self.segmentation_processor.process_frame(frame, label)

        return depth_map, label, pointcloud, segmentation
