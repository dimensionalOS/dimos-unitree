from abc import ABC, abstractmethod

class Environment(ABC):
    @abstractmethod
    def label_objects(self):
        """Label all objects in the environment."""
        pass

    @abstractmethod
    def get_visualization(self, format_type):
        """Return different visualization formats like images, NERFs, or other 3D file types."""
        pass

    @abstractmethod
    def get_segmentations(self):
        """Get segmentations using a method like 'segment anything'."""
        pass

    @abstractmethod
    def get_point_cloud(self, object_id=None):
        """Return point clouds of the entire environment or a specific object."""
        pass

    @abstractmethod
    def get_depth_map(self):
        """Return depth maps of the environment."""
        pass

    # Removed abstract decorators from initialization methods
    def initialize_from_images(self, images):
        """Initialize the environment from a set of image frames or video."""
        raise NotImplementedError("This method is not implemented for this environment type.")

    def initialize_from_file(self, file_path):
        """Initialize the environment from a spatial file type like GLTF."""
        raise NotImplementedError("This method is not implemented for this environment type.")


