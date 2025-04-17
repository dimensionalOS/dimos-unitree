import numpy as np
from typing import List, Union, Tuple, Iterator, Optional, TypeVar, Generic, Sequence
from vector import Vector

T = TypeVar('T', bound='Path')

class Path:
    """A class representing a path as a sequence of points."""
    
    def __init__(self, points: Union[List[Vector], List[np.ndarray], List[Tuple], np.ndarray] = None):
        """Initialize a path from a list of points.
        
        Args:
            points: List of Vector objects, numpy arrays, tuples, or a 2D numpy array where each row is a point.
                   If None, creates an empty path.
        
        Examples:
            Path([Vector(1, 2), Vector(3, 4)])  # from Vector objects
            Path([(1, 2), (3, 4)])              # from tuples
            Path(np.array([[1, 2], [3, 4]]))    # from 2D numpy array
        """
        self._points: List[Vector] = []
        
        if points is not None:
            if isinstance(points, np.ndarray) and points.ndim == 2:
                # Handle 2D numpy array
                self._points = [Vector(point) for point in points]
            else:
                # Handle list of vectors, tuples, etc.
                self._points = [p if isinstance(p, Vector) else Vector(p) for p in points]
    
    @property
    def points(self) -> List[Vector]:
        """Get the path points as Vector objects."""
        return self._points
    
    @property
    def numpy_points(self) -> np.ndarray:
        """Get the path points as a numpy array."""
        if not self._points:
            return np.array([])
        return np.array([p.data for p in self._points])
    
    def append(self, point: Union[Vector, np.ndarray, Tuple]) -> None:
        """Append a point to the path.
        
        Args:
            point: A Vector, numpy array, or tuple representing a point
        """
        if not isinstance(point, Vector):
            point = Vector(point)
        self._points.append(point)
    
    def extend(self, points: Union[List[Vector], List[np.ndarray], List[Tuple], 'Path']) -> None:
        """Extend the path with more points.
        
        Args:
            points: List of points or another Path object
        """
        if isinstance(points, Path):
            self._points.extend(points.points)
        else:
            for point in points:
                if not isinstance(point, Vector):
                    point = Vector(point)
                self._points.append(point)
    
    def insert(self, index: int, point: Union[Vector, np.ndarray, Tuple]) -> None:
        """Insert a point at a specific index.
        
        Args:
            index: The index at which to insert the point
            point: A Vector, numpy array, or tuple representing a point
        """
        if not isinstance(point, Vector):
            point = Vector(point)
        self._points.insert(index, point)
    
    def remove(self, index: int) -> Vector:
        """Remove and return the point at the given index.
        
        Args:
            index: The index of the point to remove
            
        Returns:
            The removed point
        """
        return self._points.pop(index)
    
    def clear(self) -> None:
        """Remove all points from the path."""
        self._points.clear()
    
    def length(self) -> float:
        """Calculate the total length of the path.
        
        Returns:
            The sum of the distances between consecutive points
        """
        if len(self._points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(self._points) - 1):
            total_length += self._points[i].distance(self._points[i + 1])
        return total_length
    
    def resample(self: T, point_spacing: float) -> T:
        """Resample the path with approximately uniform spacing between points.
        
        Args:
            point_spacing: The desired distance between consecutive points
            
        Returns:
            A new Path object with resampled points
        """
        if len(self._points) < 2 or point_spacing <= 0:
            return self.__class__(self._points)
        
        resampled_points = [self._points[0]]
        accumulated_distance = 0.0
        
        for i in range(1, len(self._points)):
            current_point = self._points[i]
            prev_point = self._points[i-1]
            segment_vector = current_point - prev_point
            segment_length = segment_vector.length()
            
            if segment_length < 1e-10:
                continue
                
            direction = segment_vector / segment_length
            
            # Add points along this segment until we reach the end
            while accumulated_distance + segment_length >= point_spacing:
                # How far along this segment the next point should be
                dist_along_segment = point_spacing - accumulated_distance
                if dist_along_segment < 0:
                    break
                    
                # Create the new point
                new_point = prev_point + direction * dist_along_segment
                resampled_points.append(new_point)
                
                # Update for next iteration
                accumulated_distance = 0
                segment_length -= dist_along_segment
                prev_point = new_point
            
            # Update the accumulated distance for the next segment
            accumulated_distance += segment_length
        
        # Add the last point if it's not already there
        if len(self._points) > 1 and (resampled_points[-1] != self._points[-1]):
            resampled_points.append(self._points[-1])
            
        return self.__class__(resampled_points)
    
    def simplify(self: T, tolerance: float) -> T:
        """Simplify the path using the Ramer-Douglas-Peucker algorithm.
        
        Args:
            tolerance: The maximum distance a point can deviate from the simplified path
            
        Returns:
            A new simplified Path object
        """
        if len(self._points) <= 2:
            return self.__class__(self._points)
            
        # Implementation of Ramer-Douglas-Peucker algorithm
        def rdp(points, epsilon, start, end):
            if end <= start + 1:
                return [start]
                
            # Find point with max distance from line
            line_vec = points[end] - points[start]
            if line_vec.length() < 1e-10:  # If start and end points are the same
                max_dist = points[start+1].distance(points[start])
                max_idx = start + 1
            else:
                max_dist = 0
                max_idx = start
                
                for i in range(start + 1, end):
                    # Distance from point to line
                    p_vec = points[i] - points[start]
                    # Project p_vec onto line_vec
                    proj = p_vec.project(line_vec)
                    # Calculate perpendicular distance
                    perp_vec = p_vec - proj
                    dist = perp_vec.length()
                    
                    if dist > max_dist:
                        max_dist = dist
                        max_idx = i
            
            # Recursive call
            result = []
            if max_dist > epsilon:
                result_left = rdp(points, epsilon, start, max_idx)
                result_right = rdp(points, epsilon, max_idx, end)
                result = result_left + result_right[1:]
            else:
                result = [start, end]
                
            return result
            
        indices = rdp(self._points, tolerance, 0, len(self._points) - 1)
        indices.append(len(self._points) - 1)  # Make sure the last point is included
        indices = sorted(set(indices))  # Remove duplicates and sort
        
        return self.__class__([self._points[i] for i in indices])
    
    def smooth(self: T, weight: float = 0.5, iterations: int = 1) -> T:
        """Smooth the path using a moving average filter.
        
        Args:
            weight: How much to weight the neighboring points (0-1)
            iterations: Number of smoothing passes to apply
            
        Returns:
            A new smoothed Path object
        """
        if len(self._points) <= 2 or weight <= 0 or iterations <= 0:
            return self.__class__(self._points)
            
        smoothed_points = list(self._points)
        
        for _ in range(iterations):
            new_points = [smoothed_points[0]]  # Keep first point unchanged
            
            for i in range(1, len(smoothed_points) - 1):
                # Apply weighted average
                prev = smoothed_points[i-1]
                current = smoothed_points[i]
                next_pt = smoothed_points[i+1]
                
                # Calculate weighted average
                neighbor_avg = 0.5 * (prev + next_pt)
                new_point = (1 - weight) * current + weight * neighbor_avg
                new_points.append(new_point)
                
            new_points.append(smoothed_points[-1])  # Keep last point unchanged
            smoothed_points = new_points
            
        return self.__class__(smoothed_points)
    
    def nearest_point_index(self, point: Union[Vector, np.ndarray, Tuple]) -> int:
        """Find the index of the closest point on the path to the given point.
        
        Args:
            point: The reference point
            
        Returns:
            Index of the closest point on the path
        """
        if not self._points:
            raise ValueError("Cannot find nearest point in an empty path")
            
        if not isinstance(point, Vector):
            point = Vector(point)
            
        min_dist = float('inf')
        min_idx = -1
        
        for i, p in enumerate(self._points):
            dist = p.distance_squared(point)  # More efficient than distance()
            if dist < min_dist:
                min_dist = dist
                min_idx = i
                
        return min_idx
    
    def reverse(self: T) -> T:
        """Reverse the path direction.
        
        Returns:
            A new Path object with points in reverse order
        """
        return self.__class__(list(reversed(self._points)))
    
    def __len__(self) -> int:
        """Return the number of points in the path."""
        return len(self._points)
    
    def __getitem__(self, idx) -> Union[Vector, 'Path']:
        """Get a point or slice of points from the path."""
        if isinstance(idx, slice):
            return self.__class__(self._points[idx])
        return self._points[idx]
    
    def __iter__(self) -> Iterator[Vector]:
        """Iterate over the points in the path."""
        return iter(self._points)
    
    def __repr__(self) -> str:
        """String representation of the path."""
        if len(self._points) == 0:
            return "Path([])"
        elif len(self._points) <= 3:
            points_str = ", ".join(repr(p) for p in self._points)
            return f"Path([{points_str}])"
        else:
            return f"Path([{repr(self._points[0])}, ..., {repr(self._points[-1])}]) ({len(self._points)} points)"