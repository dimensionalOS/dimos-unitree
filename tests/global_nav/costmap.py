import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any, List, Union
from scipy import ndimage
import heapq


class Costmap:
    """Class to hold costmap data from ROS OccupancyGrid messages."""

    def __init__(
        self,
        grid: np.ndarray,
        origin_x: float,
        origin_y: float,
        origin_theta: float,
        resolution: float = 0.05,
    ):
        """Initialize Costmap with its core attributes."""
        self.grid = grid
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.origin_theta = origin_theta
        self.width = self.grid.shape[1]
        self.height = self.grid.shape[0]

    @classmethod
    def from_msg(cls, costmap_msg) -> "Costmap":
        """Create a Costmap instance from a ROS OccupancyGrid message."""
        if costmap_msg is None:
            return cls(
                grid=np.zeros((100, 100), dtype=np.int8),
                resolution=0.1,
                origin_x=0.0,
                origin_y=0.0,
                origin_theta=0.0,
            )

        # Extract info from the message
        width = costmap_msg.info.width
        height = costmap_msg.info.height
        resolution = costmap_msg.info.resolution

        origin_x = costmap_msg.info.origin.position.x
        origin_y = costmap_msg.info.origin.position.y

        # Calculate orientation from quaternion
        qx = costmap_msg.info.origin.orientation.x
        qy = costmap_msg.info.origin.orientation.y
        qz = costmap_msg.info.origin.orientation.z
        qw = costmap_msg.info.origin.orientation.w
        origin_theta = math.atan2(
            2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)
        )

        # Convert to numpy array
        data = np.array(costmap_msg.data, dtype=np.int8)
        grid = data.reshape((height, width))

        return cls(
            grid=grid,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_theta=origin_theta,
        )

    @classmethod
    def from_pickle(cls, pickle_path: str) -> "Costmap":
        """Load costmap from a pickle file containing a ROS OccupancyGrid message."""
        with open(pickle_path, "rb") as f:
            costmap_msg = pickle.load(f)
        return cls.from_msg(costmap_msg)

    @classmethod
    def create_empty(
        cls, width: int = 100, height: int = 100, resolution: float = 0.1
    ) -> "Costmap":
        """Create an empty costmap with specified dimensions."""
        return cls(
            grid=np.zeros((height, width), dtype=np.int8),
            resolution=resolution,
            origin_x=0.0,
            origin_y=0.0,
            origin_theta=0.0,
        )

    # TODO: this needs to use some generic vector ops, does dimos have those?
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)
        return grid_x, grid_y

    # TODO: this needs to use some generic vector ops, does dimos have those?
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        world_x = grid_x * self.resolution + self.origin_x
        world_y = grid_y * self.resolution + self.origin_y
        return world_x, world_y

    def is_occupied(self, x: float, y: float, threshold: int = 50) -> bool:
        """Check if a position in world coordinates is occupied.

        Args:
            x: X coordinate in world frame
            y: Y coordinate in world frame
            threshold: Cost threshold above which a cell is considered occupied (0-100)

        Returns:
            True if position is occupied or out of bounds, False otherwise
        """
        grid_x, grid_y = self.world_to_grid(x, y)
        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            # Consider unknown (-1) as unoccupied for navigation purposes
            value = self.grid[grid_y, grid_x]
            return value > 0 and value >= threshold
        return True  # Consider out-of-bounds as occupied

    def get_value(self, x: float, y: float) -> Optional[int]:
        """Get the value at a position in world coordinates."""
        grid_x, grid_y = self.world_to_grid(x, y)
        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            return int(self.grid[grid_y, grid_x])
        return None

    def set_value(self, x: float, y: float, value: int) -> bool:
        """Set the value at a position in world coordinates."""
        grid_x, grid_y = self.world_to_grid(x, y)
        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            self.grid[grid_y, grid_x] = value
            return True
        return False

    def smudge(
        self,
        kernel_size: int = 5,
        iterations: int = 10,
        decay_factor: float = 0.8,
        threshold: int = 1,
        preserve_unknown: bool = True,
    ) -> "Costmap":
        """
        Creates a new costmap with expanded obstacles (smudged).

        Args:
            kernel_size: Size of the convolution kernel for dilation (must be odd)
            iterations: Number of dilation iterations
            decay_factor: Factor to reduce cost as distance increases (0.0-1.0)
            threshold: Minimum cost value to consider as an obstacle for expansion
            preserve_unknown: Whether to keep unknown (-1) cells as unknown

        Returns:
            A new Costmap instance with expanded obstacles
        """
        # Make sure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create a copy of the grid for processing
        grid_copy = self.grid.copy()

        # Create a mask of unknown cells if needed
        unknown_mask = None
        if preserve_unknown:
            unknown_mask = grid_copy == -1
            # Temporarily replace unknown cells with 0 for processing
            grid_copy[unknown_mask] = 0

        # Create a mask of cells that are above the threshold
        obstacle_mask = grid_copy >= threshold

        # Create a binary map of obstacles
        binary_map = obstacle_mask.astype(np.uint8) * 100

        # Create a circular kernel for dilation (instead of square)
        y, x = np.ogrid[
            -kernel_size // 2 : kernel_size // 2 + 1,
            -kernel_size // 2 : kernel_size // 2 + 1,
        ]
        kernel = (x * x + y * y <= (kernel_size // 2) * (kernel_size // 2)).astype(
            np.uint8
        )

        # Create distance map using dilation
        # Each iteration adds one 'ring' of cells around obstacles
        dilated_map = binary_map.copy()

        # Store each layer of dilation with decreasing values
        layers = []

        # First layer is the original obstacle cells
        layers.append(binary_map.copy())

        for i in range(iterations):
            # Dilate the binary map
            dilated = ndimage.binary_dilation(
                dilated_map > 0, structure=kernel, iterations=1
            ).astype(np.uint8)

            # Calculate the new layer (cells that were just added in this iteration)
            new_layer = (dilated - (dilated_map > 0).astype(np.uint8)) * 100

            # Apply decay factor based on distance from obstacle
            new_layer = new_layer * (decay_factor ** (i + 1))

            # Add to layers list
            layers.append(new_layer)

            # Update dilated map for next iteration
            dilated_map = dilated * 100

        # Combine all layers to create a distance-based cost map
        smudged_map = np.zeros_like(grid_copy)
        for layer in layers:
            # For each cell, keep the maximum value across all layers
            smudged_map = np.maximum(smudged_map, layer)

        # Preserve original obstacles
        smudged_map[obstacle_mask] = grid_copy[obstacle_mask]

        # Restore unknown cells if needed
        if preserve_unknown and unknown_mask is not None:
            smudged_map[unknown_mask] = -1

        # Ensure cost values are in valid range (0-100) except for unknown (-1)
        if preserve_unknown:
            valid_cells = ~unknown_mask
            smudged_map[valid_cells] = np.clip(smudged_map[valid_cells], 0, 100)
        else:
            smudged_map = np.clip(smudged_map, 0, 100)

        # Create a new costmap with the smudged grid
        return Costmap(
            grid=smudged_map.astype(np.int8),
            resolution=self.resolution,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            origin_theta=self.origin_theta,
        )

    def plot(
        self,
        figsize=(10, 8),
        show_axes=True,
        # cmap="viridis",
        cmap="YlOrRd",
        title="Costmap Visualization",
        show=True,
        unknown_color="lightgray",
        show_grid=True,
        grid_interval=1.0,
        grid_color="black",
        grid_alpha=0.5,
        additional_points: Dict[str, Dict[str, Any]] = {},
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)

        additional_points["origin"] = {
            "positions": [(0, 0)],  # Example position
            "color": "red",
            "marker": "x",
            "size": 100,
        }

        # Create the extent for proper world coordinate mapping
        extent = [
            self.origin_x,
            self.origin_x + self.width * self.resolution,
            self.origin_y,
            self.origin_y + self.height * self.resolution,
        ]

        # Create a masked array to handle unknown (-1) cells separately
        grid_copy = self.grid.copy()
        unknown_mask = grid_copy == -1

        # Create a custom colormap with a specific color for unknown cells
        norm = plt.Normalize(
            vmin=0, vmax=100
        )  # Adjust vmax based on your costmap values

        # Plot the known costs
        masked_grid = np.ma.array(grid_copy, mask=unknown_mask)
        im = ax.imshow(
            masked_grid,
            cmap=cmap,
            norm=norm,
            origin="lower",
            extent=extent,
            interpolation="none",
        )

        # Plot the unknown cells with a different color
        if np.any(unknown_mask):
            unknown_grid = np.ma.array(np.zeros_like(grid_copy), mask=~unknown_mask)
            ax.imshow(
                unknown_grid,
                cmap=plt.matplotlib.colors.ListedColormap([unknown_color]),
                origin="lower",
                extent=extent,
                interpolation="none",
            )

        # Add meter grid overlay
        if show_grid:
            # Calculate grid line positions for X and Y axes
            x_min, x_max = extent[0], extent[1]
            y_min, y_max = extent[2], extent[3]

            # Round to nearest grid_interval for cleaner display
            x_start = math.ceil(x_min / grid_interval) * grid_interval
            y_start = math.ceil(y_min / grid_interval) * grid_interval

            # Draw vertical grid lines (constant x-value)
            x_lines = np.arange(x_start, x_max, grid_interval)
            for x in x_lines:
                ax.axvline(
                    x=x,
                    color=grid_color,
                    linestyle="--",
                    linewidth=0.5,
                    alpha=grid_alpha,
                    zorder=1,
                )

            # Draw horizontal grid lines (constant y-value)
            y_lines = np.arange(y_start, y_max, grid_interval)
            for y in y_lines:
                ax.axhline(
                    y=y,
                    color=grid_color,
                    linestyle="--",
                    alpha=grid_alpha,
                    zorder=1,
                    linewidth=0.5,
                )

            # Add labeled tick marks at grid intervals
            ax.set_xticks(x_lines)
            ax.set_yticks(y_lines)

        # Add colorbar for known cells
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Cost Value (0-100)")

        # Plot additional points if provided
        if additional_points:
            for label, point_data in additional_points.items():
                positions = point_data.get("positions", [])
                color = point_data.get("color", "red")
                marker = point_data.get("marker", "o")
                size = point_data.get("size", 50)

                # Extract x and y coordinates
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]

                ax.scatter(
                    x_coords, y_coords, c=color, marker=marker, s=size, label=label
                )

        # Set labels and title
        ax.set_xlabel("X (world coordinates)")
        ax.set_ylabel("Y (world coordinates)")
        ax.set_title(title)

        # Hide axes if requested
        if not show_axes:
            ax.set_axis_off()

        # Show the plot
        if show:
            plt.tight_layout()
            plt.show(block=False)

        return fig

    def astar(
        self,
        goal_x: float,
        goal_y: float,
        start_x: float = 0.0,
        start_y: float = 0.0,
        cost_threshold: int = 50,
        allow_diagonal: bool = True,
    ) -> List[Tuple[float, float]]:
        """
        A* path planning algorithm from start to goal position.

        Args:
            goal_x: Goal position X coordinate in world frame
            goal_y: Goal position Y coordinate in world frame
            start_x: Start position X coordinate in world frame (default: 0.0)
            start_y: Start position Y coordinate in world frame (default: 0.0)
            cost_threshold: Cost threshold above which a cell is considered an obstacle
            allow_diagonal: Whether to allow diagonal movements

        Returns:
            List of waypoints as (x, y) tuples in world coordinates, or empty list if no path found
        """
        # Convert world coordinates to grid coordinates
        start_grid_x, start_grid_y = self.world_to_grid(start_x, start_y)
        goal_grid_x, goal_grid_y = self.world_to_grid(goal_x, goal_y)

        # Check if start or goal is out of bounds or in an obstacle
        if not (
            0 <= start_grid_x < self.width and 0 <= start_grid_y < self.height
        ) or not (0 <= goal_grid_x < self.width and 0 <= goal_grid_y < self.height):
            print("Start or goal position is out of bounds")
            return []

        # Check if start or goal is in an obstacle
        if (
            self.grid[start_grid_y, start_grid_x] >= cost_threshold
            or self.grid[goal_grid_y, goal_grid_x] >= cost_threshold
        ):
            print("Start or goal position is in an obstacle")
            return []

        # Define possible movements (8-connected grid)
        if allow_diagonal:
            # 8-connected grid: horizontal, vertical, and diagonal movements
            directions = [
                (0, 1),
                (1, 0),
                (0, -1),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]
        else:
            # 4-connected grid: only horizontal and vertical movements
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Cost for each movement (straight vs diagonal)
        movement_costs = (
            [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
            if allow_diagonal
            else [1.0, 1.0, 1.0, 1.0]
        )

        # A* algorithm implementation
        open_set = []  # Priority queue for nodes to explore
        closed_set = set()  # Set of explored nodes

        # Dictionary to store cost from start and parents for each node
        g_score = {(start_grid_x, start_grid_y): 0}
        parents = {}

        # Heuristic function (Euclidean distance)
        def heuristic(x1, y1, x2, y2):
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Start with the starting node
        f_score = g_score[(start_grid_x, start_grid_y)] + heuristic(
            start_grid_x, start_grid_y, goal_grid_x, goal_grid_y
        )
        heapq.heappush(open_set, (f_score, (start_grid_x, start_grid_y)))

        while open_set:
            # Get the node with the lowest f_score
            _, current = heapq.heappop(open_set)
            current_x, current_y = current

            # Check if we've reached the goal
            if current == (goal_grid_x, goal_grid_y):
                # Reconstruct the path
                path = []
                while current in parents:
                    world_x, world_y = self.grid_to_world(current[0], current[1])
                    path.append((world_x, world_y))
                    current = parents[current]

                # Add the start position
                world_x, world_y = self.grid_to_world(start_grid_x, start_grid_y)
                path.append((world_x, world_y))

                # Reverse the path (start to goal)
                path.reverse()

                # Add the goal position if it's not already included
                world_x, world_y = self.grid_to_world(goal_grid_x, goal_grid_y)
                if not path or path[-1] != (world_x, world_y):
                    path.append((world_x, world_y))

                return path

            # Add current node to closed set
            closed_set.add(current)

            # Explore neighbors
            for i, (dx, dy) in enumerate(directions):
                neighbor_x, neighbor_y = current_x + dx, current_y + dy
                neighbor = (neighbor_x, neighbor_y)

                # Check if the neighbor is valid
                if not (0 <= neighbor_x < self.width and 0 <= neighbor_y < self.height):
                    continue

                # Check if the neighbor is already explored
                if neighbor in closed_set:
                    continue

                # Check if the neighbor is an obstacle
                if self.grid[neighbor_y, neighbor_x] >= cost_threshold:
                    continue

                # Calculate g_score for the neighbor
                tentative_g_score = g_score[current] + movement_costs[i]

                # Get the current g_score for the neighbor or set to infinity if not yet explored
                neighbor_g_score = g_score.get(neighbor, float("inf"))

                # If this path to the neighbor is better than any previous one
                if tentative_g_score < neighbor_g_score:
                    # Update the neighbor's scores and parent
                    parents[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(
                        neighbor_x, neighbor_y, goal_grid_x, goal_grid_y
                    )

                    # Add the neighbor to the open set with its f_score
                    heapq.heappush(open_set, (f_score, neighbor))

        # If we get here, no path was found
        return []

    def plot_path(self, path: List[Tuple[float, float]], **kwargs) -> plt.Figure:
        """
        Plot the costmap with a path overlay.

        Args:
            path: List of (x, y) tuples representing the path in world coordinates
            **kwargs: Additional arguments to pass to the plot method

        Returns:
            The matplotlib Figure object
        """
        # Create a dictionary of additional points for the plot method
        additional_points = kwargs.pop("additional_points", {})

        # Add the path to the additional_points dictionary
        if path:
            start_point = path[0]
            goal_point = path[-1]

            # Add start and goal points
            additional_points["start"] = {
                "positions": [start_point],
                "color": "green",
                "marker": "o",
                "size": 100,
            }

            additional_points["goal"] = {
                "positions": [goal_point],
                "color": "blue",
                "marker": "s",  # square
                "size": 100,
            }

            # Add the path
            additional_points["path"] = {
                "positions": path,
                "color": "lime",
                "marker": ".",
                "size": 50,
            }

        # Call the regular plot method with the additional points
        return self.plot(additional_points=additional_points, **kwargs)


if __name__ == "__main__":
    costmap = Costmap.from_pickle("costmapMsg.pickle")
    print(f"Costmap dimensions: {costmap.width}x{costmap.height}")
    print(f"Resolution: {costmap.resolution}")
    print(f"Origin: ({costmap.origin_x}, {costmap.origin_y}, {costmap.origin_theta})")

    # Create a smudged version of the costmap for better planning
    smudged_costmap = costmap.smudge(
        kernel_size=10, iterations=10, threshold=80, preserve_unknown=False
    )

    # Plan a path from origin (0,0) to a goal point (adjust these coordinates as needed)
    goal_x, goal_y = 5.0, -7.0

    # Find path using A* algorithm
    path = smudged_costmap.astar(goal_x=goal_x, goal_y=goal_y, cost_threshold=80)

    if path:
        print(f"Path found with {len(path)} waypoints")
        # Plot the costmap with the path
        smudged_costmap.plot_path(path, title="A* Path on Costmap")
    else:
        print("No path found")
        # Plot the costmap without a path
        smudged_costmap.plot(title="No Path Found")

    # Block, wait for input and exit
    input("Press Enter to exit...")
