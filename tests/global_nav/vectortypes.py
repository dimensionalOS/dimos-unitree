from typing import (
    Union,
    List,
    Tuple,
    TypeVar,
    Sequence,
    Any,
    Protocol,
    runtime_checkable,
    overload,
)
import numpy as np
from vector import Vector


# Protocol approach for static type checking
@runtime_checkable
class VectorLike(Protocol):
    """Protocol for types that can be treated as vectors."""

    def __getitem__(self, key: int) -> float: ...
    def __len__(self) -> int: ...


def to_numpy(value: VectorLike) -> np.ndarray:
    """Convert a vector-compatible value to a numpy array.

    Args:
        value: Any vector-like object (Vector, numpy array, tuple, list)

    Returns:
        Numpy array representation
    """
    if isinstance(value, Vector):
        return value.data
    elif isinstance(value, np.ndarray):
        return value
    else:
        return np.array(value, dtype=float)


def to_vector(value: VectorLike) -> Vector:
    """Convert a vector-compatible value to a Vector object.

    Args:
        value: Any vector-like object (Vector, numpy array, tuple, list)

    Returns:
        Vector object
    """
    if isinstance(value, Vector):
        return value
    else:
        return Vector(value)


def to_tuple(value: VectorLike) -> Tuple[float, ...]:
    """Convert a vector-compatible value to a tuple.

    Args:
        value: Any vector-like object (Vector, numpy array, tuple, list)

    Returns:
        Tuple of floats
    """
    if isinstance(value, Vector):
        return tuple(value.data)
    elif isinstance(value, np.ndarray):
        return tuple(value.tolist())
    elif isinstance(value, tuple):
        return value
    else:
        return tuple(value)


def to_list(value: VectorLike) -> List[float]:
    """Convert a vector-compatible value to a list.

    Args:
        value: Any vector-like object (Vector, numpy array, tuple, list)

    Returns:
        List of floats
    """
    if isinstance(value, Vector):
        return value.data.tolist()
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, list):
        return value
    else:
        return list(value)


# Helper functions to check dimensionality
def is_2d(value: VectorLike) -> bool:
    """Check if a vector-compatible value is 2D.

    Args:
        value: Any vector-like object (Vector, numpy array, tuple, list)

    Returns:
        True if the value is 2D
    """
    if isinstance(value, Vector):
        return len(value) == 2
    elif isinstance(value, np.ndarray):
        return value.shape[-1] == 2 or value.size == 2
    else:
        return len(value) == 2


def is_3d(value: VectorLike) -> bool:
    """Check if a vector-compatible value is 3D.

    Args:
        value: Any vector-like object (Vector, numpy array, tuple, list)

    Returns:
        True if the value is 3D
    """
    if isinstance(value, Vector):
        return len(value) == 3
    elif isinstance(value, np.ndarray):
        return value.shape[-1] == 3 or value.size == 3
    else:
        return len(value) == 3


# Extraction functions for XYZ components
def x(value: VectorLike) -> float:
    """Get the X component of a vector-compatible value.

    Args:
        value: Any vector-like object (Vector, numpy array, tuple, list)

    Returns:
        X component as a float
    """
    if isinstance(value, Vector):
        return value.x
    else:
        return float(to_numpy(value)[0])


def y(value: VectorLike) -> float:
    """Get the Y component of a vector-compatible value.

    Args:
        value: Any vector-like object (Vector, numpy array, tuple, list)

    Returns:
        Y component as a float
    """
    if isinstance(value, Vector):
        return value.y
    else:
        arr = to_numpy(value)
        return float(arr[1]) if len(arr) > 1 else 0.0


def z(value: VectorLike) -> float:
    """Get the Z component of a vector-compatible value.

    Args:
        value: Any vector-like object (Vector, numpy array, tuple, list)

    Returns:
        Z component as a float
    """
    if isinstance(value, Vector):
        return value.z
    else:
        arr = to_numpy(value)
        return float(arr[2]) if len(arr) > 2 else 0.0


if __name__ == "__main__":
    # Test the vector compatibility functions
    print("Testing vectortypes.py conversion functions\n")

    # Create test vectors in different formats
    vector_obj = Vector(1.0, 2.0, 3.0)
    numpy_arr = np.array([4.0, 5.0, 6.0])
    tuple_vec = (7.0, 8.0, 9.0)
    list_vec = [10.0, 11.0, 12.0]

    print("Original values:")
    print(f"Vector:     {vector_obj}")
    print(f"NumPy:      {numpy_arr}")
    print(f"Tuple:      {tuple_vec}")
    print(f"List:       {list_vec}")
    print()

    # Test to_numpy
    print("to_numpy() conversions:")
    print(f"Vector → NumPy:  {to_numpy(vector_obj)}")
    print(f"NumPy → NumPy:   {to_numpy(numpy_arr)}")
    print(f"Tuple → NumPy:   {to_numpy(tuple_vec)}")
    print(f"List → NumPy:    {to_numpy(list_vec)}")
    print()

    # Test to_vector
    print("to_vector() conversions:")
    print(f"Vector → Vector:  {to_vector(vector_obj)}")
    print(f"NumPy → Vector:   {to_vector(numpy_arr)}")
    print(f"Tuple → Vector:   {to_vector(tuple_vec)}")
    print(f"List → Vector:    {to_vector(list_vec)}")
    print()

    # Test to_tuple
    print("to_tuple() conversions:")
    print(f"Vector → Tuple:  {to_tuple(vector_obj)}")
    print(f"NumPy → Tuple:   {to_tuple(numpy_arr)}")
    print(f"Tuple → Tuple:   {to_tuple(tuple_vec)}")
    print(f"List → Tuple:    {to_tuple(list_vec)}")
    print()

    # Test to_list
    print("to_list() conversions:")
    print(f"Vector → List:  {to_list(vector_obj)}")
    print(f"NumPy → List:   {to_list(numpy_arr)}")
    print(f"Tuple → List:   {to_list(tuple_vec)}")
    print(f"List → List:    {to_list(list_vec)}")
    print()

    # Test component extraction
    print("Component extraction:")
    print("x() function:")
    print(f"x(Vector):  {x(vector_obj)}")
    print(f"x(NumPy):   {x(numpy_arr)}")
    print(f"x(Tuple):   {x(tuple_vec)}")
    print(f"x(List):    {x(list_vec)}")
    print()

    print("y() function:")
    print(f"y(Vector):  {y(vector_obj)}")
    print(f"y(NumPy):   {y(numpy_arr)}")
    print(f"y(Tuple):   {y(tuple_vec)}")
    print(f"y(List):    {y(list_vec)}")
    print()

    print("z() function:")
    print(f"z(Vector):  {z(vector_obj)}")
    print(f"z(NumPy):   {z(numpy_arr)}")
    print(f"z(Tuple):   {z(tuple_vec)}")
    print(f"z(List):    {z(list_vec)}")
    print()

    # Test dimension checking
    print("Dimension checking:")
    vec2d = Vector(1.0, 2.0)
    vec3d = Vector(1.0, 2.0, 3.0)
    arr2d = np.array([1.0, 2.0])
    arr3d = np.array([1.0, 2.0, 3.0])

    print(f"is_2d(Vector(1,2)):       {is_2d(vec2d)}")
    print(f"is_2d(Vector(1,2,3)):     {is_2d(vec3d)}")
    print(f"is_2d(np.array([1,2])):   {is_2d(arr2d)}")
    print(f"is_2d(np.array([1,2,3])): {is_2d(arr3d)}")
    print(f"is_2d((1,2)):             {is_2d((1.0, 2.0))}")
    print(f"is_2d((1,2,3)):           {is_2d((1.0, 2.0, 3.0))}")
    print()

    print(f"is_3d(Vector(1,2)):       {is_3d(vec2d)}")
    print(f"is_3d(Vector(1,2,3)):     {is_3d(vec3d)}")
    print(f"is_3d(np.array([1,2])):   {is_3d(arr2d)}")
    print(f"is_3d(np.array([1,2,3])): {is_3d(arr3d)}")
    print(f"is_3d((1,2)):             {is_3d((1.0, 2.0))}")
    print(f"is_3d((1,2,3)):           {is_3d((1.0, 2.0, 3.0))}")
    print()

    # Test the Protocol interface
    print("Testing VectorLike Protocol:")
    print(f"isinstance(Vector(1,2), VectorLike):      {isinstance(vec2d, VectorLike)}")
    print(f"isinstance(np.array([1,2]), VectorLike):  {isinstance(arr2d, VectorLike)}")
    print(
        f"isinstance((1,2), VectorLike):            {isinstance((1.0, 2.0), VectorLike)}"
    )
    print(
        f"isinstance([1,2], VectorLike):            {isinstance([1.0, 2.0], VectorLike)}"
    )
    print()

    # Test mixed operations using different vector types
    # These functions aren't defined in vectortypes, but demonstrate the concept
    def distance(a, b):
        a_np = to_numpy(a)
        b_np = to_numpy(b)
        diff = a_np - b_np
        return np.sqrt(np.sum(diff * diff))

    def midpoint(a, b):
        a_np = to_numpy(a)
        b_np = to_numpy(b)
        return (a_np + b_np) / 2

    print("Mixed operations between different vector types:")
    print(
        f"distance(Vector(1,2,3), [4,5,6]):           {distance(vec3d, [4.0, 5.0, 6.0])}"
    )
    print(
        f"distance(np.array([1,2,3]), (4,5,6)):       {distance(arr3d, (4.0, 5.0, 6.0))}"
    )
    print(f"midpoint(Vector(1,2,3), np.array([4,5,6])): {midpoint(vec3d, numpy_arr)}")
