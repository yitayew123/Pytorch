# PyTorch Tensor Basics

This repository provides an introduction to PyTorch tensors, the fundamental data structures used in deep learning and machine learning. Tensors generalize matrices to higher dimensions and are used to store and manipulate data efficiently, especially for GPU-accelerated computations.

## What Are Tensors?
Tensors are multi-dimensional arrays that represent data in various dimensions:

1. **Scalar (0-D Tensor)**: A single numerical value.
2. **Vector (1-D Tensor)**: A sequence of values, like a list.
3. **Matrix (2-D Tensor)**: A table of values with rows and columns.
4. **3-D Tensor**: Data with three dimensions, such as images.
5. **N-D Tensor**: Higher-dimensional data, like video frames or batches of data.

## Example Code
Below are examples of creating and working with tensors in PyTorch:

### Scalar (0-D Tensor)
```python
import torch

scalar = torch.tensor(5)
print(scalar)  # Output: tensor(5)
print(scalar.shape)  # Output: torch.Size([])
```

### Vector (1-D Tensor)
```python
vector = torch.tensor([1, 2, 3])
print(vector)  # Output: tensor([1, 2, 3])
print(vector.shape)  # Output: torch.Size([3])
```

### Matrix (2-D Tensor)
```python
matrix = torch.tensor([[1, 2], [3, 4]])
print(matrix)
# Output:
# tensor([[1, 2],
#         [3, 4]])
print(matrix.shape)  # Output: torch.Size([2, 2])
```

### 3-D Tensor
```python
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(tensor_3d)
# Output:
# tensor([[[1, 2],
#          [3, 4]],
#         [[5, 6],
#          [7, 8]]])
print(tensor_3d.shape)  # Output: torch.Size([2, 2, 2])
```

### N-D Tensor
```python
tensor_4d = torch.rand(2, 3, 4, 5)  # Random 4-D tensor
print(tensor_4d.shape)  # Output: torch.Size([2, 3, 4, 5])
```

## Tensor Data Types
PyTorch tensors support various data types, such as:
- `float32` (default for floating-point numbers)
- `int32`, `int64` (for integers)
- `bool` (for binary values)
- `complex64` (for complex numbers)

### Example
```python
float_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)

print(float_tensor.dtype)  # Output: torch.float32
print(int_tensor.dtype)    # Output: torch.int32
```

## Practical Use Cases
- **Scalar**: Represent a single measurement (e.g., temperature).
- **Vector**: Store features for a single data point.
- **Matrix**: Represent weights in a neural network layer.
- **3-D Tensor**: Represent grayscale or color images.
- **N-D Tensor**: Handle complex data like videos or higher-dimensional feature spaces.

## Requirements
- Python 3.8+
- PyTorch 1.10+

## Installation
To install PyTorch, follow the official installation guide: [PyTorch.org](https://pytorch.org/get-started/locally/)

## License
This project is licensed under the MIT License. Feel free to use and modify the code.

## Acknowledgments
This work is part of an educational resource to introduce PyTorch tensors for deep learning enthusiasts.
