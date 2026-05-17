import numpy as np

np.random.seed(42)

arr = np.random.randint(20, 121, size=(6, 6))
print("Original array:")
print(arr)

col_medians = np.median(arr, axis=0)
print("\nMedian of each column:")
print(col_medians)

modified = arr.copy()
modified[modified < col_medians] = -1
print("\nArray after replacement:")
print(modified)

even_index_elements = arr[::2, ::2]
print("\nElements where both indices are even:")
print(even_index_elements)

col_ranges = arr.max(axis=0) - arr.min(axis=0)
print("\nRange of each column:")
print(col_ranges)

col_min = arr.min(axis=0)
col_max = arr.max(axis=0)
normalized = (arr - col_min) / (col_max - col_min)
print("\nColumn-wise min-max normalized array:")
print(normalized)