import numpy as np

np.random.seed(42)

arr = np.random.randint(10, 100, size=(6, 6))
print("Original array:")
print(arr)

row_means = arr.mean(axis=1, keepdims=True)
print("\nMean of each row:")
print(row_means.ravel())

modified = arr.copy()
modified[modified > row_means] = 0
print("\nModified array:")
print(modified)

third_col = arr[:, 2]
odd_in_third_col = third_col[third_col % 2 == 1]
print("\nOdd-valued elements in the third column:")
print(odd_in_third_col)

row_sums = arr.sum(axis=1)
max_row_index = np.argmax(row_sums)
print("\nIndex of the row with the maximum row sum:")
print(max_row_index)

sorted_rows = np.sort(arr, axis=1)
print("\nEach row sorted independently:")
print(sorted_rows)
