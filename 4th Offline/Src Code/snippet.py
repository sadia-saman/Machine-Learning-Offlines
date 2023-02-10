import numpy as np

import numpy as np

# Define the larger matrix
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# Define the submatrix by slicing the larger matrix
submatrix = matrix[0:2, 1:3]

# Find the position of the maximum element in the submatrix
max_index = np.unravel_index(submatrix.argmax(), submatrix.shape)

# The position of the maximum element in the submatrix
 
submatrix = matrix[0:2, 1:3] 
print(submatrix)
print(max_index)
print(submatrix.argmax()) 

