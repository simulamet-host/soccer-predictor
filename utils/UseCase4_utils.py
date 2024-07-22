import numpy as np

def generate_probability_matrix(size=20):
    # Initialize a random matrix
    matrix = np.random.rand(size, size)
    
    # Normalize rows
    row_sums = matrix.sum(axis=1)
    matrix = matrix / row_sums[:, np.newaxis]
    
    # Normalize columns
    col_sums = matrix.sum(axis=0)
    matrix = matrix / col_sums[np.newaxis, :]
    
    return matrix
