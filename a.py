import numpy as np

def calculate_deviation(row_X, row_Y):
    return np.abs(row_X - row_Y).mean()

# Sample input matrices
matrix_X = np.array([[1, 2, 3, 4, 5, 6],
                     [7, 8, 9, 10, 11, 12],
                     [13, 14, 15, 16, 17, 18],
                     [19, 20, 21, 22, 23, 24],
                     [25, 26, 27, 28, 29, 30],
                     [31, 32, 33, 34, 35, 36]])

matrix_Y = np.array([[11, 12, 10, 9, 8, 7],
                     [2, 3, 1, 5, 6, 4],
                     [25, 26, 27, 29, 30, 28],
                     [19, 20, 21, 22, 23, 24],
                     [34, 35, 32, 31, 36, 33],
                     [14, 16, 13, 17, 18, 15]])

best_row_Y = None
min_deviation = float('inf')

for row_X in matrix_X:
    for row_Y in matrix_Y:
        deviation = calculate_deviation(row_X, row_Y)
        if deviation < min_deviation and deviation <= 0.2:
            min_deviation = deviation
            best_row_Y = row_Y

print("Best matching row in matrix Y:")
print(best_row_Y)
