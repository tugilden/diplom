import numpy as np

def transform_combined_matrix(matrix):
    """
    Функция для преобразования второго столбца матрицы и отражения на единичной матрице
    """
    a2 = matrix[1, 0]
    b2 = matrix[1, 1]

    if b2 == 0:
        matrix[:, 1] += matrix[:, 0]

    while a2 != 0:
        if (a2 > 0 and b2 > 0) or (a2 < 0 and b2 < 0):
            if abs(a2) >= abs(b2):
                matrix[:, 0] -= matrix[:, 1]
            else:
                matrix[:, 1] -= matrix[:, 0]
        else:
            if abs(a2) >= abs(b2):
                matrix[:, 0] += matrix[:, 1]
            else:
                matrix[:, 1] += matrix[:, 0]
        a2 = matrix[1, 0]
        b2 = matrix[1, 1]
    if b2 > 0:
        matrix[:, 1] *= -1
    
    a1 = matrix[0, 0]
    b1 = matrix[0, 1]
    if a1 < 0:
        matrix[:, 0] *= -1

    while b1 < 0:
        matrix[:, 1] += matrix[:, 0]
        b1 = matrix[0][1]

    return matrix


def forward_transform(line1, line2):
    """
    Преобразует неравенства в форму ax+by=c и -y=0
    """
    matrix = np.array([[line1["alpha"], line1["beta"]],
                      [line2["alpha"], line2["beta"]]])
    identity_matrix = np.eye(2)

    combined_matrix = np.vstack((matrix, identity_matrix))
    transformed_combined_matrix = transform_combined_matrix(combined_matrix)

    transformed_matrix = transformed_combined_matrix[:2, :]
    transformed_identity_matrix = transformed_combined_matrix[2:, :]

    alpha = transformed_matrix[0][0]
    beta = transformed_matrix[0][1]
    beta2 = transformed_matrix[1][1]
    gamma = line1["gamma"] + int(line2["gamma"] / abs(beta2)) * transformed_matrix[0][1]

    return alpha, beta, gamma, beta2, transformed_identity_matrix