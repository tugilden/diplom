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
    На вход подаются неравенства в виде координат A1, B1, C1 и A2, B2, C2
    """
    # Преобразуем неравенства в матрицу коэффициентов
    matrix = np.array([[line1["alpha"], line1["beta"]],
                      [line2["alpha"], line2["beta"]]])
    identity_matrix = np.eye(2)

    # Объединяем матрицу коэффициентов с единичной матрицей
    combined_matrix = np.vstack((matrix, identity_matrix))
    transformed_combined_matrix = transform_combined_matrix(combined_matrix)

    # Разделяем преобразованную матрицу
    transformed_matrix = transformed_combined_matrix[:2, :]
    transformed_identity_matrix = transformed_combined_matrix[2:, :]

    # Вычисляем параметры для нового представления
    alpha = transformed_matrix[0][0]
    beta = transformed_matrix[0][1]
    beta2 = transformed_matrix[1][1]
    gamma = line1["gamma"] + int(line2["gamma"] / abs(beta2)) * transformed_matrix[0][1]

    return alpha, beta, gamma, beta2, transformed_identity_matrix


def transform_inequalities(A1, B1, C1, A2, B2, C2):
    """
    Преобразует неравенства в виде A1*x + B1*y <= C1 и A2*x + B2*y <= C2
    в новый вид ax+by=c и -y=0
    
    Parameters:
    A1, B1, C1 - коэффициенты первого неравенства
    A2, B2, C2 - коэффициенты второго неравенства
    
    Returns:
    tuple: (alpha, beta, gamma, beta2, identity_matrix) - преобразованные параметры
    """
    line1 = {"alpha": A1, "beta": B1, "gamma": C1}
    line2 = {"alpha": A2, "beta": B2, "gamma": C2}
    
    return forward_transform(line1, line2)