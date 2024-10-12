import numpy as np

KET_0 = np.array([
    [1],
    [0]
], dtype=complex)

#матрица Адамара, которая переводит кубит в суперпозицию
H = np.array([
    [1, 1],
    [1, -1]
], dtype=complex) / np.sqrt(2)

#создает матрицу поворота на угол theta.
#Поворот изменяет вероятности того, что кубит окажется в состоянии |0⟩ или |1⟩ при измерении
def rotation_matrix(theta: float) -> np.ndarray:
    """Возвращает матрицу поворота на угол theta."""
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ], dtype=complex)

# Используем поворотную матрицу для изменения вероятности
R_THETA = rotation_matrix(np.pi / 6)  # Замените на нужный угол для изменения распределения
