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

#матрица для операции x (NOT)
X = np.array([
[0, 1],
[1, 0]
], dtype=complex) / np.sqrt(2)

#создает матрицу поворота на угол theta.
#Поворот изменяет вероятности того, что кубит окажется в состоянии |0⟩ или |1⟩ при измерении
def rotation_matrix(theta: float) -> np.ndarray:
    """Возвращает матрицу поворота на угол theta."""
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

# двухкубитный гейт CNOT
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)