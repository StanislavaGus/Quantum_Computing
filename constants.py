import numpy as np

KET_0 = np.array([
    [1],
    [0]
], dtype=complex)

KET_1 = np.array([
    [0],
    [1]
], dtype=complex)

KET_00 = np.kron(KET_0, KET_0)
KET_01 = np.kron(KET_0, KET_1)
KET_10 = np.kron(KET_1, KET_0)
KET_11 = np.kron(KET_1, KET_1)



#матрица Адамара, которая переводит кубит в суперпозицию
H = np.array([
    [1, 1],
    [1, -1]
], dtype=complex) / np.sqrt(2)

def HN(N: int):
    H_N = H.copy()
    for i in range(N - 1):
        H_N = np.kron(H_N, H)
    return H_N

#матрица для операции x (NOT)
X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)


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

PAULI_X = np.array([[0, 1],
                    [1, 0]])

PAULI_Y = np.array([[0, -1j],
                    [1j, 0]])

PAULI_Z = np.array([[1, 0],
                    [0, -1]])


def RX(angle):
    rotation_matrix = np.array([[np.cos(angle / 2), -1j * np.sin(angle / 2)],
                                [-1j * np.sin(angle / 2), np.cos(angle / 2)]])
    return rotation_matrix

P_0 = np.array([[1, 0], [0, 0]])  # |0><0|
P_1 = np.array([[0, 0], [0, 1]])  # |1><1|