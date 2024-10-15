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

def CNOT(N, c, t):
    if c >= t:
        raise ValueError("CNOT generator is correct only with c < t.")
    # based on:
    # https://quantumcomputing.stackexchange.com/questions/4252/how-to-derive-the-cnot-matrix-for-a-3-qubit-system-where-the-control-target-qu
    # AND
    # https://quantumcomputing.stackexchange.com/questions/4078/how-to-construct-a-multi-qubit-controlled-z-from-elementary-gates

    # I ⊗c
    I_c = np.eye(2**c)

    # |0><0| и |1><1|
    zero_projector = P_0
    one_projector = P_1

    # X
    X_gate = PAULI_X

    # I ⊗(t-c-1)
    I_tc = np.eye(2**(t-c-1))

    # I ⊗(n-t-1)
    I_nt = np.eye(2**(N-t-1))

    # I^⊗c ⊗ |0><0| ⊗ I^(n-c-1)
    term1 = np.kron(np.kron(I_c, zero_projector), np.eye(2**(N-c-1)))

    # I^⊗c ⊗ |1><1| ⊗ I^⊗(t-c-1) ⊗ X ⊗ I^⊗(n-t-1)
    term2 = np.kron(np.kron(np.kron(np.kron(I_c, one_projector), I_tc), X_gate), I_nt)

    # CNOT = term1 + term2
    CNOT_matrix = term1 + term2

    return CNOT_matrix

"""
Этот код создает матрицу CNOT-гейта (Controlled-NOT) для многокубитной системы, 
где один кубит управляет состоянием другого.
"""