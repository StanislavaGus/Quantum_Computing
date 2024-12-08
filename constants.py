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
Z = PAULI_Z

def RX(angle):
    rotation_matrix = np.array([[np.cos(angle / 2), -1j * np.sin(angle / 2)],
                                [-1j * np.sin(angle / 2), np.cos(angle / 2)]])
    return rotation_matrix

P_0 = np.array([[1, 0], [0, 0]])  # |0><0|
P_1 = np.array([[0, 0], [0, 1]])  # |1><1|

import numpy as np


def CNOT_func(N, c, t):
    # Проверяем, что управляющий кубит (c) находится перед целевым кубитом (t)
    # В противном случае поднимается ошибка, так как гейт корректен только при c < t
    if c >= t:
        raise ValueError("Некорректный ввод, должно быть c < t.")

    # Создаем единичную матрицу размером 2^c, которая будет действовать как I ⊗ c
    # Это соответствует всем кубитам, находящимся перед управляющим кубитом
    I_c = np.eye(2 ** c)

    # Проекторы на состояния |0><0| и |1><1|
    # Эти проекторы определяют, в каком состоянии находится управляющий кубит:
    # - Если он в состоянии |0>, то ничего не происходит.
    # - Если он в состоянии |1>, то целевой кубит инвертируется.
    zero_projector = P_0  # Проектор на состояние |0> (управляющий кубит в состоянии 0)
    one_projector = P_1  # Проектор на состояние |1> (управляющий кубит в состоянии 1)

    # Оператор Паули-X (гейт NOT), который инвертирует состояние целевого кубита
    X_gate = PAULI_X

    # Создаем единичную матрицу для всех кубитов между управляющим и целевым кубитом
    # Это будет I ⊗ (t-c-1), соответствующее кубитам между управляющим и целевым
    I_tc = np.eye(2 ** (t - c - 1))

    # Создаем единичную матрицу для всех кубитов, следующих за целевым кубитом
    # Это будет I ⊗ (N-t-1), соответствующее кубитам, идущим после целевого
    I_nt = np.eye(2 ** (N - t - 1))

    # Первая часть терма CNOT:
    # I^⊗c ⊗ |0><0| ⊗ I^(n-c-1) - соответствует ситуации, когда управляющий кубит |0>
    # В этом случае целевой кубит остается неизменным
    term1 = np.kron(np.kron(I_c, zero_projector), np.eye(2 ** (N - c - 1)))

    # Вторая часть терма CNOT:
    # I^⊗c ⊗ |1><1| ⊗ I^⊗(t-c-1) ⊗ X ⊗ I^⊗(n-t-1) - соответствует ситуации, когда управляющий кубит |1>
    # В этом случае к целевому кубиту применяется гейт Паули-X (он инвертирует состояние целевого кубита)
    term2 = np.kron(np.kron(np.kron(np.kron(I_c, one_projector), I_tc), X_gate), I_nt)

    # Итоговая матрица CNOT-гейта - это сумма двух термов:
    # 1. Когда управляющий кубит в состоянии |0> - целевой кубит не изменяется.
    # 2. Когда управляющий кубит в состоянии |1> - состояние целевого кубита инвертируется.
    CNOT_matrix = term1 + term2

    # Возвращаем матрицу CNOT для многокубитной системы
    return CNOT_matrix


"""
Этот код создает матрицу CNOT-гейта (Controlled-NOT) для многокубитной системы, 
где один кубит (управляющий) управляет состоянием другого кубита (целевого).
CNOT-гейт — это двухкубитный гейт, который инвертирует состояние целевого кубита, 
если управляющий кубит находится в состоянии |1>. В противном случае состояние целевого кубита остается неизменным.
"""