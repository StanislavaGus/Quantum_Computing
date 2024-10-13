import numpy as np
from interface import Qubit, QuantumDevice
from constants import H, X, KET_0,rotation_matrix, CNOT


class SimulatedQubit(Qubit):
    def __init__(self):
        self.reset()

    def h(self):
        """Применение матрицы Адамара (50/50 распределение)."""
        self.state = H @ self.state

    def x(self):
        """Применение матрицы х (NOT) чтобы инвертировать состояние кубита"""
        self.state = X @ self.state

    def rotation(self, theta:float):
        """Применение поворотной матрицы для изменения распределения."""
        R_THETA = rotation_matrix(theta)
        self.state = R_THETA @ self.state

    def measure(self) -> bool: #смотри ниже объяснение работы этой функции
        """Измерение состояния кубита."""
        pr0 = np.abs(self.state[0, 0]) ** 2
        #вероятность состояния |0⟩, которая вычисляется как квадрат модуля амплитуды вероятности
        sample = np.random.random() <= pr0
        #генерируется случайное число от 0 до 1. Если оно меньше или равно вероятности состояния |0⟩,
        # результатом будет 0, иначе 1
        return bool(0 if sample else 1)

    def reset(self):
        """Сброс состояния кубита в |0>."""
        self.state = KET_0.copy()


class SingleQubitSimulator(QuantumDevice):
    def allocate_qubit(self) -> Qubit:
        return SimulatedQubit()

    def deallocate_qubit(self, qubit: Qubit):
        pass  # Не нужно ничего делать для симулятора



class TwoQubitSimulator(QuantumDevice):
    qubits = [SimulatedQubit(), SimulatedQubit()]
    state = np.kron(qubits[0].state, qubits[1].state) #тензорное произведение

    def allocate_qubit(self) -> SimulatedQubit:
        if self.qubits:
            return self.qubits.pop()

    def deallocate_qubit(self, qubit: SimulatedQubit):
        self.qubits.append(qubit)

    def apply_single_qubit_gate(self, gate, qubit_idx: int):
        if qubit_idx == 0:
            identity = np.eye(2)
            operation = np.kron(gate, identity)  # gate к 1му
        elif qubit_idx == 1:
            identity = np.eye(2)
            operation = np.kron(identity, gate)  # gate ко 2му
        else:
            raise ValueError("Недопустимый индекс кубита.")

        # Apply operation to quantum state.
        self.state = operation @ self.state

    def apply_two_qubit_gate(self, gate):
        """Применяем двухкубитную операцию к состоянию."""
        self.state = gate @ self.state  # применяем двухкубитную операцию ко всей системе

    def cnot(self):
        self.apply_two_qubit_gate(CNOT)

    def measure(self, qubit_idx: int) -> bool:
        """
        Измерить состояние одного кубита в системе
        """
        if qubit_idx == 0:
            # Вероятность |00> или |01>.
            probability0 = np.abs(self.state[0, 0]) ** 2 + np.abs(self.state[1, 0]) ** 2
        elif qubit_idx == 1:
            # Вероятность |00> или |10>.
            probability0 = np.abs(self.state[0, 0]) ** 2 + np.abs(self.state[2, 0]) ** 2
        else:
            raise ValueError("Недопустимый индекс кубита.")

        is_measured_0 = np.random.random() <= probability0
        return bool(0 if is_measured_0 else 1)

    def reset(self):
        # Сбрасываем состояние каждого кубита индивидуально
        #for qubit in self.qubits:
         #   qubit.reset()
        # Сбрасываем состояние всей системы (два кубита)
        #self.state = np.kron(self.qubits[0].state, self.qubits[1].state)  # Сбрасываем систему в |00> состояние
        self.state = np.kron(KET_0, KET_0)  # Сбрасываем систему в |00> состояние

    def set_state(self, state):
        self.state = state