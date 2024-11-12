import numpy as np
from constants import H
from symulator import NQubitSimulator


class GroverAlgorithm:
    def __init__(self, n_qubits, marked_item):
        self.n_qubits = n_qubits
        self.marked_item = marked_item
        self.simulator = NQubitSimulator(n_qubits)

    def apply_hadamard_to_all(self):
        """ Применяет оператор Адамара ко всем кубитам, создавая суперпозицию. """
        for i in range(self.n_qubits):
            self.simulator.apply_single_qubit_gate(H, i)

    def apply_oracle(self):
        """ Оракул меняет фазу целевого состояния. """
        # Используем controlled X с учетом отметки
        control_state = np.eye(2 ** self.n_qubits)
        control_state[self.marked_item, self.marked_item] = -1
        self.simulator.apply_n_qubit_gate(control_state)

    def apply_diffusion_operator(self):
        """ Оператор диффузии для усиления амплитуды целевого состояния. """
        self.apply_hadamard_to_all()
        self.apply_phase_flip()
        self.apply_hadamard_to_all()

    def apply_phase_flip(self):
        """Переворачивает фазу всех состояний кроме |0>."""
        # Создаем состояние |0⟩^{⊗ n} для n-кубитной системы
        KET_0_n = np.zeros((2 ** self.n_qubits, 1))
        KET_0_n[0, 0] = 1  # Состояние |0⟩^{⊗ n}

        # Создаем фазовый оператор, который отражает все состояния относительно |0⟩^{⊗ n}
        phase_flip = 2 * np.outer(KET_0_n, KET_0_n) - np.eye(2 ** self.n_qubits)
        self.simulator.apply_n_qubit_gate(phase_flip)

    def run(self):
        """ Выполнение алгоритма Гровера """
        # Инициализация суперпозиции
        self.apply_hadamard_to_all()

        # Определение количества итераций
        iterations = int(np.floor(np.pi / 4 * np.sqrt(2 ** self.n_qubits)))
        for _ in range(iterations):
            self.apply_oracle()
            self.apply_diffusion_operator()

        # Измерение
        return self.simulator.measure_multiple_qubits(range(self.n_qubits))


def run_experiment(max_qubits=5):
    """Проводит эксперимент с алгоритмом Гровера для различных размерностей устройств и целевых состояний."""
    results = []

    for n_qubits in range(2, max_qubits + 1):  # Варьируем количество кубитов от 2 до max_qubits
        # Количество возможных состояний — это 2^n_qubits
        num_states = 2 ** n_qubits

        # Выбираем случайный индекс для целевого состояния в диапазоне [0, num_states - 1]
        marked_item = np.random.randint(0, num_states)

        # Создаем экземпляр алгоритма Гровера для текущего количества кубитов и целевого состояния
        grover = GroverAlgorithm(n_qubits, marked_item)

        # Запускаем алгоритм и получаем результат измерения
        result_bits = grover.run()

        # Конвертируем результат измерения (список битов) в целое число
        result_index = int("".join(map(str, result_bits)), 2)

        # Сохраняем результат эксперимента для дальнейшего анализа
        results.append({
            "n_qubits": n_qubits,
            "marked_item": marked_item,
            "result_index": result_index,
            "success": result_index == marked_item  # Проверка на успешность
        })

    # Итоговый вывод для анализа
    for res in results:
        print(f"Кубиты: {res['n_qubits']}, Целевой элемент: {res['marked_item']}, "
              f"Найденный индекс: {res['result_index']}, Успешность: {'Успех' if res['success'] else 'Неудача'}")

    return results


# Запуск эксперимента
experiment_results = run_experiment(max_qubits=7)
