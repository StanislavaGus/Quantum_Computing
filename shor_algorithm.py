import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from fractions import Fraction
from math import gcd
import random


def func21(a, power):
    """
    Создает унитарное преобразование для операции a^(2^power) mod 21.

    Аргументы:
    - a (int): Число a, взаимно простое с 21.
    - power (int): Степень для возведения в a^(2^power).

    Возвращает:
    - c_U (Gate): Контролируемый вентиль унитарного преобразования для a^(2^power) mod 21.
    """
    # Проверка, что a подходит для работы с модулем 21
    assert a in [2, 4, 5, 8, 10, 11, 13, 16, 17,
                 19], "Значение a должно быть одним из [2, 4, 5, 8, 10, 11, 13, 16, 17, 19]."

    # Создаем квантовую схему с четырьмя кубитами, необходимыми для реализации функции
    U = QuantumCircuit(4)

    # Выполняем операцию a^(2^power) mod 21 в зависимости от значения a
    for _ in range(power):
        if a in [2, 19]:  # Операция для a = 2 или 19
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        elif a in [4, 17]:  # Операция для a = 4 или 17
            U.swap(1, 3)
            U.swap(0, 2)
        elif a in [5, 16]:  # Операция для a = 5 или 16
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)

        # Применение X-гейтов для некоторых значений a
        if a in [10, 13, 17]:
            for q in range(4):
                U.x(q)

    # Конвертируем квантовую схему в вентиль
    U = U.to_gate()
    U.name = f"{a}^{power} mod 21"  # Устанавливаем имя вентиля
    c_U = U.control()  # Создаем контролируемую версию вентиля
    return c_U


def take_random(N):
    """
    Выбирает случайное число a, взаимно простое с N. Если gcd(a, N) != 1, возвращает делитель N.

    Аргументы:
    - N (int): Число для факторизации.

    Возвращает:
    - a (int): Случайное число, взаимно простое с N, либо делитель, если найден.
    """
    while True:
        a = random.randint(2, N - 1)  # Случайное значение a в диапазоне от 2 до N-1
        common_divisor = gcd(a, N)  # Вычисляем НОД a и N
        if common_divisor != 1:  # Если НОД(a, N) больше 1, нашли делитель
            print(f"Найден делитель {common_divisor} при выборе a = {a}.")
            return common_divisor  # Возвращаем найденный делитель
        return a  # Возвращаем a, если оно взаимно просто с N


def reverse_qtf(n):
    """
    Реализует обратное квантовое преобразование Фурье.

    Аргументы:
    - n (int): Количество кубитов для преобразования.

    Возвращает:
    - qc (QuantumCircuit): Схема с обратным квантовым преобразованием Фурье.
    """
    qc = QuantumCircuit(n)

    # Меняем порядок кубитов (реверсируем)
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)

    # Применение фазовых сдвигов и гейтов Адамара
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / float(2 ** (j - m)), m, j)  # Фазовый сдвиг между кубитами
        qc.h(j)  # Гейт Адамара для создания суперпозиции
    qc.name = "QFT-1"  # Название схемы
    return qc


def detection_period(a, N, shots=1):
    """
    Квантовый алгоритм Шора для нахождения периода функции a^x mod N.

    Аргументы:
    - a (int): Параметр функции a.
    - N (int): Число для факторизации.
    - shots (int): Количество симуляций для измерения.

    Возвращает:
    - r (int): Найденный период.
    """
    n_count = 8  # Число расчетных кубитов

    # Создаем квантовую схему с расчетными и вспомогательными кубитами
    qc = QuantumCircuit(QuantumRegister(8, 'x'), QuantumRegister(4, 'f(x)'), ClassicalRegister(8))

    # Инициализация расчетных кубитов в состоянии |+> для суперпозиции
    for q in range(n_count):
        qc.h(q)

    # Устанавливаем вспомогательный регистр в состояние |1>
    qc.x(3 + n_count)

    # Применяем контролируемые операции func21 для нахождения a^(2^power) mod N
    for q in range(n_count):
        qc.append(func21(a, 2 ** q), [q] + [i + n_count for i in range(4)])


    # Применяем обратное квантовое преобразование Фурье (QFT)
    qc.append(reverse_qtf(n_count), range(n_count))

    # Измерение расчетных кубитов
    qc.measure(range(n_count), range(n_count))

    # Вывод квантовой схемы
    print("\nКвантовая схема для текущего a:")
    print(qc)

    # Симуляция на квантовом симуляторе
    qasm_sim = Aer.get_backend('aer_simulator')
    t_qc = transpile(qc, qasm_sim)
    result = qasm_sim.run(t_qc, shots=shots).result()
    counts = result.get_counts()  # Получаем результаты измерений

    # Поиск состояния с максимальной вероятностью
    max_state = max(counts, key=counts.get)
    phase = int(max_state, 2) / (2 ** n_count)
    frac = Fraction(phase).limit_denominator(N)  # Преобразуем фазу в дробь
    r = frac.denominator  # Период - это знаменатель дроби
    if r == 0:
        raise ValueError("Невозможно найти период. Попробуйте снова.")
    return r


def factorization(N):
    """
    Алгоритм факторизации числа с использованием квантового метода для нахождения делителей.

    Аргументы:
    - N (int): Число, которое нужно факторизовать.
    """
    attempt = 0  # Счетчик попыток

    while True:
        attempt += 1  # Увеличиваем номер попытки

        # Выбираем случайное значение a
        a = take_random(N)
        print(f"\nПопытка {attempt}: \nСлучайное а = {a}")

        # Если сразу найден делитель при выборе a, завершаем алгоритм
        if isinstance(a, int) and gcd(a, N) != 1:
            print(f"Простые делители числа {N}: {a} и {N // a}")
            return

        try:
            r = detection_period(a, N)  # Запускаем квантовый алгоритм для нахождения периода
            print(f"Период: r = {r}")
            if r % 2 != 0 or r == 0:  # Проверка на четность и ненулевое значение периода
                print("Неправильный период. Пробуем снова.")
                continue

            # Попытка нахождения делителей на основе найденного периода
            guesses = [gcd(a ** (r // 2) - 1, N), gcd(a ** (r // 2) + 1, N)]
            for guess in guesses:
                if guess in [1, N]:  # Пропускаем тривиальные делители
                    print(f"Угаданный делитель {guess} является тривиальным. Пропускаем.")
                    continue
                if (N % guess) == 0:  # Проверяем, является ли предположение делителем
                    print(f"Простые делители числа {N}: {guess} и {N // guess}")
                    print("Факторизация завершена.")
                    return
                else:
                    print(f"Предположение {guess} некорректно.")
        except ValueError as e:
            print(f"Ошибка: {e}. Пробуем снова.")


# Запускаем факторизацию числа N
N = 21  # Число, которое необходимо факторизовать
factorization(N)
