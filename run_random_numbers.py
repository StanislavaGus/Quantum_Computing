import numpy as np
from symulator import SingleQubitSimulator
from random_generator import RandomGenerator as rg


if __name__ == "__main__":
    qsim = SingleQubitSimulator()

    total_samples = 10000  # Количество генераций для каждой методики
    count_zeros_hadamard = 0
    count_ones_hadamard = 0
    count_zeros_rotation = 0
    count_ones_rotation = 0

    # Генерация с использованием матрицы Адамара (50/50)
    print("Генерация с использованием матрицы Адамара (50/50):")
    for idx_sample in range(total_samples):
        random_sample = rg.qrng(qsim)
        if random_sample == 0:
            count_zeros_hadamard += 1
        else:
            count_ones_hadamard += 1

        #print(f"QRNG-генератор вернул {random_sample}.")

    # Подсчет процентов для Адамара
    percentage_zeros_hadamard = (count_zeros_hadamard / total_samples) * 100
    percentage_ones_hadamard = (count_ones_hadamard / total_samples) * 100

    print("\nРезультаты генерации с матрицей Адамара:")
    print(f"Количество 0: {count_zeros_hadamard} ({percentage_zeros_hadamard:.2f}%)")
    print(f"Количество 1: {count_ones_hadamard} ({percentage_ones_hadamard:.2f}%)\n")

    # Генерация с использованием поворотной матрицы
    print("Генерация с использованием поворотной матрицы:")
    for idx_sample in range(total_samples):
        random_sample = rg.qrng_with_rotation(qsim, np.pi / 3)
        if random_sample == 0:
            count_zeros_rotation += 1
        else:
            count_ones_rotation += 1

    # Подсчет процентов для поворотной матрицы
    percentage_zeros_rotation = (count_zeros_rotation / total_samples) * 100
    percentage_ones_rotation = (count_ones_rotation / total_samples) * 100

    print("\nРезультаты генерации с поворотной матрицей:")
    print(f"Количество 0: {count_zeros_rotation} ({percentage_zeros_rotation:.2f}%)")
    print(f"Количество 1: {count_ones_rotation} ({percentage_ones_rotation:.2f}%)")
