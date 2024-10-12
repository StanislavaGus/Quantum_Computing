from interface import QuantumDevice
class RandomGenerator():
    @staticmethod
    def qrng(device: QuantumDevice) -> bool:
        """Генерация с использованием матрицы Адамара (50/50)."""
        with device.using_qubit() as q:
            q.h()
            return q.measure()

    @staticmethod
    def qrng_with_rotation(device: QuantumDevice) -> bool:
        """Генерация с использованием поворотной матрицы для изменения распределения."""
        with device.using_qubit() as q:
            q.rotation()  # Используем поворотную матрицу для изменения вероятности
            return q.measure()
