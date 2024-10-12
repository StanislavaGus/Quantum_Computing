from typing import List

from interface import QuantumDevice
from interface import Qubit
from symulator import SingleQubitSimulator

class BB84():

    """
    вспомогательные функции перед обменом ключами
    """

    @staticmethod
    def sample_random_bit(device: QuantumDevice) -> bool:
        with device.using_qubit() as q:
            q.h()
            result = q.measure()
            q.reset()
        return result

    @staticmethod
    def prepare_message_qubit(message: bool, basis: bool, q: Qubit) -> None:
        if message:
            q.x()
        if basis:
            q.h() #переводит в базис Адамара (базис: плюс минус)

    @staticmethod
    def measure_message_qubit(basis: bool, q: Qubit) -> bool:
        if basis:
            q.h()
        result = q.measure()
        q.reset()
        return result

    @staticmethod
    def convert_to_hex(bits: List[bool]) -> str:
        return hex(int("".join(["1" if bit else "0" for bit in bits]), 2)) #двоичной системе (основание 2)

    """
    протокол BB84 для отправки классического бита
    """

    @staticmethod
    def send_single_bit_with_bb84(
            your_device: QuantumDevice,
            eve_device: QuantumDevice
        ) -> tuple:

        [your_message, your_basis] = [
            BB84.sample_random_bit(your_device) for _ in range(2)
        ]

        eve_basis = BB84.sample_random_bit(eve_device)

        with your_device.using_qubit() as q:
            BB84.prepare_message_qubit(your_message, your_basis, q)
            # ОТПРАВКА КУБИТА...
            eve_result = BB84.measure_message_qubit(eve_basis, q)

        return ((your_message, your_basis), (eve_result, eve_basis))

    """
    протокол BB84 для обмена ключом с Евой
    """

    @staticmethod
    def simulate_bb84(n_bits: int) -> tuple:
        your_device = SingleQubitSimulator()
        eve_device = SingleQubitSimulator()

        key = []
        n_rounds = 0
        while len(key) < n_bits:
            n_rounds += 1
            ((your_message, your_basis), (eve_result, eve_basis)) = \
                (BB84.send_single_bit_with_bb84(your_device, eve_device))

            if your_basis == eve_basis:
                assert your_message == eve_result #проверка на прослушку
                key.append(your_message)

        print(f"Потребовалось {n_rounds} раундов, чтобы сгенерировать {n_bits} - битовый ключ.")
        return key

    """
    протокол BB84 для обмена ключом с Евой
    """

    def apply_one_time_pad(message: List[bool], key: List[bool]) -> List[bool]: #шифрование
        return [
            message_bit ^ key_bit #XOR
        for (message_bit, key_bit) in zip(message, key)
        ]

    """
    Пример работы zip
    
    message = [True, False, True]
    key = [False, True, True]
    
    zipped = list(zip(message, key))
    print(zipped)
    """