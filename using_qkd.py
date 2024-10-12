from symulator import SingleQubitSimulator
from random_generator import RandomGenerator as rg
from quantum_key_distribution import QuantumKeyDistribution as qkd_class

qrng_simulator = SingleQubitSimulator()

key_bit = int(rg.qrng(qrng_simulator))

qkd_simulator = SingleQubitSimulator()

with qkd_simulator.using_qubit() as q:
    qkd_class.prepare_classical_message(key_bit, q)
    print(f"Вы подготовили классический бит ключа: {key_bit}")
    eve_measurement = int(qkd_class.eve_measure(q))
    print(f"Ева измерила классический бит ключа: {eve_measurement}")

print("\n\n\n")

qsim = SingleQubitSimulator()
qkd_class.send_classical_bit_wrong_basis(device=qsim, bit=0)