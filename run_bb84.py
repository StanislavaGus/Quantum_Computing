from bb84 import BB84

if __name__ == "__main__":
    print("Генерирование 96-битового ключа путем симулирования BB84...")
    key = BB84.simulate_bb84(96)
    print(f"Получен ключ {BB84.convert_to_hex(key)}.")
    message = [
    1, 1, 0, 1, 1, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 0, 1,
    1, 1, 0, 1, 1, 1, 0, 0,
    1, 0, 0, 1, 0, 1, 1, 0,
    1, 1, 0, 1, 1, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 0, 1,
    1, 1, 0, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 1,
    1, 1, 0, 1, 1, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 0, 1,
    1, 1, 0, 1, 1, 1, 0, 0,
    1, 0, 1, 1, 1, 0, 1, 1
    ]
    print(f"Использование ключа для отправки секретного сообщения: {BB84.convert_to_hex(message)}.")
    encrypted_message = BB84.apply_one_time_pad(message, key)
    print(f"Зашифрованное сообщение: {BB84.convert_to_hex(encrypted_message)}.")
    decrypted_message = BB84.apply_one_time_pad(encrypted_message, key)
    print(f"Ева расшифровала, получив: {BB84.convert_to_hex(decrypted_message)}.")