# file: training_manager.py

import subprocess
import sys
import os
from datetime import datetime

# Путь к оригинальным скриптам проекта
# Важно: Убедитесь, что эти пути верны для вашей структуры проекта
PREPROCESS_SCRIPT_PATH = os.path.join("finetune", "core", "preprocess_data_conditional_xcodec.py")
TRAIN_LORA_SCRIPT_PATH = os.path.join("finetune", "scripts", "train_lora.py")
TOKENIZER_PATH = os.path.join("finetune", "core", "tokenizer") # Путь к папке с токенайзером

def tokenize_dataset(dataset_path: str, progress=gr.Progress(track_tqdm=True)):
    """
    Запускает скрипт токенизации датасета из .npy в .mmap формат.
    
    Args:
        dataset_path (str): Путь к папке с .npy файлами.
    
    Returns:
        str: Сообщение о статусе выполнения.
    """
    print("Начинается токенизация датасета...")
    if not os.path.isdir(dataset_path):
        return f"ОШИБКА: Директория '{dataset_path}' не найдена."
        
    # Формируем команду для запуска скрипта предобработки
    command = [
        sys.executable,
        PREPROCESS_SCRIPT_PATH,
        "--input", dataset_path,
        "--output-prefix", os.path.join(dataset_path, "tokenized_data"),
        "--tokenizer-type", "MMTokenizer",
        "--tokenizer-path", TOKENIZER_PATH,
        "--workers", str(os.cpu_count() or 1) # Используем все доступные ядра CPU
    ]
    
    print(f"Команда для токенизации: {' '.join(command)}")

    try:
        # Запускаем процесс и ждем его завершения
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')

        # Выводим лог в реальном времени
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        success_message = f"Токенизация успешно завершена! Файлы сохранены в '{dataset_path}'"
        print(success_message)
        return success_message

    except Exception as e:
        error_message = f"ОШИБКА во время токенизации: {e}"
        print(error_message)
        return error_message


def run_training(
    model_name: str,
    base_model_path: str,
    tokenized_dataset_prefix: str,
    lora_rank: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    progress=gr.Progress(track_tqdm=True)
):
    """
    Запускает скрипт дообучения LoRA с параметрами из интерфейса.
    """
    print("Начинается процесс дообучения модели LoRA...")
    
    output_path = os.path.join("lora_models", model_name, datetime.now().strftime("%Y-%m-%d_%H-%M"))
    
    # Формируем команду для запуска скрипта обучения
    command = [
        sys.executable,
        TRAIN_LORA_SCRIPT_PATH,
        "--deepspeed", os.path.join("finetune", "config", "ds_config_zero2.json"),
        "--model-path", base_model_path,
        "--data-path", tokenized_dataset_prefix,
        "--output-path", output_path,
        "--lora-rank", str(lora_rank),
        "--lr", str(learning_rate),
        "--epochs", str(epochs),
        "--micro-batch-size", str(batch_size),
        "--seq-len", "2048",
        "--log-interval", "10",
        "--save-interval", "100",
        "--eval-interval", "100",
        "--train-warmup-steps", "100",
        "--weight-decay", "0.01",
        "--use-lora", "1",
        "--lora-trainable", "q_proj,v_proj",
        "--tokenizer-path", TOKENIZER_PATH,
        "--tokenizer-type", "MMTokenizer",
        "--dataset-impl", "mmap"
    ]

    print(f"Команда для обучения: {' '.join(command)}")

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        
        # Выводим лог в реальном времени
        # (Это позволит нам видеть прогресс обучения прямо в интерфейсе)
        log_content = ""
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                log_line = output.strip()
                print(log_line)
                log_content += log_line + "\n"
                # Используем yield для потоковой передачи логов в Gradio
                yield f"**Лог обучения:**\n\n```\n{log_content}\n```"

        if process.returncode != 0:
             raise subprocess.CalledProcessError(process.returncode, command)

        success_message = f"Обучение успешно завершено! Модель сохранена в: '{output_path}'"
        print(success_message)
        yield success_message
        
    except Exception as e:
        error_message = f"ОШИБКА во время обучения: {e}"
        print(error_message)
        yield error_message