# file: inference_manager.py (обновленная, финальная версия)

import gradio as gr
import subprocess
import sys
import os
import numpy as np
import soundfile as sf
from datetime import datetime

INFERENCE_SCRIPT_PATH = os.path.join("inference", "infer.py")

# --- Новые и обновленные функции ---

def get_available_models(model_dir="pretrained_models"):
    """Находит все доступные модели (базовые и exllamav2)."""
    # ... (логика без изменений) ...
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        return ["/path/to/your/model"]
    models = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    if not models:
        return ["/path/to/your/model"]
    return models

def get_available_lora_models(lora_dir="lora_models"):
    """Находит все LoRA модели."""
    # ... (логика без изменений) ...
    if not os.path.isdir(lora_dir): return ["Отсутствует"]
    lora_files = ["Отсутствует"]
    for root, _, files in os.walk(lora_dir):
        for file in files:
            if file.endswith((".pt", ".bin", ".safetensors")):
                lora_files.append(os.path.join(root, file))
    return lora_files

def generate_silence(duration_seconds: float, sample_rate: int = 44100):
    """
    НОВАЯ ФУНКЦИЯ: Генерирует .wav файл с абсолютной тишиной заданной длительности.
    Возвращает путь к созданному файлу.
    """
    print(f"Генерация файла тишины длительностью {duration_seconds} сек.")
    silence_dir = os.path.join("temp_audio")
    os.makedirs(silence_dir, exist_ok=True)
    
    num_samples = int(duration_seconds * sample_rate)
    silence = np.zeros(num_samples, dtype=np.float32)
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    file_path = os.path.join(silence_dir, f"silence_{timestamp}.wav")
    
    sf.write(file_path, silence, sample_rate)
    print(f"Файл тишины сохранен в: {file_path}")
    return file_path

def run_inference(
    # Основные параметры
    base_model_path: str,
    lora_model_path: str,
    is_exllamav2: bool,
    # Параметры кэша
    stage1_cache_size: int,
    stage2_cache_size: int,
    # Параметры промпта
    prompt_type: str,
    text_prompt: str,
    single_audio_prompt_path: str,
    single_audio_prompt_type: str,
    instrumental_prompt_path: str,
    vocal_prompt_path: str,
    prompt_start_time: float,
    prompt_end_time: float,
    # Параметры генерации
    duration: int,
    max_new_tokens: int,
    stage1_guidance_scale: float,
    stage1_top_p: float,
    stage1_temperature: float,
    stage1_repetition_penalty: float,
    progress=gr.Progress(track_tqdm=True)
):
    """
    Запускает скрипт генерации музыки со всеми новыми параметрами.
    """
    print("Начинается процесс генерации музыки с расширенными параметрами...")
    
    output_filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    output_path = os.path.join("audio_outputs", output_filename)
    os.makedirs("audio_outputs", exist_ok=True)

    # --- Сборка команды ---
    command = [
        sys.executable, INFERENCE_SCRIPT_PATH,
        "--model-path", base_model_path,
        "--output-path", output_path,
        "--duration", str(duration),
        "--max-new-tokens", str(max_new_tokens),
        # Новые параметры кэша
        "--stage1-cache-size", str(stage1_cache_size),
        "--stage2-cache-size", str(stage2_cache_size),
        # Новые параметры генерации Stage 1
        "--stage1-guidance-scale", str(stage1_guidance_scale),
        "--stage1-top-p", str(stage1_top_p),
        "--stage1-temperature", str(stage1_temperature),
        "--stage1-repetition-penalty", str(stage1_repetition_penalty),
    ]

    # Добавляем тип модели, если это exllamav2
    if is_exllamav2:
        command.append("--is-exllamav2")

    # Добавляем LoRA, если выбрана
    if lora_model_path and lora_model_path != "Отсутствует":
        command.extend(["--lora-path", lora_model_path])
        
    # --- Логика обработки аудио-промптов ---
    final_instrumental_path = None
    final_vocal_path = None
    use_audio_prompt_flag = False

    if prompt_type == "Одиночный аудио-промпт" and single_audio_prompt_path:
        use_audio_prompt_flag = True
        prompt_duration = prompt_end_time - prompt_start_time
        if prompt_duration <= 0:
            yield "Ошибка: 'End Time' должен быть больше 'Start Time'.", None
            return
            
        silent_path = generate_silence(prompt_duration)
        if single_audio_prompt_type == "Инструментал":
            final_instrumental_path = single_audio_prompt_path
            final_vocal_path = silent_path
        else: # Вокал
            final_instrumental_path = silent_path
            final_vocal_path = single_audio_prompt_path
            
    elif prompt_type == "Двойной аудио-промпт" and instrumental_prompt_path and vocal_prompt_path:
        use_audio_prompt_flag = True
        command.append("--use-dual-tracks-prompt") # Отдельный флаг для двойного промпта
        final_instrumental_path = instrumental_prompt_path
        final_vocal_path = vocal_prompt_path

    # Добавляем параметры аудио-промпта в команду, если он используется
    if use_audio_prompt_flag:
        command.extend([
            "--use-audio-prompt",
            "--instrumental-track-prompt-path", final_instrumental_path,
            "--vocal-track-prompt-path", final_vocal_path,
            "--prompt-start-time", str(prompt_start_time),
            "--prompt-end-time", str(prompt_end_time),
        ])
    else: # Если используется текстовый промпт
        command.extend(["--prompt", f"\"{text_prompt}\""])


    # --- Запуск процесса ---
    print(f"Финальная команда для генерации: {' '.join(command)}")
    status = "Начинаю генерацию... Это может занять некоторое время."
    yield status, None

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        # ... (логика вывода логов и обработки ошибок без изменений) ...
        log_content = ""
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None: break
            if output:
                print(output.strip())
                log_content += output.strip() + "\n"
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, output=log_content)

        success_message = f"✅ Генерация успешно завершена! Аудио сохранено в: '{output_path}'"
        yield success_message, output_path

    except Exception as e:
        error_message = f"❌ ОШИБКА во время генерации: {e}"
        yield error_message, None