# file: app.py (финальная версия с продвинутой генерацией)

import gradio as gr
import os
from datetime import datetime

# Импортируем все наши менеджеры
from data_preparer import prepare_dataset
from training_manager import tokenize_dataset, run_training
# ВАЖНО: импортируем обновленный inference_manager
from inference_manager import get_available_models, get_available_lora_models, run_inference

# --- UI-блок для генерации ---
def create_generation_tab():
    with gr.Blocks() as generation_block:
        gr.Markdown("## 🎵 Генерация Музыки (PRO)")
        
        with gr.Row():
            base_model_dropdown = gr.Dropdown(label="1. Базовая модель", choices=get_available_models(), scale=3)
            lora_model_dropdown = gr.Dropdown(label="2. LoRA модель (опционально)", choices=get_available_lora_models(), scale=3)
            is_exllamav2_checkbox = gr.Checkbox(label="Exllamav2 модель?", value=False, scale=1)
            refresh_models_button = gr.Button("🔄 Обновить модели")

        gr.Markdown("---")
        gr.Markdown("### ⚙️ Настройки промпта")
        prompt_type_radio = gr.Radio(
            ["Текстовый промпт", "Одиночный аудио-промпт", "Двойной аудио-промпт"],
            label="Тип промпта",
            value="Текстовый промпт"
        )
        
        # --- Блок для текстового промпта (виден по умолчанию) ---
        with gr.Blocks(visible=True) as text_prompt_group:
            prompt_input = gr.Textbox(label="Текст", lines=3, placeholder="epic cinematic orchestral music...")

        # --- Блок для ОДИНОЧНОГО аудио-промпта (скрыт) ---
        with gr.Blocks(visible=False) as single_audio_group:
            gr.Markdown("Загрузите один аудиофайл и укажите его тип. Второй трек будет заменен тишиной.")
            single_audio_upload = gr.Audio(label="Загрузите аудио-промпт", type="filepath")
            single_audio_type_radio = gr.Radio(["Вокал", "Инструментал"], label="Тип загруженного аудио", value="Инструментал")

        # --- Блок для ДВОЙНОГО аудио-промпта (скрыт) ---
        with gr.Blocks(visible=False) as dual_audio_group:
            gr.Markdown("Загрузите отдельно дорожку с вокалом и инструменталом.")
            instrumental_upload = gr.Audio(label="Загрузите инструментал", type="filepath")
            vocal_upload = gr.Audio(label="Загрузите вокал", type="filepath")
        
        # --- Общие настройки для аудио-промптов (скрыты) ---
        with gr.Blocks(visible=False) as audio_timing_group:
            with gr.Row():
                prompt_start_time_input = gr.Number(label="Начало промпта (сек)", value=0)
                prompt_end_time_input = gr.Number(label="Конец промпта (сек)", value=10)

        # --- Расширенные настройки генерации ---
        with gr.Accordion("Дополнительные настройки генерации и кэша", open=False):
            with gr.Row():
                duration_slider = gr.Slider(minimum=5, maximum=300, value=30, step=1, label="Финальная длительность (сек)")
                max_new_tokens_slider = gr.Slider(minimum=64, maximum=4096, value=1024, step=64, label="Max New Tokens")
            gr.Markdown("##### Параметры кэша")
            with gr.Row():
                stage1_cache_size_input = gr.Number(label="Кэш Stage 1 (GB)", value=2)
                stage2_cache_size_input = gr.Number(label="Кэш Stage 2 (GB)", value=2)
            gr.Markdown("##### Параметры Stage 1")
            with gr.Row():
                s1_guidance_scale_slider = gr.Slider(1, 15, 3.5, step=0.1, label="Guidance Scale")
                s1_top_p_slider = gr.Slider(0.1, 1.0, 0.95, step=0.01, label="Top-P")
                s1_temp_slider = gr.Slider(0.1, 1.5, 1.0, step=0.05, label="Temperature")
                s1_rep_penalty_slider = gr.Slider(1.0, 1.5, 1.2, step=0.01, label="Repetition Penalty")

        # --- Кнопка запуска и вывод ---
        gr.Markdown("---")
        generate_button = gr.Button("🎹 Сгенерировать музыку!", variant="primary", size="lg")
        status_output = gr.Textbox(label="Статус", interactive=False)
        audio_output = gr.Audio(label="Ваш трек", type="filepath")

        # --- Динамический UI ---
        def switch_prompt_ui(prompt_type):
            return {
                text_prompt_group: gr.update(visible=prompt_type == "Текстовый промпт"),
                single_audio_group: gr.update(visible=prompt_type == "Одиночный аудио-промпт"),
                dual_audio_group: gr.update(visible=prompt_type == "Двойной аудио-промпт"),
                audio_timing_group: gr.update(visible=prompt_type in ["Одиночный аудио-промпт", "Двойной аудио-промпт"])
            }
        prompt_type_radio.change(fn=switch_prompt_ui, inputs=prompt_type_radio, outputs=[text_prompt_group, single_audio_group, dual_audio_group, audio_timing_group])
        
        # --- Связывание с бэкендом ---
        refresh_models_button.click(
            fn=lambda: (gr.update(choices=get_available_models()), gr.update(choices=get_available_lora_models())),
            outputs=[base_model_dropdown, lora_model_dropdown]
        )
        generate_button.click(
            fn=run_inference,
            inputs=[
                base_model_dropdown, lora_model_dropdown, is_exllamav2_checkbox,
                stage1_cache_size_input, stage2_cache_size_input,
                prompt_type_radio, prompt_input, single_audio_upload, single_audio_type_radio,
                instrumental_upload, vocal_upload,
                prompt_start_time_input, prompt_end_time_input,
                duration_slider, max_new_tokens_slider,
                s1_guidance_scale_slider, s1_top_p_slider, s1_temp_slider, s1_rep_penalty_slider
            ],
            outputs=[status_output, audio_output]
        )
    return generation_block

# ... (Код для create_finetune_tab() и запуска demo остается без изменений) ...

with gr.Blocks(title="YUE Music AI Studio PRO", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎹 YUE Music AI Studio (PRO Version)")
    # ...
    with gr.Tabs():
        with gr.TabItem("🎵 Генерация Музыки"):
            create_generation_tab()
        with gr.TabItem("⚙️ Дообучение Модели (Fine-tuning)"):
            create_finetune_tab() # Здесь используется функция из предыдущих шагов

if __name__ == "__main__":
    # ...
    demo.launch(debug=True, inbrowser=True)