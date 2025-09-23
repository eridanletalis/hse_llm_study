import os
import json
import logging
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch
import psutil
import GPUtil
import matplotlib.pyplot as plt

# =============================
# Настройка логирования
# =============================
def setup_logging(log_dir: str = "./logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# =============================
# Загрузка датасета
# =============================
def load_dataset_from_file(file_path: str) -> Dataset:
    logger.info(f"Загрузка датасета из {file_path}...")
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        data = df.to_dict('records')
    else:
        raise ValueError("Поддерживаются только .json, .jsonl, .csv")

    for item in data:
        if "instruction" not in item or "output" not in item:
            raise ValueError("Каждая запись должна содержать 'instruction' и 'output'")
        if "input" not in item:
            item["input"] = ""

    def format_instruction(example):
        if example['input']:
            return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
        else:
            return f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""

    texts = [format_instruction(item) for item in data]
    logger.info(f"Успешно загружено {len(texts)} примеров.")
    return Dataset.from_dict({"text": texts})


# =============================
# Логирование использования GPU
# =============================
def log_gpu_memory():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        logger.info(f"GPU {gpu.id}: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB (Free: {gpu.memoryFree}MB)")


# =============================
# Основной скрипт
# =============================
def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with QLoRA")
    parser.add_argument("--model_path", type=str, required=True, help="Путь к локальной модели")
    parser.add_argument("--data_path", type=str, required=True, help="Путь к файлу с обучающими данными")
    parser.add_argument("--output_dir", type=str, default="./results", help="Каталог для сохранения результатов")
    parser.add_argument("--epochs", type=int, default=3, help="Количество эпох")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (на устройство)")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Градиентные накопления")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Максимальная длина последовательности")
    parser.add_argument("--val_split", type=float, default=0.1, help="Доля валидационной выборки (0.1 = 10%)")
    args = parser.parse_args()

    # Инициализация логгера
    global logger
    logger = setup_logging(os.path.join(args.output_dir, "logs"))

    logger.info("="*60)
    logger.info("🚀 НАЧАЛО ОБУЧЕНИЯ")
    logger.info("="*60)
    logger.info(f"Модель: {args.model_path}")
    logger.info(f"Данные: {args.data_path}")
    logger.info(f"Выходная директория: {args.output_dir}")
    logger.info(f"Эпохи: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation steps: {args.grad_accum_steps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Max sequence length: {args.max_seq_length}")
    logger.info(f"Validation split: {args.val_split}")

    # Логирование памяти до загрузки модели
    logger.info("📊 Память до загрузки модели:")
    log_gpu_memory()

    # Загрузка датасета
    full_dataset = load_dataset_from_file(args.data_path)

    # Разделение на train / val
    if args.val_split > 0:
        dataset_dict = full_dataset.train_test_split(test_size=args.val_split, seed=42)
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]
        logger.info(f"Разделение: {len(train_dataset)} train, {len(eval_dataset)} val")
    else:
        train_dataset = full_dataset
        eval_dataset = None
        logger.info("Валидационная выборка не используется.")

    # Конфигурация квантования
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Загрузка модели и токенизатора
    logger.info("📥 Загрузка модели и токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("🔧 Подготовка модели для k-bit обучения...")
    model = prepare_model_for_kbit_training(model)

    # LoRA конфиг
    peft_config = LoraConfig(
        r=256,  # было 64 и 128, поменяли для улучшения обучения сложным зависимостям
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )


    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    logger.info("📊 Память после загрузки модели:")
    log_gpu_memory()

    # Токенизация
    logger.info("🔤 Токенизация данных...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_length,
            return_tensors="pt"
        )

    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    if eval_dataset:
        tokenized_eval = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
    else:
        tokenized_eval = None

    # Training Arguments — БЕЗ wandb
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=50 if eval_dataset else None,
        logging_steps=10,
        learning_rate=args.learning_rate,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",  # Отключаем wandb и другие интеграции
        save_total_limit=2,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        push_to_hub=False,
    )

    # Создание тренера — БЕЗ wandb
    logger.info("🏋️‍♂️ Создание тренера...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        peft_config=peft_config
    )

    # Списки для хранения метрик
    train_loss_history = []
    eval_loss_history = []
    steps = []

    # Callback для сохранения метрик
    class LossHistoryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                if "loss" in logs:
                    train_loss_history.append(logs["loss"])
                    steps.append(state.global_step)
                if "eval_loss" in logs:
                    eval_loss_history.append(logs["eval_loss"])

    trainer.add_callback(LossHistoryCallback())

    # Обучение
    logger.info("🔥 Начало обучения...")
    train_result = trainer.train()

    # Логирование результатов
    logger.info("📈 Результаты обучения:")
    logger.info(f"  Общее количество шагов: {train_result.global_step}")
    logger.info(f"  Последняя train loss: {train_result.training_loss}")

    if eval_dataset:
        eval_result = trainer.evaluate()
        logger.info(f"  Валидационная loss: {eval_result['eval_loss']}")
        # Сохраняем валидационные метрики в файл
        with open(os.path.join(args.output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=2)

    # Сохранение
    logger.info("💾 Сохранение адаптеров...")
    final_lora_path = os.path.join(args.output_dir, "final_lora")
    model.save_pretrained(final_lora_path)
    tokenizer.save_pretrained(final_lora_path)

    logger.info("✅ Обучение успешно завершено!")
    logger.info(f"Адаптеры сохранены в: {final_lora_path}")

    logger.info("📊 Память после обучения:")
    log_gpu_memory()

    # Сохранение истории лоссов
    history = {
        "steps": steps,
        "train_loss": train_loss_history,
        "eval_loss": eval_loss_history
    }
    with open(os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # Построение графиков (опционально)
    if len(train_loss_history) > 0:
        os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(steps, train_loss_history, label="Train Loss", marker='o')
        if len(eval_loss_history) > 0:
            eval_steps = steps[::len(steps)//len(eval_loss_history)][:len(eval_loss_history)]  # приближённое выравнивание
            plt.plot(eval_steps, eval_loss_history, label="Eval Loss", marker='s')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "plots", "loss_curve.png"))
        plt.close()
        logger.info("📈 График loss сохранён в plots/loss_curve.png")

    logger.info("="*60)
    logger.info("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО")
    logger.info("="*60)


# Добавляем TrainerCallback для сбора метрик
from transformers import TrainerCallback

if __name__ == "__main__":
    main()