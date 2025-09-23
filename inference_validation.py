import os
import json
import argparse
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import torch
import sqlvalidator  # pip install sqlvalidator

# =============================
# Загрузка и разделение датасета (воспроизводимо)
# =============================
def load_and_split_dataset(file_path: str, val_split: float = 0.1):
    """Загружает датасет и делит его на train/val с тем же random_state=42"""
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

    # Форматирование в промпт (как в обучении)
    def format_instruction(example):
        if example['input']:
            return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
"""
        else:
            return f"""### Instruction:
{example['instruction']}

### Response:
"""

    texts = [format_instruction(item) for item in data]
    dataset = Dataset.from_dict({"text": texts, "output": [item["output"] for item in data]})

    # Разделение с тем же random_state=42
    dataset = dataset.train_test_split(test_size=val_split, seed=42)
    return dataset["test"]  # возвращаем валидационную выборку


# =============================
# Функция генерации SQL
# =============================
def generate_sql(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Извлекаем только сгенерированную часть
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Очистка: удаляем лишние переносы, обрезаем по точке с запятой
    decoded = decoded.strip()
    if "###" in decoded:
        decoded = decoded.split("###")[0].strip()
    if "\n\n" in decoded:
        decoded = decoded.split("\n\n")[0].strip()
    if ";" in decoded:
        decoded = decoded.split(";")[0].strip() + ";"
    else:
        decoded = decoded.rstrip() + ";"

    return decoded


# =============================
# Проверка SQL на корректность
# =============================
def is_sql_valid(sql: str) -> bool:
    """Проверяет, является ли SQL-запрос синтаксически корректным"""
    try:
        # Удаляем лишние символы в начале/конце
        sql = sql.strip().rstrip(";") + ";"
        sql_query = sqlvalidator.parse(sql)
        return sql_query.is_valid()
    except Exception:
        return False


# =============================
# Основной скрипт
# =============================
def main():
    parser = argparse.ArgumentParser(description="Inference on validation set")
    parser.add_argument("--model_path", type=str, required=True, help="Путь к оригинальной модели")
    parser.add_argument("--adapter_path", type=str, required=True, help="Путь к адаптерам LoRA")
    parser.add_argument("--data_path", type=str, required=True, help="Путь к файлу с данными")
    parser.add_argument("--val_split", type=float, default=0.1, help="Доля валидационной выборки")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Максимальное количество генерируемых токенов")
    parser.add_argument("--output_file", type=str, default="inference_results.jsonl", help="Файл для сохранения результатов")
    args = parser.parse_args()

    print("🔍 Загрузка валидационной выборки...")
    val_dataset = load_and_split_dataset(args.data_path, args.val_split)
    print(f"✅ Загружено {len(val_dataset)} примеров для валидации")

    # =============================
    # Загрузка оригинальной модели
    # =============================
    print("📥 Загрузка оригинальной модели...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    base_model.eval()

    # =============================
    # Загрузка дообученной модели (с адаптерами)
    # =============================
    print("📥 Загрузка дообученной модели с адаптерами...")
    fine_tuned_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    fine_tuned_model.eval()

    # =============================
    # Инференс и оценка
    # =============================
    results = []
    base_correct = 0
    ft_correct = 0

    print("🚀 Начало инференса...")
    for i, example in enumerate(val_dataset):
        prompt = example["text"]
        true_sql = example["output"]

        # Инференс на оригинальной модели
        base_sql = generate_sql(base_model, tokenizer, prompt, args.max_new_tokens)

        # Инференс на дообученной модели
        ft_sql = generate_sql(fine_tuned_model, tokenizer, prompt, args.max_new_tokens)

        # Проверка корректности
        base_valid = is_sql_valid(base_sql)
        ft_valid = is_sql_valid(ft_sql)

        if base_valid:
            base_correct += 1
        if ft_valid:
            ft_correct += 1

        # Сохраняем результат
        result = {
            "id": i + 1,
            "prompt": prompt,
            "true_sql": true_sql,
            "base_model_sql": base_sql,
            "base_model_valid": base_valid,
            "fine_tuned_sql": ft_sql,
            "fine_tuned_valid": ft_valid
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"✅ Обработано {i + 1} / {len(val_dataset)} примеров")

    # =============================
    # Сохранение результатов
    # =============================
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # =============================
    # Вывод метрик
    # =============================
    total = len(val_dataset)
    base_accuracy = base_correct / total
    ft_accuracy = ft_correct / total

    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ ИНФЕРЕНСА")
    print("="*60)
    print(f"Всего примеров: {total}")
    print(f"Оригинальная модель — корректных SQL: {base_correct} ({base_accuracy:.2%})")
    print(f"Дообученная модель — корректных SQL: {ft_correct} ({ft_accuracy:.2%})")
    print(f"Улучшение: {ft_accuracy - base_accuracy:+.2%}")
    print("="*60)

    if ft_accuracy > base_accuracy:
        print("🎉 Дообучение улучшило качество генерации SQL!")
    elif ft_accuracy == base_accuracy:
        print("➡️ Качество осталось на том же уровне.")
    else:
        print("⚠️ Качество ухудшилось — проверьте гиперпараметры обучения.")

    print(f"\n💾 Результаты сохранены в: {args.output_file}")


if __name__ == "__main__":
    main()