import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import re
import sqlvalidator  # pip install sqlvalidator

# =============================
# Схема БД (подставляется автоматически)
# =============================
DATABASE_SCHEMA = """
### Database Schema:

Table: Product
Purpose: Contains information about manufacturers and product models
Columns:
  - maker (varchar(10)): Manufacturer name
  - model (varchar(50)): Unique model number (primary key)
  - type (varchar(50)): Product type - 'PC', 'Laptop', or 'Printer'

Table: PC
Purpose: Stores characteristics of desktop computers
Columns:
  - code (int): Unique identifier for each PC
  - model (varchar(50)): Foreign key referencing Product.model
  - speed (smallint): Processor speed in MHz
  - ram (smallint): RAM size in MB
  - hd (real): Hard disk size in GB
  - cd (varchar(10)): CD-ROM speed (e.g., '4x')
  - price (money): Price in dollars

Table: Laptop
Purpose: Stores characteristics of laptops
Columns:
  - code (int): Unique identifier for each laptop
  - model (varchar(50)): Foreign key referencing Product.model
  - speed (smallint): Processor speed in MHz
  - ram (smallint): RAM size in MB
  - hd (real): Hard disk size in GB
  - price (money): Price in dollars
  - screen (tinyint): Screen size in inches

Table: Printer
Purpose: Stores characteristics of printers
Columns:
  - code (int): Unique identifier for each printer
  - model (varchar(50)): Foreign key referencing Product.model
  - color (char(1)): Color capability - 'y' for color, 'n' for monochrome
  - type (varchar(10)): Printer type - 'Laser', 'Jet', or 'Matrix'
  - price (money): Price in dollars

Relationships:
  - Product.model is referenced by PC.model, Laptop.model, and Printer.model
  - Each product type has its own specific table with detailed characteristics
"""

# =============================
# Формат промпта
# =============================
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{schema}

### Response:
"""

# =============================
# Загрузка модели и токенизатора
# =============================
def load_model_and_tokenizer(model_path: str, adapter_path: str):
    print("📥 Загрузка модели и токенизатора...")

    # Конфигурация квантования
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Загрузка LoRA-адаптеров
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    print("✅ Модель успешно загружена!")
    return model, tokenizer

# =============================
# Генерация SQL
# =============================
def generate_sql(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    # Формируем промпт
    prompt = PROMPT_TEMPLATE.format(
        instruction=question.strip(),
        schema=DATABASE_SCHEMA.strip()
    )

    # Токенизация
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    # Генерация
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Декодируем только сгенерированную часть
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Очистка: удаляем лишний текст после SQL
    decoded = decoded.strip()
    if "###" in decoded:
        decoded = decoded.split("###")[0].strip()
    if "\n\n" in decoded:
        decoded = decoded.split("\n\n")[0].strip()
    if ";" not in decoded:
        decoded = decoded.rstrip() + ";"
    else:
        decoded = decoded.split(";")[0].strip() + ";"

    return decoded

# =============================
# Проверка SQL (опционально)
# =============================
def validate_sql(sql: str) -> bool:
    try:
        sql_query = sqlvalidator.parse(sql)
        return sql_query.is_valid()
    except Exception:
        return False

# =============================
# Основной интерактивный цикл
# =============================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Interactive SQL Generation with Fine-tuned Model")
    parser.add_argument("--model_path", type=str, required=True, help="Путь к оригинальной модели (например, ./sqlcoder-7b)")
    parser.add_argument("--adapter_path", type=str, required=True, help="Путь к адаптерам LoRA (например, ./sqlcoder-finetuned/final_lora)")
    args = parser.parse_args()

    # Загрузка модели
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.adapter_path)

    print("\n" + "="*80)
    print("🚀 ИНТЕРАКТИВНЫЙ ИНФЕРЕНС ЗАПУЩЕН")
    print("="*80)
    print("Введите текстовый запрос на русском или английском языке.")
    print("Для выхода введите: quit или exit")
    print("="*80 + "\n")

    while True:
        try:
            user_input = input("Ваш запрос: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                print("👋 До свидания!")
                break
            if not user_input:
                continue

            print("🧠 Генерация SQL...")
            sql = generate_sql(model, tokenizer, user_input)

            print(f"\n✅ Сгенерированный SQL:\n{sql}\n")

            # Опционально: проверка синтаксиса
            if validate_sql(sql):
                print("🟢 SQL-запрос синтаксически корректен.\n")
            else:
                print("🔴 SQL-запрос содержит синтаксические ошибки.\n")

            print("-" * 80)

        except KeyboardInterrupt:
            print("\n\n👋 Принудительный выход. До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {str(e)}\n")


if __name__ == "__main__":
    main()