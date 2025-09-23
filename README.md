Запуск дообучения
python fine-tuning.py   --model_path ./sqlcoder-7b   --data_path ./train_data_english.jsonl   --output_dir ./sqlcoder-finetuned   --epochs 8   --batch_size 4   --grad_accum_steps 4   --learning_rate 1e-4   --max_seq_length 2048   --val_split 0.1


Интерактивный инференс
 python interactive_inference.py   --model_path ./sqlcoder-7b   --adapter_path ./sqlcoder-finetuned/final_lora


Автоматический инференс для валидации
python inference_validation.py   --model_path ./sqlcoder-7b   --adapter_path ./sqlcoder-finetuned/final_lora   --data_path ./augmented_train_data_v2.jsonl   --val_split 0.1   --max_new_tokens 512   --output_file inference_results.jsonl
