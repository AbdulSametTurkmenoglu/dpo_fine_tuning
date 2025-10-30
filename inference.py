import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


def main():
    parser = argparse.ArgumentParser(description="Inference script for DPO-tuned model")
    parser.add_argument("--adapter_path", type=str, default="dpo_tinyllama_model",
                        help="Path to the saved fine-tuned model (LoRA adapter)")
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Base model name used for tuning")
    args = parser.parse_args()

    print(f"Temel model yükleniyor: {args.base_model}...")

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Cihaz kullanılıyor: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto"
    )

    print(f"LoRA adaptörü yükleniyor: {args.adapter_path}...")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    # Not: DPO eğitimi sonrası 'merge_and_unload()' bazen
    # 'ref_model'den kalan bağlantılar nedeniyle sorun çıkarabilir.
    # Genellikle inference için en güvenli yol, adaptörü bu şekilde yüklemektir.
    # Hız için birleştirmek isterseniz: model = model.merge_and_unload()
    model.eval()

    print("Model başarıyla yüklendi. İnteraktif test (çıkmak için 'exit' yazın).")
    print("=" * 60)

    while True:
        try:
            prompt = input(" Prompt: ")
            if prompt.lower() == 'exit':
                break

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=250,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()  # Sadece cevabı al

            print("-" * 60)
            print(f" Response:\n{response}")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nÇıkılıyor...")
            break


if __name__ == "__main__":
    main()