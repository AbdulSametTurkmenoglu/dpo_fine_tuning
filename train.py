import argparse
import torch
from src.fine_tuner import DPOFineTuner
from src.data_utils import load_dataset_with_preferences



def run_comparison(trainer, tokenizer, test_prompt: str):
    """
    Eğitim sonrası, referans model ile eğitilmiş modeli karşılaştırır.
    'trainer' objesinden 'model' (eğitilmiş) ve 'ref_model' (orijinal) alır.
    """
    print("\n" + "=" * 60)
    print(" EĞITIM ÖNCESİ vs SONRASI KARŞILAŞTIRMA")
    print("=" * 60)

    ref_model = trainer.ref_model
    policy_model = trainer.model

    ref_model.eval()
    policy_model.eval()

    device = next(policy_model.parameters()).device

    print(f"\nTest Prompt: {test_prompt}\n")
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

    gen_kwargs = {
        "max_new_tokens": 150,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    print(" EĞITIM ÖNCESİ (Reference Model):")
    print("-" * 60)
    with torch.no_grad():
        outputs_before = ref_model.generate(**inputs, **gen_kwargs)

    response_before = tokenizer.decode(outputs_before[0], skip_special_tokens=True)
    response_before = response_before[len(test_prompt):].strip()  # Sadece cevabı al
    print(response_before if response_before else "[Model cevap üretemedi]")

    print("\n EĞITIM SONRASI (DPO Fine-tuned Model):")
    print("-" * 60)
    with torch.no_grad():
        outputs_after = policy_model.generate(**inputs, **gen_kwargs)

    response_after = tokenizer.decode(outputs_after[0], skip_special_tokens=True)
    response_after = response_after[len(test_prompt):].strip()  # Sadece cevabı al
    print(response_after if response_after else "[Model cevap üretemedi]")
    print("\n" + "=" * 60)



def main():
    parser = argparse.ArgumentParser(description="DPO Fine-Tuning Script")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Hugging Face model adı")
    parser.add_argument("--dataset_name", type=str, default="argilla/ultrafeedback-binarized-preferences-cleaned",
                        help="DPO veri seti adı")
    parser.add_argument("--output_dir", type=str, default="dpo_tinyllama_model",
                        help="Eğitilmiş modelin kaydedileceği dizin")
    parser.add_argument("--num_samples", type=int, default=500, help="Veri setinden kullanılacak örnek sayısı")
    parser.add_argument("--epochs", type=int, default=1, help="Eğitim epoch sayısı")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch boyutu (DPO için düşük olmalı)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Öğrenme oranı")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty) katsayısı")
    parser.add_argument("--no_quantization", action="store_true",
                        help="4-bit quantization'ı devre dışı bırak (Örn: CPU veya MPS için)")

    args = parser.parse_args()

    print("=" * 60)
    print(" DPO (Direct Preference Optimization) Başlatılıyor")
    print("=" * 60)

    fine_tuner = DPOFineTuner(
        model_name=args.model_name,
        output_dir=args.output_dir
    )

    dataset = load_dataset_with_preferences(args.dataset_name, num_samples=args.num_samples)

    use_quant = (fine_tuner.device == 'cuda') and not args.no_quantization
    fine_tuner.setup_model_and_tokenizer(use_quantization=use_quant)

    training_args = fine_tuner.create_dpo_config(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta
    )

    test_prompt = "How can I become more productive?"
    print(f"\nKarşılaştırma için test prompt'u: {test_prompt}")

    trainer = fine_tuner.train(dataset, training_args)

    print("\nEğitim tamamlandı. Modeller karşılaştırılıyor...")
    run_comparison(trainer, fine_tuner.tokenizer, test_prompt)

    print("\n" + "=" * 60)
    print(" DPO Fine-Tuning Tamamlandı!")
    print("=" * 60)
    print(f"\n Model kaydedildi: {fine_tuner.output_dir}")


if __name__ == "__main__":
    main()