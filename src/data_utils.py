from datasets import load_dataset


def load_dataset_with_preferences(dataset_name: str, num_samples: int = 1000):
    """
    Tercih çiftleri içeren veri setini yükle
    DPO için: prompt, chosen (seçilen), rejected (reddedilen) formatında
    """
    print(f"\nVeri seti yükleniyor: {dataset_name}...")

    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    def format_for_dpo(example):
        """
        UltraFeedback veri setini 'prompt', 'chosen', 'rejected' formatına getirir.
        Not: Bu veri seti, 'chosen' ve 'rejected' listelerindeki [1] indeksinde
        asıl model yanıtlarını tutar ([0] indeksinde kullanıcı prompt'u vardır).
        """
        try:
            prompt = example["prompt"]
            chosen = example["chosen"][1]["content"]
            rejected = example["rejected"][1]["content"]

            if not chosen or not rejected:
                return None

            return {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            }
        except (IndexError, TypeError, KeyError):
            return None

    original_count = len(dataset)
    dataset = dataset.map(format_for_dpo)

    dataset = dataset.filter(lambda x: x is not None and x['prompt'] is not None)
    filtered_count = len(dataset)
    print(f"  {original_count - filtered_count} hatalı/boş örnek filtrelendi.")

    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    print(f" {len(dataset['train'])} eğitim, {len(dataset['test'])} test örneği yüklendi")

    if len(dataset['train']) > 0:
        print(f"\nÖrnek DPO verisi:")
        print(f"  Prompt: {dataset['train'][0]['prompt'][:100]}...")
        print(f"  Seçilen (Chosen): {dataset['train'][0]['chosen'][:100]}...")
        print(f"  Reddedilen (Rejected): {dataset['train'][0]['rejected'][:100]}...")
    else:
        print("Uyarı: Yüklenecek geçerli veri bulunamadı.")

    return dataset