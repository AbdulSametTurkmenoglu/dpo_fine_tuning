#  DPO (Direct Preference Optimization) Projesi

Bu proje, Hugging Face `trl` (Transformer Reinforcement Learning), `peft` (QLoRA) ve `transformers` kütüphanelerini kullanarak bir dil modeline (örn: TinyLlama) DPO ile ince ayar yapar.

DPO, RLHF'e (Reinforcement Learning from Human Feedback) kıyasla daha basit ve stabil bir alternatiftir. Bir ödül modeli (reward model) eğitmek yerine, "tercih edilen" (chosen) ve "reddedilen" (rejected) yanıt çiftlerini kullanarak modeli doğrudan optimize eder.



##  Özellikler

* **DPO:** `trl` kütüphanesinin `DPOTrainer` sınıfı ile verimli eğitim.
* **QLoRA:** 4-bit kuantizasyon ile (CUDA GPU varsa) bellek-verimli eğitim.
* **Modüler Yapı:** Veri işleme, model sınıfı ve eğitim script'i birbirinden ayrılmıştır.
* **Referans Model Yönetimi:** DPO için gerekli olan 'policy' (eğitilen) ve 'reference' (sabit) modelleri otomatik olarak yönetir.
* **CLI Desteği:** `argparse` ile model, veri seti, epoch gibi parametreleri komut satırından kolayca değiştirme.



##  Kurulum

1.  **Depoyu Klonlama:**
    ```bash
    git clone [https://github.com/kullanici-adiniz/dpo.git](https://github.com/kullanici-adiniz/dpo.git)
    cd dpo
    ```

2.  **Sanal Ortam (Önerilir):**
    ```bash
    python -m venv .venv
    # Windows: .\.venv\Scripts\activate
    # macOS/Linux: source .venv/bin/activate
    ```

3.  **Gerekli Kütüphaneleri Yükleme:**
    ```bash
    pip install -r requirements.txt
    ```

## ⚡ Kullanım

### 1. Modeli Eğitme

Ana eğitim script'i `train.py`'dir.

**Varsayılan ayarlarla (TinyLlama + UltraFeedback verisi + QLoRA) eğitimi başlat:**

```bash
python train.py
```

**Örnek 2: Daha fazla veri ve farklı beta değeri ile eğitim:**

```bash
python train.py --num_samples 1000 --epochs 2 --beta 0.15 --output_dir dpo_model_v2
```

**Örnek 3: Quantization olmadan (CPU veya MPS üzerinde):**

```bash
python train.py --no_quantization
```

Tüm argümanları görmek için:
```bash
python train.py --help
```

### 2. Eğitilmiş Model ile Çıkarım (Inference)

Eğitim tamamlandığında (`dpo_tinyllama_model` klasörü oluştuğunda), `inference.py` script'ini kullanarak modelle interaktif olarak sohbet edebilirsiniz.

Bu script, temel modeli yükler ve üzerine eğitilmiş LoRA adaptörünü uygular.

```bash
python inference.py
```

Eğer farklı bir çıktı klasörü kullandıysanız:
```bash
python inference.py --base_model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --adapter_path "dpo_model_v2"
```