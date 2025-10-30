import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
import warnings

warnings.filterwarnings('ignore')


class DPOFineTuner:
    """DPO kullanarak model ince ayarı yapan sınıf"""

    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"DPO Fine-Tuning Başlatılıyor...")
        print(f" Cihaz: {self.device}")
        print(f" Model: {model_name}")
        print(f" Çıktı: {output_dir}")

        self.model = None
        self.model_ref = None
        self.tokenizer = None

    def setup_model_and_tokenizer(self, use_quantization: bool = True):
        """
        Model, Referans Model ve Tokenizer'ı hazırla.
        DPO için iki model gerekir:
        1. 'model' (policy): Eğitilecek olan (LoRA adaptörü eklenmiş).
        2. 'model_ref' (reference): Değişmeyen, KL-penalty için kullanılan temel model.
        """
        print("\nModel ve tokenizer yükleniyor...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        if use_quantization and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            print("   4-bit Quantization (QLoRA) etkin.")
        else:
            bnb_config = None
            if use_quantization and self.device != "cuda":
                print("   Uyarı: Quantization sadece CUDA ile desteklenir. Devre dışı bırakıldı.")

        model_dtype = torch.float16 if self.device == "cuda" else torch.float32

        print("    Referans (ref) model yükleniyor...")
        self.model_ref = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config if use_quantization else None,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=model_dtype,
            trust_remote_code=True,
        )

        print("    Ana (policy) model yükleniyor...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config if use_quantization else None,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=model_dtype,
            trust_remote_code=True,
        )

        # LoRA (PEFT) Konfigürasyonu (Sadece ana modele uygulanır)
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # TinyLlama/Gemma için
        )

        if use_quantization and self.device == "cuda":
            self.model = prepare_model_for_kbit_training(self.model)

        self.model = get_peft_model(self.model, peft_config)

        print(" Model (PEFT) hazır!")
        self.model.print_trainable_parameters()

        return self.model, self.model_ref, self.tokenizer

    def create_dpo_config(self, num_epochs: int = 1, batch_size: int = 2, learning_rate: float = 5e-5,
                          beta: float = 0.1):
        """trl.DPOConfig kullanarak eğitim ayarlarını oluşturur"""
        print("\nDPO konfigürasyonu hazırlanıyor...")

        training_args = DPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=100,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=2,

            # DPO'ya özgü parametreler
            beta=beta,  # KL katsayısı. 0.1 standart bir değerdir.
            max_prompt_length=512,  # Prompt'un max uzunluğu
            max_length=1024,  # Prompt + cevabın max uzunluğu

            optim="adamw_torch",
            gradient_checkpointing=True,
            report_to="none",
            logging_first_step=True,
            sync_ref_model=False,
        )

        print(f" Konfigürasyon hazır!")
        print(f"  • Beta (KL katsayısı): {training_args.beta}")
        print(f"  • Learning rate: {training_args.learning_rate}")
        print(f"  • Epochs: {training_args.num_train_epochs}")

        return training_args

    def train(self, dataset, training_args):
        """DPO eğitimini başlat"""
        print("\nDPO Eğitimi Başlıyor...")
        print("=" * 60)

        if self.model is None or self.model_ref is None or self.tokenizer is None:
            raise ValueError("setup_model_and_tokenizer() önce çağrılmalı.")

        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.model_ref,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.tokenizer
        )

        print("\n Eğitim devam ediyor...")
        trainer.train()
        print("\n Eğitim tamamlandı!")

        print(f"\n Model (LoRA adaptörü) kaydediliyor: {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        print(" LoRA adaptörü ve tokenizer başarıyla kaydedildi.")
        return trainer