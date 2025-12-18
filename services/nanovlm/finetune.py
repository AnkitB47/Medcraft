import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from .model import NanoVLM
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import json
import os
import mlflow

def finetune_nanovlm(data_path, output_dir):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please provide 'data/curated_feedback.jsonl'.")

    # Load model
    print("Loading model for QLoRA...")
    model = NanoVLM(use_qlora=True)
    
    # Prepare for k-bit training
    model.text_decoder = prepare_model_for_kbit_training(model.text_decoder)
    
    # Lora Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # TinyLlama targets
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model.text_decoder = get_peft_model(model.text_decoder, peft_config)
    model.print_trainable_parameters()
    
    # Load data
    print(f"Loading data from {data_path}...")
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    # Dataset class
    class VLMDataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer):
            self.data = data
            self.tokenizer = tokenizer
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            item = self.data[idx]
            # Format: {"image_path": "...", "prompt": "...", "answer": "..."}
            # We need to load image and tokenize text
            # For simplicity in this script, we assume pre-processed or handle loading here.
            # But wait, we need images.
            # If image_path is provided, we load it.
            # If not, we might fail or skip.
            
            # Real implementation would load image.
            # For this hardening, we must prove reality.
            # If image_path doesn't exist, we fail.
            
            image_path = item.get("image_path")
            if not os.path.exists(image_path):
                 raise FileNotFoundError(f"Image not found: {image_path}")
                 
            # Tokenize
            full_text = f"User: {item['prompt']}\nAssistant: {item['answer']}"
            enc = self.tokenizer(full_text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            
            # We need to return inputs compatible with model.forward
            # model.forward expects (images, input_ids)
            # But Trainer expects standard keys.
            # We might need a custom data collator or modify model forward to handle trainer inputs.
            # Or just return what's needed.
            
            # Since we are using a custom model, we might need a custom training loop or ensure Trainer passes correct args.
            # Standard Trainer passes input_ids, labels, attention_mask.
            # Our model needs 'images'.
            
            # Let's assume we load image and return it.
            # Image loading logic (using PIL)
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            # Process image
            # We need the processor from the model, but it's inside the model.
            # We should probably pass processor to dataset.
            
            # For this script, let's just use the tokenizer and assume images are handled if we had the processor.
            # But we need to be real.
            # We can instantiate processor here.
            from transformers import CLIPImageProcessor
            processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
            
            return {
                "input_ids": enc.input_ids.squeeze(0),
                "attention_mask": enc.attention_mask.squeeze(0),
                "labels": enc.input_ids.squeeze(0), # Causal LM training
                "images": pixel_values
            }
            
    train_dataset = VLMDataset(data, model.tokenizer)
    
    # MLflow Setup
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("nanovlm_finetune")
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100,
        fp16=True,
        logging_dir="./logs",
        report_to=["mlflow"],
        remove_unused_columns=False # Important for custom models
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    finetune_nanovlm("data/curated_feedback.jsonl", "models/nanovlm_finetuned")
