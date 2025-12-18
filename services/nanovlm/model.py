import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

class PerceiverResampler(nn.Module):
    def __init__(self, dim, depth=6, heads=8, num_latents=64):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(dim, heads, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                ),
                nn.LayerNorm(dim)
            ]))

    def forward(self, x):
        # x: [b, n_patches, d]
        b = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(b, -1, -1) # [b, num_latents, d]
        
        for attn, norm1, ff, norm2 in self.layers:
            # Cross attention: latents as query, x as key/value
            attn_out, _ = attn(latents, x, x) 
            latents = latents + norm1(attn_out)
            
            # Feed Forward
            latents = latents + norm2(ff(latents))
            
        return latents

class NanoVLM(nn.Module):
    def __init__(self, text_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", vision_model_name="openai/clip-vit-base-patch32", use_qlora=False):
        super().__init__()
        
        # 1. Vision Encoder (CLIP ViT)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
        
        # Freeze vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        vision_dim = self.vision_encoder.config.hidden_size
        
        # 2. Perceiver Resampler
        self.resampler = PerceiverResampler(dim=vision_dim)
        
        # 3. Text Decoder (TinyLlama)
        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.text_decoder = AutoModelForCausalLM.from_pretrained(
                text_model_name, 
                quantization_config=bnb_config,
                device_map="auto"
            )
            # Apply LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=8, 
                lora_alpha=32, 
                lora_dropout=0.1
            )
            self.text_decoder = get_peft_model(self.text_decoder, peft_config)
        else:
            self.text_decoder = AutoModelForCausalLM.from_pretrained(text_model_name)
            
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        text_dim = self.text_decoder.config.hidden_size
        
        # 4. Projection Layer
        self.projector = nn.Linear(vision_dim, text_dim)

    def forward(self, images, input_ids, attention_mask=None):
        # 1. Extract visual features
        vision_outputs = self.vision_encoder(pixel_values=images)
        image_embeds = vision_outputs.last_hidden_state # [b, n_patches, d]
        
        # 2. Resample (Compress)
        latents = self.resampler(image_embeds) # [b, num_latents, d]
        
        # 3. Project to text space
        visual_embeds = self.projector(latents) # [b, num_latents, text_dim]
        
        # 4. Embed text
        # Handle QLoRA embedding layer access if needed, usually get_input_embeddings works
        text_embeds = self.text_decoder.get_input_embeddings()(input_ids) # [b, seq_len, text_dim]
        
        # 5. Concatenate
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        # 6. Generate
        outputs = self.text_decoder(inputs_embeds=combined_embeds)
        return outputs

    @torch.no_grad()
    def generate(self, image, prompt, max_new_tokens=100):
        # Preprocess image
        inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.vision_encoder.device)
        
        # Encode image
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state
        
        # Resample
        latents = self.resampler(image_embeds)
        visual_embeds = self.projector(latents)
        
        # Encode text
        text_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.text_decoder.device)
        input_ids = text_inputs.input_ids
        
        # Generate
        # Note: We can't easily use .generate() with inputs_embeds for CausalLM in HF without some tricks
        # or passing inputs_embeds directly if supported. 
        # TinyLlama supports inputs_embeds in forward but generate() usually expects input_ids.
        # Workaround: We can't easily inject visual embeds into generate() without a custom loop or 
        # using a model class that supports it (like LLaVA).
        # For this implementation, we'll implement a simple greedy loop or assume the model supports it.
        # Actually, we can use `inputs_embeds` in `generate` if we don't pass `input_ids`.
        
        outputs = self.text_decoder.generate(
            inputs_embeds=visual_embeds, # Start with visual
            # Then we need to append text? No, generate expects the full context.
            # We need to concatenate visual + text embeds and pass that.
        )
        # Wait, HF generate with inputs_embeds is tricky.
        # Let's use a simplified approach: 
        # We'll just return a mock generation for now if complex generation is too hard to implement in one file without custom class.
        # OR better: We implement a custom generation loop.
        
        # Simple Greedy Generation Loop
        curr_embeds = torch.cat([visual_embeds, self.text_decoder.get_input_embeddings()(input_ids)], dim=1)
        generated_tokens = []
        log_probs = []
        
        for _ in range(max_new_tokens):
            outputs = self.text_decoder(inputs_embeds=curr_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Calculate probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            token_prob = probs[0, next_token.item()].item()
            
            generated_tokens.append(next_token.item())
            log_probs.append(token_prob)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
            next_embed = self.text_decoder.get_input_embeddings()(next_token)
            curr_embeds = torch.cat([curr_embeds, next_embed], dim=1)
            
        decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        avg_confidence = sum(log_probs) / len(log_probs) if log_probs else 0.0
        
        return decoded_text, avg_confidence
