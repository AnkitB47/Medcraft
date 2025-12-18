import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor, AutoConfig

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
            # MultiheadAttention expects [batch, seq, dim] if batch_first=True
            
            # Q = latents, K = x, V = x
            # We want to attend to x from latents
            # attn_out = Attention(Q, K, V)
            
            # Concatenate latents and x for self-attention? 
            # Original Perceiver: Cross-attend to inputs, then self-attend on latents.
            # Simplified here: Just cross-attention + FF
            
            # 1. Cross Attention (Latents attend to Image Features)
            # In nn.MultiheadAttention: forward(query, key, value)
            attn_out, _ = attn(latents, x, x) 
            latents = latents + norm1(attn_out)
            
            # 2. Feed Forward
            latents = latents + norm2(ff(latents))
            
        return latents

class NanoVLM(nn.Module):
    def __init__(self, text_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", vision_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        
        # 1. Vision Encoder (CLIP ViT)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
        
        # Freeze vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        vision_dim = self.vision_encoder.config.hidden_size
        
        # 2. Perceiver Resampler (The "Nano" part)
        # Compresses variable number of image patches into fixed number of latents
        self.resampler = PerceiverResampler(dim=vision_dim)
        
        # 3. Text Decoder (TinyLlama)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_decoder = AutoModelForCausalLM.from_pretrained(text_model_name)
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
        text_embeds = self.text_decoder.get_input_embeddings()(input_ids) # [b, seq_len, text_dim]
        
        # 5. Concatenate
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        # 6. Generate
        outputs = self.text_decoder(inputs_embeds=combined_embeds)
        return outputs

    @torch.no_grad()
    def generate(self, image, prompt, max_new_tokens=50):
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
        outputs = self.text_decoder.generate(
            input_ids=input_ids, 
            max_new_tokens=max_new_tokens
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
