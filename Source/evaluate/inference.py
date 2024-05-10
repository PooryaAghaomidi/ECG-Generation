import torch
from PIL import Image
from utils import model_loader
from evaluate import generation
from transformers import CLIPTokenizer


class StableDiffusion:
    def __init__(self,
                 ALLOW_CUDA=False,
                 ALLOW_MPS=False,
                 tokenizier_path="checkpoints/vocab.json",
                 merg_path="checkpoints/merges.txt",
                 model_file="checkpoints/v1-5-pruned.ckpt",
                 sampler="ddpm",
                 num_inference_steps=50,
                 seed=42,
                 uncond_prompt="",
                 do_cfg=True,
                 cfg_scale=8,
                 input_image=None,
                 image_path="",
                 strength=0.9):
        self.seed = seed
        self.sampler = sampler
        self.uncond_prompt = uncond_prompt
        self.do_cfg = do_cfg
        self.cfg_scale = cfg_scale
        self.input_image = input_image
        self.image_path = image_path
        self.strength = strength
        self.num_inference_steps = num_inference_steps

        if torch.cuda.is_available() and ALLOW_CUDA:
            self.device = "cuda"
        elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
            self.device = "mps"
        print(f"Using device: {self.device}")

        self.tokenizer = CLIPTokenizer(tokenizier_path, merges_file=merg_path)
        self.models = model_loader.preload_models_from_standard_weights(model_file, self.device)

    def generate_image(self, prompt):
        output_image = generation.generate(prompt=prompt,
                                           uncond_prompt=self.uncond_prompt,
                                           input_image=self.input_image,
                                           strength=self.strength,
                                           do_cfg=self.do_cfg,
                                           cfg_scale=self.cfg_scale,
                                           sampler_name=self.sampler,
                                           n_inference_steps=self.num_inference_steps,
                                           seed=self.seed,
                                           models=self.models,
                                           device=self.device,
                                           idle_device="cuda",
                                           tokenizer=self.tokenizer)

        return Image.fromarray(output_image)
