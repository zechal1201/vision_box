import os
import time
import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline, StableDiffusion3Pipeline
from torch.nn import Module


class UnifiedDiffusionPipeline:
    def __init__(self, model_dir, device, image_height, image_width,
                 scheduler=None, seed=None, guidance_scale=7.5,
                 denoising_steps=20, num_images_per_prompt=1,
                 output_dir="output", lora=None, adapter_weights=None,
                 lora_scale=1.0, merge_lora=False, model_type="sdxl"):
        """
        统一的 Diffusion Pipeline 类，支持 SDXL 和 SD3。

        参数:
            model_type (str): 模型类型，"sdxl" 或 "sd3"
        """
        self.model_dir = model_dir
        self.device = device
        self.image_height = image_height
        self.image_width = image_width
        self.scheduler_name = scheduler
        self.seed = seed
        self.guidance_scale = guidance_scale
        self.denoising_steps = denoising_steps
        self.num_images_per_prompt = num_images_per_prompt
        self.output_dir = output_dir
        self.lora = lora or []
        self.adapter_weights = adapter_weights or []
        self.lora_scale = lora_scale
        self.merge_lora_flag = merge_lora
        self.model_type = model_type

        os.makedirs(self.output_dir, exist_ok=True)
        self.pipe = self._load_model()

    def _load_model(self):
        dtype = torch.float16 if self.device in ["cuda", "gcu"] else torch.float32

        if self.model_type == "sdxl":
            pipe = DiffusionPipeline.from_pretrained(self.model_dir, torch_dtype=dtype)
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.vae.to(memory_format=torch.channels_last)
        elif self.model_type == "sd3":
            pipe = StableDiffusion3Pipeline.from_pretrained(self.model_dir, torch_dtype=dtype)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        pipe = pipe.to(self.device)
        return pipe

    def _set_scheduler(self):
        if self.model_type == "sdxl":
            from diffusers import DDIMScheduler
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def _prepare_prompts(self, prompt, prompt_2=None, prompt_3=None,
                         negative_prompt=None, negative_prompt_2=None, negative_prompt_3=None):
        batch_size = len(prompt)
        prompt_2 = prompt_2 or [''] * batch_size
        prompt_3 = prompt_3 or [''] * batch_size
        negative_prompt = negative_prompt or [''] * batch_size
        negative_prompt_2 = negative_prompt_2 or [''] * batch_size
        negative_prompt_3 = negative_prompt_3 or [''] * batch_size

        prompt_2 = prompt_2[:batch_size] if len(prompt_2) > batch_size else prompt_2 + [''] * (batch_size - len(prompt_2))
        prompt_3 = prompt_3[:batch_size] if len(prompt_3) > batch_size else prompt_3 + [''] * (batch_size - len(prompt_3))
        negative_prompt = negative_prompt[:batch_size] if len(negative_prompt) > batch_size else negative_prompt + [''] * (batch_size - len(negative_prompt))
        negative_prompt_2 = negative_prompt_2[:batch_size] if len(negative_prompt_2) > batch_size else negative_prompt_2 + [''] * (batch_size - len(negative_prompt_2))
        negative_prompt_3 = negative_prompt_3[:batch_size] if len(negative_prompt_3) > batch_size else negative_prompt_3 + [''] * (batch_size - len(negative_prompt_3))

        return prompt, prompt_2, prompt_3, negative_prompt, negative_prompt_2, negative_prompt_3

    def _generate_latents(self):
        seed = self.seed if self.seed is not None else 42
        generator = torch.Generator(device='cpu').manual_seed(seed)

        vae_scale_factor = self.pipe.vae_scale_factor
        num_channels_latents = 4 if self.model_type == "sdxl" else self.pipe.transformer.config.in_channels
        shape = (self.num_images_per_prompt * len(self.prompt), num_channels_latents,
                 self.image_height // vae_scale_factor, self.image_width // vae_scale_factor)
        latents = randn_tensor(shape, generator=generator, device=torch.device('cpu'), dtype=torch.float16)
        if self.device == 'cpu':
            latents = latents.to(torch.float32)
        return latents

    def run(self, prompt, prompt_2=None, prompt_3=None,
            negative_prompt=None, negative_prompt_2=None, negative_prompt_3=None):
        """
        执行图像生成任务。

        支持根据模型类型自动适配参数。
        """
        self.prompt = prompt
        self._set_scheduler()

        # 准备提示词
        prompt, prompt_2, prompt_3, negative_prompt, negative_prompt_2, negative_prompt_3 = self._prepare_prompts(
            prompt, prompt_2, prompt_3, negative_prompt, negative_prompt_2, negative_prompt_3
        )

        # 生成 latent
        latents = self._generate_latents()

        # 构造参数
        common_kwargs = {
            "prompt": prompt,
            "height": self.image_height,
            "width": self.image_width,
            "num_images_per_prompt": self.num_images_per_prompt,
            "num_inference_steps": self.denoising_steps,
            "guidance_scale": self.guidance_scale,
            "latents": latents
        }

        # 根据模型类型添加不同参数
        if self.model_type == "sdxl":
            cross_attention_kwargs = {"scale": self.lora_scale} if len(self.lora) > 0 and not self.merge_lora_flag else None
            common_kwargs.update({
                "prompt_2": prompt_2,
                "negative_prompt": negative_prompt,
                "negative_prompt_2": negative_prompt_2,
                "cross_attention_kwargs": cross_attention_kwargs
            })
        elif self.model_type == "sd3":
            common_kwargs.update({
                "prompt_2": prompt_2,
                "prompt_3": prompt_3,
                "negative_prompt": negative_prompt,
                "negative_prompt_2": negative_prompt_2,
                "negative_prompt_3": negative_prompt_3
            })

        # 执行推理
        images = self.pipe(**common_kwargs).images

        # 保存图像
        self._save_images(images, prompt, seed=self.seed if self.seed else 42)

    def _save_images(self, images, prompt_list, seed):
        for i, prompt in enumerate(prompt_list):
            prompt_str = prompt.replace(" ", "_")[:130]
            for j in range(self.num_images_per_prompt):
                image = images[i * self.num_images_per_prompt + j]
                img_name = f"{seed}-prompt_{i}-img_{j}-steps_{self.denoising_steps}-cfg_{self.guidance_scale}-{prompt_str}.png"
                t_save_start = time.time()
                print(f"Saving {img_name}... (模拟保存)")
                t_save_end = time.time()
                print(f'saving current picture costs time: {t_save_end - t_save_start}')


# ================== 测试用例 ==================
if __name__ == "__main__":
    print("=== 测试 SDXL ===")
    sdxl_pipeline = UnifiedDiffusionPipeline(
        model_dir="/models/stable-diffusion-xl-base-1.0",
        device="gcu",
        image_height=512,
        image_width=512,
        scheduler="ddim",
        seed=42,
        lora=["/models/latent-consistency-lcm-lora-sdxl"],
        adapter_weights=[0.5],
        lora_scale=0.8,
        merge_lora=True,
        output_dir="/home/images",
        model_type="sdxl"
    )

    sdxl_pipeline.run(
        prompt=["A beautiful dragon flying in the sky"],
        prompt_2=[""],
        negative_prompt=["low quality, blurry"],
        negative_prompt_2=[""]
    )

    print("\n=== 测试 SD3 ===")
    sd3_pipeline = UnifiedDiffusionPipeline(
        model_dir="stabilityai/stable-diffusion-3-medium",
        device="cpu",
        image_height=1024,
        image_width=1024,
        seed=42,
        output_dir="output_sd3",
        model_type="sd3"
    )

    sd3_pipeline.run(
        prompt=["A futuristic city at night"],
        prompt_2=["detailed architecture"],
        prompt_3=["high resolution"],
        negative_prompt=["low quality, cartoon style"],
        negative_prompt_2=["dark shadows"],
        negative_prompt_3=["blurry"]
    )
