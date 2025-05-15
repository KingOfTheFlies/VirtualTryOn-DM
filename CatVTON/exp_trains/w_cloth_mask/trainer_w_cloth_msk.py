import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import inspect
import os
from typing import Union

import PIL
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

import wandb

from tqdm import tqdm
from accelerate import load_checkpoint_in_model, Accelerator
# from accelerate.logging import get_logger       # TODO: logger in train.py
# logger = get_logger(__name__)

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.utils.torch_utils import randn_tensor

from huggingface_hub import snapshot_download
from transformers import CLIPImageProcessor

from model.attn_processor import SkipAttnProcessor
from model.utils import get_trainable_module, init_adapter
from utils import (compute_vae_encodings, numpy_to_pil, prepare_image,
                   prepare_mask_image, resize_and_crop, resize_and_padding)


class CatVTONTrainerWithClothMask:
    def __init__(
        self, 
        base_ckpt,
        train_dataset,
        train_dataloader,
        attn_ckpt,
        save_attn_ckpt_pth,
        opt_ckpt="no",
        cfg_dropout_prob=0.1,
        attn_ckpt_version="mix",
        weight_dtype=torch.float32,
        device='cpu',
        compile=False,
        skip_safety_check=True,            # TODO: remove
        use_tf32=True,
        batch_size=32,
        learning_rate=1e-5,
        weight_decay=0.01,
        max_train_steps=60000,
        log_step=250,
        wandb_logger=None,
        save_chkp_every=1000,
    ):
        self.device = device
        self.weight_dtype = weight_dtype
        self.skip_safety_check = skip_safety_check
        self.max_train_steps = max_train_steps
        self.cfg_dropout_prob = cfg_dropout_prob
        self.log_step = log_step
        
        self.wandb = wandb_logger

        
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader

        self.save_every = save_chkp_every
        self.save_attn_ckpt_pth = save_attn_ckpt_pth
        
        self.concat_dim = -1 # по горизонтали

        # assert skip_safety_check and not attn_ckpt, "attn_ckpt is blank"        # TODO

        self.noise_scheduler = DDIMScheduler.from_pretrained(base_ckpt, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device, dtype=weight_dtype)
        if not skip_safety_check:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(base_ckpt, subfolder="feature_extractor")
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(base_ckpt, subfolder="safety_checker").to(device, dtype=weight_dtype)
        self.unet = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet").to(device, dtype=weight_dtype)
        init_adapter(self.unet, cross_attn_cls=SkipAttnProcessor)  # Skip Cross-Attention
        self.attn_modules = get_trainable_module(self.unet, "attention")
        
        # self.original_pretrained_attn_ckpt_load(attn_ckpt, attn_ckpt_version)
        
        if attn_ckpt != "no":                  # TODO: improve
            self.load_ft_attn_checkpoint(attn_ckpt)
            
        if opt_ckpt != "no":
            self.load_opt_checkpoint(opt_ckpt)

        # Pytorch 2.0 Compile
        if compile:
            self.unet = torch.compile(self.unet)
            self.vae = torch.compile(self.vae, mode="reduce-overhead")
        # Enable TF32 for faster training on Ampere GPUs (A100 and RTX 30 series).
        if use_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            
            
        self.on_train_mode()
        trainable_params = list(p for p in self.attn_modules.parameters() if p.requires_grad)
        self.optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

            
        # logger.info("~~~~~ Trainer successfully intialized ~~~~~")
        print("~~~~~ Trainer successfully intialized ~~~~~")


    def on_train_mode(self):
        for param in self.unet.parameters():
            param.requires_grad = False

        for param in self.attn_modules.parameters():
            param.requires_grad = True

        self.unet.train()
        
    def original_pretrained_attn_ckpt_load(self, attn_ckpt, version):       # for loading original checkpoint
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }[version]
        if os.path.exists(attn_ckpt):
            load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, sub_folder, 'attention'))
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, sub_folder, 'attention'))
            
    def attn_ckpt_load(self, attn_ckpt, version):           # TODO: to continue training (load from checkpoint)
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }[version]
        if os.path.exists(attn_ckpt):
            load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, sub_folder, 'attention'))
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, sub_folder, 'attention'))
            
            
    def load_opt_checkpoint(self, path):
        self.optimizer.load_state_dict(torch.load(path))
        print(f"Optimizer checkpoint loaded from: {path}")
            
    def save_ft_attn_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        att_ckp_path = path+".pth"
        torch.save(self.attn_modules.state_dict(), att_ckp_path)
        print(f"Attention module checkpoint saved to: {att_ckp_path}")      # TODO: to logs

    def save_opt_checkpoint(self, path):
        opt_path = path+'.pth'
        torch.save(self.optimizer.state_dict(), opt_path)
        print(f"Optimizer module checkpoint saved to: {opt_path}")
        
  
    def load_ft_attn_checkpoint(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        state_dict = torch.load(path, map_location=self.device)
        self.attn_modules.load_state_dict(state_dict)
        print(f"Attention module weights loaded from: {path}")

    # def run_safety_checker(self, image):        # TODO: Remove
    #     if self.safety_checker is None:
    #         has_nsfw_concept = None
    #     else:
    #         safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.device)
    #         image, has_nsfw_concept = self.safety_checker(
    #             images=image, clip_input=safety_checker_input.pixel_values.to(self.weight_dtype)
    #         )
    #     return image, has_nsfw_concept
    
    def check_inputs(self, image, condition_image, mask, width, height):
        if isinstance(image, torch.Tensor) and isinstance(condition_image, torch.Tensor) and isinstance(mask, torch.Tensor):
            return image, condition_image, mask
        assert image.size == mask.size, "Image and mask must have the same size"
        image = resize_and_crop(image, (width, height))
        mask = resize_and_crop(mask, (width, height))
        condition_image = resize_and_padding(condition_image, (width, height))
        return image, condition_image, mask
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def train(self):
        
        
        # logger.info("***** Running training *****")
        # logger.info(f"  Num examples = {len(self.train_dataset)}")
        # logger.info(f"  batch_size = {self.batch_size}")
        # logger.info(f"  Num steps = {self.max_train_steps}")
        # logger.info(f"  Trainable parameters: {sum(p.numel() for p in self.unet.parameters() if p.requires_grad)}")
        # logger.info(f"  Total parameters: {sum(p.numel() for p in self.unet.parameters())}")
        # logger.info(f"  Model is on cuda: {next(self.unet.parameters()).is_cuda}")
        
        print("***** Running training *****")
        print(f"  Num examples = {len(self.train_dataset)}")
        print(f"  batch_size = {self.batch_size}")
        print(f"  Num steps = {self.max_train_steps}")
        print(f"  Trainable parameters: {sum(p.numel() for p in self.unet.parameters() if p.requires_grad)}")
        print(f"  Total parameters: {sum(p.numel() for p in self.unet.parameters())}")
        print(f"  Model is on cuda: {next(self.unet.parameters()).is_cuda}")
        
        progress_bar = tqdm(
            range(0, self.max_train_steps),
            desc="Training Steps",
        )
        
        step = 0
        loss_history = []           # TODO: remove

        train_iter = iter(self.train_dataloader)
        
        
        while step < self.max_train_steps:
            try:
                batch = next(train_iter)  # Берем следующий батч
            except StopIteration:
                train_iter = iter(self.train_dataloader)  # Перезапускаем даталоадер
                batch = next(train_iter)

            # Подготовка данных
            images = batch["img"]
            condition_images = batch["cloth"]
            masks = batch["agnostic_mask"][:, 0, :, :].unsqueeze(1)     # TODO: correct mask?
            cond_images_masks = batch["cloth_mask"][:, 0, :, :].unsqueeze(1)     # TODO: correct mask?

            images = images.to(self.device)
            condition_images = condition_images.to(self.device)
            masks = masks.to(self.device)
            cond_images_masks = cond_images_masks.to(self.device)
            
            # print("Mask_IS_NON_ZERO", torch.all(masks == 0.))
            
            masked_images = images * (masks < 0.5)         #TODO: BOBABOBA маскирование входного изображения

            # print(f"Models Image shape:{images.shape}")
            # print(f"Colthes Image shape:{condition_images.shape}")
            # print(f"Masks Image shape:{masks.shape}")            
            
            
            with torch.no_grad():
                image_latents = self.vae.encode(images).latent_dist.sample()                # TODO: use compute_vae_encodings
                masked_image_latents = self.vae.encode(masked_images).latent_dist.sample()
                condition_latents = self.vae.encode(condition_images).latent_dist.sample()
                
            image_latents = image_latents * self.vae.config.scaling_factor
            masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
            condition_latents = condition_latents * self.vae.config.scaling_factor
            
            mask_latents = F.interpolate(masks, size=image_latents.shape[-2:], mode="nearest")
            cond_mask_latents = F.interpolate(cond_images_masks, size=condition_latents.shape[-2:], mode="nearest")

            # print(f"image_latents shape:{image_latents.shape}")
            # print(f"condition_latents shape:{condition_latents.shape}")
            # print(f"mask_latents shape:{mask_latents.shape}")   

            # TODO: корректно обнулить
            condition_latents_orig = condition_latents.clone()
            condition_latents_uncond = torch.zeros_like(condition_latents)
            mask_dropout = torch.rand(images.shape[0], device=self.device) < self.cfg_dropout_prob
            condition_latents[mask_dropout] = condition_latents_uncond[mask_dropout]
            cond_mask_latents[mask_dropout] = cond_mask_latents[mask_dropout]

            # print("mask_dropout:", mask_dropout)
            
            
            masked_latent_concat = torch.cat([masked_image_latents, condition_latents], dim=self.concat_dim)
            mask_latent_concat = torch.cat([mask_latents, cond_mask_latents], dim=self.concat_dim)              # BOBABOBA: cond_mask_latents вместо torch.zeros_like(mask_latents)
            
            target_latent_concat = torch.cat([image_latents, condition_latents_orig], dim=self.concat_dim)
            
            # print(f"masked_latent_concat shape:{masked_latent_concat.shape}")
            # print(f"mask_latent_concat shape:{mask_latent_concat.shape}")
            
            # Генерация шума
            noise = torch.randn_like(target_latent_concat)
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (images.shape[0],), device=self.device).long()
            noisy_latents = self.noise_scheduler.add_noise(target_latent_concat, noise, timesteps)

            # Входной тензор для UNet
            model_input = torch.cat([noisy_latents, mask_latent_concat, masked_latent_concat], dim=1)

            noise_pred = self.unet(model_input, timesteps, encoder_hidden_states=None, return_dict=False)[0]

            loss = F.mse_loss(noise_pred, noise)
            # print("LOSS:", loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            current_step = step
            
            if self.wandb is not None:
                self.wandb.log({"train/loss": loss.item(),  "global_step": current_step}, step=current_step)

            loss_history.append(loss.item())
            progress_bar.set_postfix({"loss": loss.item()})


            # Сохранение чекпоинта      TODO: saving only attention weights accelerator needed + step (state)
            if step != 0 and step % self.save_every == 0:
                self.save_ft_attn_checkpoint(f"{self.save_attn_ckpt_pth}/checkpoint_{current_step}")
                self.save_opt_checkpoint(f'{self.save_attn_ckpt_pth}/opt_state_last')


            if self.wandb is not None and step % self.log_step == 0:       #TODO: add eval on test
                self.unet.eval()
                sample_image = images[0].unsqueeze(0)  # берем 1 картинку
                sample_condition = condition_images[0].unsqueeze(0)
                sample_mask = masks[0].unsqueeze(0)
                sample_masked_image = masked_images[0].unsqueeze(0)
                sample_cond_images_masks = cond_images_masks[0].unsqueeze(0)
                
                log_images = [
                    wandb.Image(batch["img"][0].unsqueeze(0).cpu(), caption="Original"),
                    wandb.Image(sample_mask.cpu(), caption="Agnostic Mask"),
                    wandb.Image(sample_condition.cpu(), caption="Condition (Cloth)"),
                    wandb.Image(sample_cond_images_masks.cpu(), caption="Cloth Mask"),
                ]
                
                generated_images = self.orig_infer(
                    sample_image,
                    sample_condition,
                    sample_mask,
                    num_inference_steps=75,
                    guidance_scale=1.,     # TODO: w/h
                    
                )
                log_images.append(wandb.Image(sample_masked_image.cpu(), caption="Masked"))
                log_images.append(wandb.Image(generated_images[0], caption="Generated"))
                
                # mask_for_display = sample_mask.squeeze(0)  # [1, H, W] -> [H, W]
                # if mask_for_display.ndim == 2:
                #     mask_for_display = mask_for_display.unsqueeze(0)  # [1, H, W]
                # mask_for_display = mask_for_display.repeat(3, 1, 1)   # [3, H, W]
                
                

                # Логируем в один блок, все изображения
                self.wandb.log({"examples": log_images}, step=current_step, commit=True)
                self.unet.train()
                
            step += 1
            progress_bar.update(1)


    @torch.no_grad()
    def orig_infer(
        self, 
        image: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        mask: Union[PIL.Image.Image, torch.Tensor],
        num_inference_steps: int = 75,
        guidance_scale: float = 2.5,
        height: int = 512,
        width: int = 384,
        generator=None,
        eta=1.0,
        **kwargs
    ):
        # Prepare inputs to Tensor
        image, condition_image, mask = self.check_inputs(image, condition_image, mask, width, height)
        image = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        # mask = prepare_mask_image(mask).to(self.device, dtype=self.weight_dtype)
        mask = mask[:, 0, :, :].unsqueeze(1).to(self.device, dtype=self.weight_dtype)
        # Mask image
        masked_image = image * (mask < 0.5)
        # VAE encoding
        masked_latent = compute_vae_encodings(masked_image, self.vae)
        condition_latent = compute_vae_encodings(condition_image, self.vae)
        mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
        del image, mask, condition_image
        # Concatenate latents
        masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=self.concat_dim)
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=self.concat_dim)
        # Prepare noise
        latents = randn_tensor(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.weight_dtype,
        )
        # Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma
        # Classifier-Free Guidance
        if do_classifier_free_guidance := (guidance_scale > 1.0):
            masked_latent_concat = torch.cat(
                [
                    torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=self.concat_dim),
                    masked_latent_concat,
                ]
            )
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)

        # Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (len(timesteps) - num_inference_steps * self.noise_scheduler.order)
        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                non_inpainting_latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
                non_inpainting_latent_model_input = self.noise_scheduler.scale_model_input(non_inpainting_latent_model_input, t)
                # prepare the input for the inpainting model
                inpainting_latent_model_input = torch.cat([non_inpainting_latent_model_input, mask_latent_concat, masked_latent_concat], dim=1)
                # predict the noise residual
                noise_pred= self.unet(
                    inpainting_latent_model_input,
                    t.to(self.device),
                    encoder_hidden_states=None, # FIXME
                    return_dict=False,
                )[0]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        # Decode the final latents
        latents = latents.split(latents.shape[self.concat_dim] // 2, dim=self.concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)
        return image
