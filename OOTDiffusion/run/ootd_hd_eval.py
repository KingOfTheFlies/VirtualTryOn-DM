import os

import json
from os import path as osp

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils import data
from torchvision import transforms
import argparse
from utils.dataloader import VITONDataset,VITONDataLoader,make_train_dataset
from safetensors.torch import save_file
from torchvision.transforms import ToTensor

import warnings
warnings.filterwarnings("ignore")

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=False)

    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--load_height', type=int, default=1024)
    parser.add_argument('--load_width', type=int, default=768)
    parser.add_argument('--shuffle', action='store_true')

    parser.add_argument('--dataset_dir', type=str, default='dataset_dir')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('--dataset_list', type=str, default='test_pairs.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--save_dir', type=str, default='./results/')


    # common
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of human-parsing map classes')
    parser.add_argument('--init_type', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'], default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    # for GMM
    parser.add_argument('--grid_size', type=int, default=5)

    # for ALIASGenerator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                        help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                             'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')
    

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./OOTD-model",
        help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--model_type", type=str, default='hd', help="hd or dc."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )


    # my args
    parser.add_argument(
        "--do_classifier_free_guidance",
        type = bool,
        default=False,
    )

    parser.add_argument(
        "--image_guidance_scale",
        type = float,
        default=1.,
    )
    
    parser.add_argument(
        "--save_outputs",
        type = bool,
        default=False,
    )

    parser.add_argument(
        "--save_path", type=str, default='./ootd_output_results', help="dir of saving result images"
    )

    # opt = parser.parse_args()
    opt, unknown = parser.parse_known_args()
    return opt


def log_memory(stage):
    print(f"[{stage}] Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"[{stage}] Reserved memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"[{stage}] Max allocated memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    print("---------------------------------------------------")

def log_params(name, model):
    print("Trainable params in", name, ":", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Total params in", name, ":", sum(p.numel() for p in model.parameters()))
    print("---------------------------------------------------")

import sys

log_memory("Start")

opt = get_opt()
opt.batch_size = opt.train_batch_size

test_dataset = VITONDataset(opt)
test_loader = VITONDataLoader(opt, test_dataset)

train_dataset = test_dataset
train_dataloader = test_loader.data_loader

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from diffusers.optimization import get_scheduler
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import UniPCMultistepScheduler,PNDMScheduler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
import torch.nn.functional as F
sys.path.append(r'../ootd')

from pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel
from pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel

VIT_PATH = "VIT_PATH"
VAE_PATH = "VAE_PATH"
MODEL_PATH = "MODEL_PATH"
scheduler_path = 'scheduler_path'

UNET_PATH = "UNET_PATH"

vae = AutoencoderKL.from_pretrained(
            VAE_PATH,
            torch_dtype=torch.float16,
        )

unet_garm = UNetGarm2DConditionModel.from_pretrained(
            UNET_PATH,
            subfolder="unet_garm_train",
            torch_dtype=torch.float32,
            # use_safetensors=True,
        )

unet_vton = UNetVton2DConditionModel.from_pretrained(
            UNET_PATH,
            subfolder="unet_vton_train",
            torch_dtype=torch.float32,
            use_safetensors=True,
            # low_cpu_mem_usage=False,
            # ignore_mismatched_sizes=True,                       # ignoring
        )

noise_scheduler = PNDMScheduler.from_config(scheduler_path)
        
auto_processor = AutoProcessor.from_pretrained(VIT_PATH)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH)

tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_PATH,
            subfolder="tokenizer",
        )
text_encoder = CLIPTextModel.from_pretrained(
            MODEL_PATH,
            subfolder="text_encoder",
        )
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

vae.requires_grad_(False)
unet_garm.requires_grad_(False)
unet_vton.requires_grad_(False)
image_encoder.requires_grad_(False)
text_encoder.requires_grad_(False)


import math
from pathlib import Path
args = opt
logging_dir = Path(args.output_dir, args.logging_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weight_dtype = torch.float32              # For float32

log_memory("Before moving to device")

print("Model device:", device)
image_encoder.to(device, dtype=weight_dtype)
log_memory("After image_encoder to device")
log_params("image_encoder", image_encoder)

text_encoder.to(device, dtype=weight_dtype)
log_memory("After text_encoder to device")
log_params("text_encoder", text_encoder)

vae.to(device, dtype=weight_dtype)
log_memory("After vae to device")
log_params("vae", vae)

unet_garm.to(device)
log_memory("After unet_garm to device")
log_params("unet_garm", unet_garm)

unet_vton.to(device)
log_memory("After unet_vton to device")
log_params("unet_vton", unet_vton)


log_memory("After Moving to device")

# Metrics
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

fid = FrechetInceptionDistance(feature=64).to(device)
kid = KernelInceptionDistance(subset_size=50).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=255).to(device)
lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)

vae.eval()
unet_garm.eval()
unet_vton.eval()
image_encoder.eval()
text_encoder.eval()

def tokenize_captions( captions, max_length):
        inputs = tokenizer(
            captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

batchsize = args.train_batch_size

from tqdm.auto import tqdm
print("Starting validation...")
all_lpips = []
all_losses = []
total_samples = 0

results_dir = args.save_path
os.makedirs(results_dir, exist_ok=True)

with torch.no_grad():
    for step, batch in tqdm(enumerate(train_dataloader), desc="Validating ViTONHD"):
        batch_size = batch['img'].size(0)
        total_samples += batch_size

        image_garm = batch['cloth']['paired'].to(device).to(dtype=weight_dtype)
        print("image_Resolution", image_garm[0].shape)
        image_vton = batch['img_agnostic'].to(device).to(dtype=weight_dtype)
        image_ori = batch['img'].to(device).to(dtype=weight_dtype)

        prompt_image = auto_processor(images=image_garm, return_tensors="pt").to(device)
        prompt_image = image_encoder(prompt_image.data['pixel_values']).image_embeds
        prompt_image = prompt_image.unsqueeze(1)
        
        if args.model_type == 'hd':
            prompt_embeds = text_encoder(tokenize_captions([''] * batchsize, 2).to(device))[0]
            prompt_embeds[:, 1:] = prompt_image[:]
    
        image_garm = image_processor.preprocess(image_garm)
        image_vton = image_processor.preprocess(image_vton)
        image_ori = image_processor.preprocess(image_ori)

        latents = vae.encode(image_ori).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=device)
        
        bs_embed, seq_len, _ = prompt_embeds.shape
        num_images_per_prompt = 1
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        image_latents_garm = vae.encode(image_garm).latent_dist.mode()
        image_latents_garm = torch.cat([image_latents_garm], dim=0)

        image_latents_vton = vae.encode(image_vton).latent_dist.mode()        
        image_latents_vton = torch.cat([image_latents_vton], dim=0)

        sample, spatial_attn_outputs = unet_garm(
            image_latents_garm,
            0,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )
        print("spatial_attn_outputs len", len(spatial_attn_outputs))
        latent_vton_model_input = torch.cat([noisy_latents, image_latents_vton], dim=1)

        if args.do_classifier_free_guidance:
            latent_vton_model_input = torch.cat([latent_vton_model_input] * 2, dim=0)  # Удваиваем данные
            prompt_embeds = torch.cat([prompt_embeds, torch.zeros_like(prompt_embeds)], dim=0)  # Добавляем нулевые условия
            timesteps = torch.cat([timesteps, timesteps], dim=0)  # Удваиваем timesteps
            spatial_attn_outputs = [torch.cat([output, output], dim=0) for output in spatial_attn_outputs]
            # noisy_latents = torch.cat([noisy_latents] * 2, dim=0)

        noise_pred = unet_vton(
            latent_vton_model_input,
            spatial_attn_outputs,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        if args.do_classifier_free_guidance:
            noise_pred_text_image, noise_pred_text = noise_pred.chunk(2)
            noise_pred = (
                noise_pred_text
                + args.image_guidance_scale * (noise_pred_text_image - noise_pred_text)
            )
            # noise_pred = noise_pred[:noisy_latents.size(0)]

        alpha_t = noise_scheduler.alphas_cumprod.to(device)[timesteps].sqrt().view(-1, 1, 1, 1)
        alpha_t_bar = (1 - noise_scheduler.alphas_cumprod.to(device)[timesteps]).sqrt().view(-1, 1, 1, 1)

        alpha_t = alpha_t[:noise_pred.shape[0]]
        alpha_t_bar = alpha_t_bar[:noise_pred.shape[0]]

        denoised_latents = (noisy_latents - alpha_t_bar.to(device) * noise_pred) / alpha_t.to(device)

        decoded_images = vae.decode(denoised_latents / vae.config.scaling_factor, return_dict=False)[0]
        decoded_pil_images = image_processor.postprocess(decoded_images, output_type="pil", do_denormalize=[True] * decoded_images.shape[0])

        loss = F.mse_loss(noise_pred.float(), noise.float())
        all_losses.append(loss.item() * batch_size)

        if args.save_outputs:
            for i, img in enumerate(decoded_pil_images):
                print("Output image shape:", img.size)
                save_path = os.path.join(results_dir, f"{batch['img_name'][i]}.png")
                img.save(save_path)

        def convert_to_uint8(images):
            images = (images * 255).clamp(0, 255).to(torch.uint8)
            return images

        def normalize_images(images, value_range=(-1, 1)):
            min_val, max_val = value_range
            images = images.clamp(min_val, max_val)
            return images

        image_ori_uint8 = convert_to_uint8(image_ori)
        decoded_images_uint8 = convert_to_uint8(decoded_images)

        fid.update(image_ori_uint8, real=True)
        fid.update(decoded_images_uint8, real=False)

        kid.update(image_ori_uint8, real=True)
        kid.update(decoded_images_uint8, real=False)

        lpips_score = lpips(normalize_images(image_ori), normalize_images(decoded_images)).item()
        all_lpips.append(lpips_score * batch_size)

        ssim_metric.update(image_ori, decoded_images)

fid_score = fid.compute().item()
kid_score = kid.compute()[0].item()
mean_lpips = sum(all_lpips) / total_samples
mean_ssim = ssim_metric.compute().item()
mean_losses = sum(all_losses) / total_samples

print(f"Validation complete. FID: {fid_score:.4f}, KID: {kid_score:.4f}, LPIPS: {mean_lpips:.4f}, SSIM: {mean_ssim:.4f}, Loss: {mean_losses}")