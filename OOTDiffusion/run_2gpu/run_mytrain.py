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

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=False)

    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--load_height', type=int, default=1024)
    parser.add_argument('--load_width', type=int, default=768)
    parser.add_argument('--shuffle', action='store_true')

    parser.add_argument('--dataset_dir', type=str, default='')
    parser.add_argument('--dataset_mode', type=str, default='train')
    parser.add_argument('--dataset_list', type=str, default='train_pairs.txt')
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
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
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
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
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
    parser.add_argument(

        "--conditioning_dropout_prob",
        type = float,
        default=0.1,
        
    )
    
    # opt = parser.parse_args()
    opt, unknown = parser.parse_known_args()
    return opt


def log_memory(stage):
    num_devices = torch.cuda.device_count()
    for device_id in range(num_devices):
        torch.cuda.set_device(device_id)  # Устанавливаем активное устройство
        print(f"[{stage}] GPU {device_id}:")
        print(f"  Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  Reserved memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"  Max allocated memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        print("---------------------------------------------------")

def log_params(name, model):
    print("Trainable params in", name, ":", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Total params in", name, ":", sum(p.numel() for p in model.parameters()))
    print("---------------------------------------------------")

import sys
# sys.argv = ['ootd_train.py']

log_memory("Start")

opt = get_opt()
opt.batch_size = opt.train_batch_size

test_dataset = VITONDataset(opt)
test_loader = VITONDataLoader(opt, test_dataset)
# test_loader.data_loader.sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
# torch.distributed.init_process_group(backend="nccl", rank=0, world_size=2)
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
UNET_PATH = "UNET_PATH"
MODEL_PATH = "MODEL_PATH"
scheduler_path = 'scheduler_path'

vae = AutoencoderKL.from_pretrained(
            VAE_PATH,
            torch_dtype=torch.float16,
        )

unet_garm = UNetGarm2DConditionModel.from_pretrained(
            UNET_PATH,
            subfolder="unet_garm_train",
            torch_dtype=torch.float32,
            use_safetensors=True,
        )

unet_vton = UNetVton2DConditionModel.from_pretrained(
            UNET_PATH,
            subfolder="unet_vton_train",
            torch_dtype=torch.float32,
            use_safetensors=True,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,                       # ignoring
        )


################
if unet_vton.conv_in.in_channels == 8:
    print("sxxxxxxxxxx")
    with torch.no_grad():
        new_in_channels = 8
        # Replace the first conv layer of the unet with a new one with the correct number of input channels
        conv_new = torch.nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=unet_vton.conv_in.out_channels,
            kernel_size=3,
            padding=1,
        )
        
        torch.nn.init.kaiming_normal_(conv_new.weight)  # Initialize new conv layer
        conv_new.weight.data = conv_new.weight.data * 0.  # Zero-initialize new conv layer
        
        conv_new.weight.data[:, :4] = unet_vton.conv_in.weight.data[:, :4]  # Copy weights from old conv layer
        conv_new.bias.data = unet_vton.conv_in.bias.data  # Copy bias from old conv layer
        
        unet_vton.conv_in = conv_new  # replace conv layer in unet
        print('#######Replace the first conv layer of the unet with a new one with the correct number of input channels#######')
        # unet_garm.config['in_channels'] = new_in_channels  # update config
#################


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
# register_to_config(requires_safety_checker=requires_safety_checker)

vae.requires_grad_(False)
unet_garm.requires_grad_(True)
unet_vton.requires_grad_(True)
image_encoder.requires_grad_(False)
text_encoder.requires_grad_(False)

unet_garm.train()
unet_vton.train()
 # Optimizer creation
import math
from pathlib import Path
args = opt
logging_dir = Path(args.output_dir, args.logging_dir)

accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        device_placement=True,              # for multi
    )

optimizer_class = torch.optim.AdamW
######单机单卡
params_to_optimize = list(unet_garm.parameters()) + list(unet_vton.parameters())
optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
#######单机多卡
# params_to_optimize = list(unet_garm.parameters())
# optimizer = optimizer_class(
#         params_to_optimize,
#         lr=args.learning_rate,
#         betas=(args.adam_beta1, args.adam_beta2),
#         weight_decay=args.adam_weight_decay,
#         eps=args.adam_epsilon,
#     )
# optimizer.add_param_group({'params': list(unet_vton.parameters())})

# Scheduler and math around the number of training steps.
overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if args.max_train_steps is None:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

# if accelerator.state.deepspeed_plugin is not None:
#   kwargs = {
              
#               "train_micro_batch_size_per_gpu": 1,
#               "train_batch_size": 1,
              
#           } 
#     accelerator.state.deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)c


#Prepare everything with our `accelerator`.
# class Unet_(torch.nn.Module):
#     def __init__(self, vton_model, garm_model):
#         super(Unet_, self).__init__()
#         self.unet_vton = vton_model
#         self.unet_garm = garm_model
#         # 其他初始化代码...

# unet_ = Unet_(unet_vton, unet_garm)

# unet_ ,optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
#         unet_, optimizer, train_dataloader, lr_scheduler
#     )



# unet_garm,unet_vton,optimizer, train_dataloader,test_loader, lr_scheduler = accelerator.prepare(
#          unet_garm,unet_vton,optimizer, train_dataloader, test_loader,lr_scheduler
#     )

# weight_dtype = torch.float32
# if accelerator.mixed_precision == "fp16":
#     weight_dtype = torch.float16
# elif accelerator.mixed_precision == "bf16":
#     weight_dtype = torch.bfloat16
weight_dtype = torch.float32              # For float32

log_memory("Before moving to device")

print("Model device:", accelerator.device)
# Move vae, unet and text_encoder to device and cast to weight_dtype
image_encoder.to(accelerator.device, dtype=weight_dtype)
log_memory("After image_encoder to device")
log_params("image_encoder", image_encoder)

text_encoder.to(accelerator.device, dtype=weight_dtype)
log_memory("After text_encoder to device")
log_params("text_encoder", text_encoder)

vae.to(accelerator.device, dtype=weight_dtype)
log_memory("After vae to device")
log_params("vae", vae)

unet_garm.to(accelerator.device)
log_memory("After unet_garm to device")
log_params("unet_garm", unet_garm)

unet_vton.to(accelerator.device)
log_memory("After unet_vton to device")
log_params("unet_vton", unet_vton)


log_memory("After Moving to device")

# image_encoder.to("cpu", dtype=weight_dtype)
# text_encoder.to("cpu", dtype=weight_dtype)
# vae.to(accelerator.device, dtype=weight_dtype)
#--------------------------------------------------
# unet_garm.to(dtype=weight_dtype)
# unet_vton.to(dtype=weight_dtype)




# We need to recalculate our total training steps as the size of the training dataloader may have changed.
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if overrode_max_train_steps:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
# Afterwards we recalculate our number of training epochs
args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

def tokenize_captions( captions, max_length):
        inputs = tokenizer(
            captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

batchsize = args.train_batch_size

from accelerate.logging import get_logger
from tqdm import tqdm
logger = get_logger(__name__)
# Train!
total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(train_dataset)}")
logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
logger.info(f"  Num Epochs = {args.num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {args.max_train_steps}")

print("***** Running training *****")
print(f"  Num examples = {len(train_dataset)}")
print(f"  Num batches each epoch = {len(train_dataloader)}")
print(f"  Num Epochs = {args.num_train_epochs}")
print(f"  Instantaneous batch size per device = {args.train_batch_size}")
print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
print(f"  Total optimization steps = {args.max_train_steps}")

global_step = 0
first_epoch = 0

initial_global_step = 0

# progress_bar = tqdm(
#         range(0, args.max_train_steps),
#         initial=initial_global_step,
#         desc="Steps",
#         # Only show the progress bar once on each machine.
#         disable=not accelerator.is_local_main_process,
#     )

import wandb

wandb.login(key="API_KEY", relogin=True)


if accelerator.is_main_process:
    wandb.init(project="OOTD-train")
    wandb.config.update(args)
else:
    print("Secondary process; skipping wandb initialization")


def decode_and_save_image(latents, save_path, filename, save_loc=False):
    """Сохраняет изображение из латентов."""
    with torch.no_grad():
        # Декодируем латенты обратно в изображение
        image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        image = image_processor.postprocess(image, output_type="pil", do_denormalize= [True] * image.shape[0])
        # print("IMG SHAPE:", len(image))
        # print("IMG TYPE:", type(image))
        wandb_images = []
        for i, img in enumerate(image):
            if save_loc:
                img.save(os.path.join(save_path, f"{filename}_{i}.png"))
            # img = Image.fromarray(image)
            # img.save(os.path.join(save_path, f"{filename}_{i}.png"))
            if i < 3:
                wandb_images.append(wandb.Image(img, caption=f"{filename}_{i}"))
            else: 
                break
        if accelerator.is_main_process:
            wandb.log({f"Deniosed IMGs": wandb_images})
total_steps = 0
checkpoint_interval = 6000
images_output_interval = 200

for epoch in tqdm(range(first_epoch, args.num_train_epochs)):
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(unet_vton),accelerator.accumulate(unet_garm):
        # with accelerator.accumulate(unet_vton):

            # log_memory("Before step")


            image_garm = batch['cloth']['paired'].to(accelerator.device).to(dtype=weight_dtype)
            image_vton = batch['img_agnostic'].to(accelerator.device).to(dtype=weight_dtype)
            image_ori = batch['img'].to(accelerator.device).to(dtype=weight_dtype)
            # log_memory("After Moving batch to device")
            #get prompt embeds
            prompt_image = auto_processor(images=image_garm, return_tensors="pt").to(accelerator.device)
            prompt_image = image_encoder(prompt_image.data['pixel_values']).image_embeds
            prompt_image = prompt_image.unsqueeze(1)
            # log_memory("After getting prompt embeds")
            
            if args.model_type == 'hd':

                prompt_embeds = text_encoder(tokenize_captions(['']*batchsize, 2).to(accelerator.device))[0]
                prompt_embeds[:, 1:] = prompt_image[:]
            elif args.model_type == 'dc':
                prompt_embeds = text_encoder(tokenize_captions([category], 3).to(accelerator.device))[0]
                prompt_embeds = torch.cat([prompt_embeds, prompt_image], dim=1)
            else:
                raise ValueError("model_type must be \'hd\' or \'dc\'!")
            # log_memory("After getting prompt prompt_embeds")

            ######preprocess把[0,1]转为【-1，1】

            image_garm = image_processor.preprocess(image_garm)
            image_vton = image_processor.preprocess(image_vton)
            image_ori = image_processor.preprocess(image_ori)
            # log_memory("After preprocessing")
            # Convert images to latent space

            latents = vae.encode(image_ori).latent_dist.sample()
            # log_memory("After getting vae latents")
            # latents = vae.encode(image_ori.to(weight_dtype).latent_dist.sample().to(accelerator.device))
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 2. Encode input prompt
            prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=accelerator.device)
            
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            num_images_per_prompt = 1
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            # log_memory("After moving prompt_embeds to device")

            image_latents_garm = vae.encode(image_garm).latent_dist.mode()
            image_latents_garm = torch.cat([image_latents_garm], dim=0)
            # log_memory("After getting image_latents_garm")

            image_latents_vton = vae.encode(image_vton).latent_dist.mode()
            # log_memory("After getting image_latents_vton (before unet_garm)")
            # image_ori_latents = vae.encode(image_ori).latent_dist.mode()
            
            image_latents_vton = torch.cat([image_latents_vton], dim=0)
             
            if args.conditioning_dropout_prob is not None:
                random_p = torch.rand(bsz, device=latents.device)
                #########################################################

                # Sample masks for the cloth images.
                image_mask_dtype = image_latents_garm.dtype
                image_mask = 1 - (
                    (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                    * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                )
                image_mask = image_mask.reshape(bsz, 1, 1, 1)
                # Final image conditioning.
                image_latents_garm = image_mask * image_latents_garm
            ####################################################################

            sample,spatial_attn_outputs = unet_garm(
            image_latents_garm,
            0,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,)
            # log_memory("After unet_garm forward")

            latent_vton_model_input = torch.cat([noisy_latents, image_latents_vton], dim=1)

            # spatial_attn_inputs = spatial_attn_outputs.copy()
            
            noise_pred = unet_vton(
                    latent_vton_model_input,
                    spatial_attn_outputs,
                    # spatial_attn_inputs,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
            # log_memory("After unet_vton forward")

            # with accelerator.autocast():
            # util_adv_loss = torch.nn.functional.softplus(-sample[0]).mean() * 0 
            os.makedirs('./ootd_train_images', exist_ok=True)

            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            if accelerator.is_main_process:
                wandb.log({"loss": loss.item(), "epoch": epoch, "step": total_steps})

            # print(loss.item())

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # progress_bar.update(7)#####单机多卡，每次增加卡的数量
            accelerator.log({"training_loss": loss}, step=step)
            total_steps += 1
            # log_memory("After step")
            if total_steps % images_output_interval == 0:
                timesteps = timesteps.to(noise_scheduler.alphas_cumprod.device)
                noise_pred = noise_pred.to(accelerator.device)
                noisy_latents = noisy_latents.to(accelerator.device)

                alpha_t = noise_scheduler.alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
                alpha_t_bar = (1 - noise_scheduler.alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
                # print("noisy_latents SHAPE", noisy_latents.shape)
                # print("alpha_t_bar SHAPE", alpha_t_bar.shape)
                # print("noise_pred SHAPE", noisy_latents.shape)
                # print("alpha_t SHAPE", alpha_t.shape)
                denoised_latents = (noisy_latents - alpha_t_bar.to(accelerator.device) * noise_pred) / alpha_t.to(accelerator.device)
                wandb_garm = []
                wandb_ori = []
                wandb_vton = []
                for i in range(image_garm.shape[0]):
                    if i < 3:
                        wandb_ori.append(wandb.Image(image_ori[i], caption=f"{total_steps}_{i}"))
                        wandb_garm.append(wandb.Image(image_garm[i], caption=f"{total_steps}_{i}"))
                        wandb_vton.append(wandb.Image(image_vton[i], caption=f"{total_steps}_{i}"))
                        
                    else: 
                        break
                if accelerator.is_main_process:

                    wandb.log({f"Garm IMGs": wandb_garm})
                    wandb.log({f"Original IMGs": wandb_ori})
                    wandb.log({f"Agnostic IMGs": wandb_vton})
                # print("denoised_latents shape:", denoised_latents.shape)
                decode_and_save_image(denoised_latents, './ootd_train_images', f"step{total_steps}")

            # Сохранение чекпоинтов каждые checkpoint_interval шагов
            if total_steps % checkpoint_interval == 0:
                state_dict_unet_vton = unet_vton.state_dict()
                for key in state_dict_unet_vton.keys():
                    state_dict_unet_vton[key] = state_dict_unet_vton[key].to('cpu')
                save_file(state_dict_unet_vton, f"./ootd_train_checkpoints/unet_vton-step{total_steps}.safetensors")

                state_dict_unet_garm = unet_garm.state_dict()
                for key in state_dict_unet_garm.keys():
                    state_dict_unet_garm[key] = state_dict_unet_garm[key].to('cpu')
                save_file(state_dict_unet_garm, f"./ootd_train_checkpoints/unet_garm-step{total_steps}.safetensors")
                print(f"Checkpoints saved at step {total_steps}")
accelerator.end_training()
# from safetensors.torch import save_file
# save_file(unet_vton.to('cpu').state_dict(), "./unet_vton.safetensors")
# save_file(unet_garm.to('cpu').state_dict(),"./unet_garm.safetensors")


