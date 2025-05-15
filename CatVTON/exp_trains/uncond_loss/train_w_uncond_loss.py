import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import argparse
from trainer_w_uncond_loss import CatVTONTrainerWUncondLoss 
import wandb
from datasets.vitonhd import VITONDataset
from torch.utils import data

import wandb

def main():
    parser = argparse.ArgumentParser(description="Training CatVTON")

    # Основные пути
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to VTON-HD dataset")
    parser.add_argument("--base_ckpt", type=str, required=True, help="Path to base checkpoint")
    # runwayml/stable-diffusion-inpainting
    parser.add_argument("--attn_ckpt", type=str, default="no", help="no is training from scratch, else use attn chkp path")
    
    
    parser.add_argument("--device", type=str, default="cpu", required=True)

    # Гиперпараметры
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_train_steps", type=int, default=60000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--compile", type=bool, default=False, help="Using torch compile for unet and vae")


    parser.add_argument("--cfg_dropout_prob", type=float, default=0.1)
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--save_chkp_every", type=int, default=2000)
    parser.add_argument("--save_attn_ckpt_path", type=str, required=True)

    # Логгинг
    parser.add_argument("--wandb_project", type=str, default="catvton-1gpu-training")
    parser.add_argument("--wandb_run_name", type=str, default="catvton-run-true-cfg")
    parser.add_argument("--wandb_api_key", type=str, default="None", help="None as str if no wandb logs")
    # 79824aa6b958aeebce669281f175fe198eb060dd
    args = parser.parse_args()

    # Логинимся в wandb
    wandb_logger = None
    if args.wandb_api_key != "None":
        wandb.login(key=args.wandb_api_key)
        wandb_logger = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "batch_size": args.batch_size,
                "max_train_steps": args.max_train_steps,
                "learning_rate": args.learning_rate,
                "cfg_dropout_prob": args.cfg_dropout_prob
            }
        )

    # Загружаем датасет
    train_dataset = VITONDataset(dataset_dir=args.dataset_dir)
    train_dataloader = data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=8, pin_memory=True, drop_last=True
        )

    trainer = CatVTONTrainerWUncondLoss(
        base_ckpt=args.base_ckpt,
        train_dataset=train_dataset,
        train_dataloader=train_dataloader,
        device=args.device,
        compile=args.compile,
        attn_ckpt=args.attn_ckpt,
        batch_size=args.batch_size,
        max_train_steps=args.max_train_steps,
        cfg_dropout_prob=args.cfg_dropout_prob,
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        log_step=args.log_step,
        wandb_logger=wandb_logger,
        save_chkp_every = args.save_chkp_every,
        save_attn_ckpt_pth= args.save_attn_ckpt_path
    )

    trainer.train()


if __name__ == "__main__":
    main()
