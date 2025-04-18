import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data import create_dataloader
from model import FTCLIP
import shutil
import datetime
from torch.utils.tensorboard import SummaryWriter

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
stamp = datetime.datetime.now().strftime("%m%d_%H%M")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune CLIP")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/data5/tangyinzhou/geotiffs/tier1/clean_images",
    )
    parser.add_argument(
        "--label_dir", type=str, default="/data5/tangyinzhou/geotiffs/tier1/labels"
    )
    parser.add_argument(
        "--sam_path",
        type=str,
        default="/data5/tangyinzhou/DisGrounding/model_weights/sam_vit_h_4b8939.pth",
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        default="/data5/tangyinzhou/DisGrounding/model_weights/CLIP",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--eval_every_n_epoch",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=f"/data5/tangyinzhou/DisGrounding/algos/finetune_clip/ckpts/ckpts_{stamp}",
    )
    parser.add_argument("--batch_size", type=int, default=20)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 构建 dataloader
    train_loader, val_loader = create_dataloader(
        args.image_dir, args.label_dir, args.batch_size
    )

    # 定义模型
    model = FTCLIP(sam_path=args.sam_path, clip_path=args.clip_path)
    model.cuda()

    # 仅微调特定参数
    for name, param in model.clip.named_parameters():
        if "bias" in name or "LayerNorm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for name, param in model.SAM_to_CLIP.named_parameters():
        param.requires_grad = True

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.clip.parameters()), lr=1e-4
    )
    loss_fn = nn.CrossEntropyLoss()
    epochs = args.epochs

    best_loss = float("inf")
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 初始化 TensorBoard 的 SummaryWriter
    log_dir = os.path.join(checkpoint_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        # train
        total_loss = 0
        model.train()
        for batch in tqdm(train_loader):
            pre_image, post_image, description = batch

            logits = model(
                pre_image=pre_image.cuda(),
                post_image=post_image.cuda(),
                description=description,
            )
            labels = torch.arange(len(logits)).to(
                model.clip.device
            )  # 假设每个图像和文本对是匹配的
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        print(f"【Train】:Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # eval
        total_eval_loss = 0
        model.eval()
        for batch in tqdm(val_loader):
            pre_image, post_image, description = batch
            logits = model(
                pre_image=pre_image.cuda(),
                post_image=post_image.cuda(),
                description=description,
            )
            labels = torch.arange(len(logits)).to(
                model.clip.device
            )  # 假设每个图像和文本对是匹配的
            loss = loss_fn(logits, labels)
            total_eval_loss += loss.item()
        avg_eval_loss = total_eval_loss / len(val_loader)
        print(f"Eval:Epoch {epoch+1}/{epochs}, Loss: {avg_eval_loss}")
        writer.add_scalar("Loss/eval", avg_eval_loss, epoch)

        # 保存检查点
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_eval_loss,
            },
            checkpoint_path,
        )

        # 保存最佳模型
        if avg_eval_loss < best_loss:
            best_loss = avg_eval_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
            shutil.copyfile(checkpoint_path, best_checkpoint_path)

    writer.close()


if __name__ == "__main__":
    main()
