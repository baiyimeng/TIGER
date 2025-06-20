import argparse
import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, Adam
from model.id.rqvae import RQVAE
from time import time


class IDDataset(Dataset):
    def __init__(self, num_items):
        self.data = torch.arange(1, num_items + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RQVAE for Generative Recommendation"
    )
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--gpu_id", type=str, default="1")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr_id", type=float, default=1e-3)
    parser.add_argument("--wd_id", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--dataset", type=str, default="Beauty")
    parser.add_argument("--in_dim", type=int, default=4096)
    parser.add_argument("--codebook_size_list", type=list, default=[256] * 3)
    parser.add_argument("--codebook_dim", type=int, default=32)
    parser.add_argument("--layers", type=list, default=[2048, 1024, 512, 256, 128, 64])
    parser.add_argument("--dropout_prob", type=float, default=0)
    parser.add_argument("--kmeans_iters", type=int, default=10)
    parser.add_argument("--sk_epsilons", type=eval, default=[0.0, 0.0, 0.0, 0.0])
    parser.add_argument(
        "--id_mode", type=str, choices=["tiger", "letter"], default="tiger"
    )
    parser.add_argument("--cf_emb_path", type=str, default="")
    parser.add_argument("--alpha", type=float, default=2e-2)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--eval_epoch", type=int, default=100)
    return parser.parse_args()


def get_ckpt_path(args):
    os.makedirs("./ckpt", exist_ok=True)
    if args.id_mode == "tiger":
        return (
            f"./ckpt/rqvae-{args.dataset}-tiger-lr_{args.lr_id}-wd_{args.wd_id}.pth",
            f"./ckpt/rqvae-{args.dataset}-tiger-lr_{args.lr_id}-wd_{args.wd_id}-last.pth",
        )
    elif args.id_mode == "letter":
        return (
            f"./ckpt/rqvae-{args.dataset}-letter-alpha_{args.alpha}-beta_{args.beta}-lr_{args.lr_id}-wd_{args.wd_id}.pth",
            f"./ckpt/rqvae-{args.dataset}-letter-alpha_{args.alpha}-beta_{args.beta}-lr_{args.lr_id}-wd_{args.wd_id}-last.pth",
        )
    else:
        raise NotImplementedError(f"Unsupported id_mode: {args.id_mode}")


def initialize_model(args, device):
    model = RQVAE(
        in_dim=args.in_dim,
        codebook_size_list=args.codebook_size_list,
        codebook_dim=args.codebook_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        kmeans_iters=args.kmeans_iters,
        data_path=args.data_path,
        dataset=args.dataset,
        letter_mode=args.id_mode == "letter",
        cf_emb_path=args.cf_emb_path,
        alpha=args.alpha,
        beta=args.beta,
    ).to(device)
    model.build_indices()
    return model


def initialize_optimizer(model, args):
    if args.optimizer == "adamw":
        return AdamW(model.parameters(), lr=args.lr_id, weight_decay=args.wd_id)
    elif args.optimizer == "adam":
        return Adam(model.parameters(), lr=args.lr_id, weight_decay=args.wd_id)
    raise NotImplementedError(f"Unsupported optimizer: {args.optimizer}")


def main():
    args = parse_args()
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda")

    model = initialize_model(args, device)
    optimizer = initialize_optimizer(model, args)

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(model)

    print("\nInitializing codebooks with K-Means...")
    start = time()
    model.kmeans_init()
    model.build_indices()
    print(f"K-Means init completed in {time() - start:.2f}s")

    num_items = model.base_embedding.weight.size(0) - 1
    dataloader = DataLoader(
        IDDataset(num_items),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    best_collision_rate = 1.0
    ckpt_path, ckpt_last_path = get_ckpt_path(args)

    for epoch in range(args.max_epochs):
        model.train()
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, _, loss = model(batch)
            loss.backward()
            optimizer.step()

        model.build_indices()

        if (epoch + 1) % 1 == 0:
            print(f"[Epoch {epoch}] RQVAE Loss: {loss.item():.6f}")

        if (epoch + 1) % args.eval_epoch == 0:
            model.eval()
            with torch.no_grad():
                uniq_keys, coll_rate = model.get_collision_rate()
            print(
                f"Epoch {epoch} — Unique Keys: {uniq_keys}, Collision Rate: {coll_rate:.6f}"
            )
            if coll_rate < best_collision_rate:
                best_collision_rate = coll_rate
                torch.save(model, ckpt_path)

    # save the last epoch
    model.eval()
    model.build_indices()
    model.update_cluster_labels()
    torch.save(model, ckpt_last_path)

    # save the best-collision epoch
    model = torch.load(ckpt_path)
    model.eval()
    model.build_indices()
    torch.save(model, ckpt_path)

    print("Training completed.")
    print(f"Best Collision Rate: {best_collision_rate:.6f}")


if __name__ == "__main__":
    main()
