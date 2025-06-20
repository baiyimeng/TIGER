import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from .layers import MLPLayers
from .rq import ResidualVectorQuantizer
import os
from collections import defaultdict


class RQVAE(nn.Module):
    def __init__(
        self,
        in_dim=4096,
        codebook_size_list=[256] * 4,
        codebook_dim=32,
        layers=[2048, 1024, 512, 256, 128, 64],
        activation="relu",
        dropout_prob=0.0,
        bn=False,
        quant_loss_weight=1.0,
        kmeans_iters=10,
        data_path="",
        dataset="",
        letter_mode=False,
        cf_emb_path="",
        alpha=0.01,
        beta=0.0001,
        sk_epsilons=[0.0, 0.0, 0.0, 0.0],
        sk_iters=50,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.codebook_size_list = codebook_size_list
        self.codebook_dim = codebook_dim
        self.quant_loss_weight = quant_loss_weight
        self.kmeans_iters = kmeans_iters
        self.letter_mode = letter_mode
        self.alpha = alpha
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        self.encode_layer_dims = [in_dim] + layers + [codebook_dim]
        self.encoder = MLPLayers(self.encode_layer_dims, dropout_prob, activation, bn)

        self.rq = ResidualVectorQuantizer(
            codebook_size_list,
            codebook_dim,
            letter_mode=letter_mode,
            beta=beta,
            sk_epsilons=sk_epsilons,
            sk_iters=sk_iters,
        )

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(self.decode_layer_dims, dropout_prob, activation, bn)

        load_path = os.path.join(data_path, dataset, dataset + ".emb-llama-td.npy")
        load_weights = np.load(load_path)
        assert load_weights.shape[1] == in_dim

        padding_row = np.zeros((1, in_dim), dtype=load_weights.dtype)
        weights_with_padding = np.vstack([padding_row, load_weights])

        self.base_embedding = nn.Embedding(weights_with_padding.shape[0], in_dim)
        self.base_embedding.weight.data.copy_(
            torch.tensor(weights_with_padding, dtype=torch.float32)
        )
        self.base_embedding.weight.requires_grad = False

        # Load cf_embedding only once and keep as tensor on CPU
        if self.letter_mode and cf_emb_path:
            cf_emb_np = np.load(cf_emb_path)
            self.cf_embedding = torch.from_numpy(cf_emb_np).float()
        else:
            self.cf_embedding = None

        # Cache for generation mapping
        self.all_indices = None  # store the indices of all items without unique suffix
        self.full_all_indices = None  # store the full indices with unique suffix
        self.code2item = None  # store the mapping from code(str) to items
        self.unique_indices = None  # store the unique suffix indices

    def kmeans_init(self):
        with torch.no_grad():
            emb_weight = self.base_embedding.weight.data[1:]  # exclude padding
            features = self.encoder(emb_weight)
            self.rq.kmeans_init(features, self.kmeans_iters)

    def forward(self, x):
        assert x.dtype in (torch.int16, torch.int32, torch.int64)
        x_mask = (x != 0).float()

        x_in = self.base_embedding(x)
        encoded = self.encoder(x_in)
        hard_indices, soft_probs, x_q, quant_losses = self.rq(encoded)

        decoded = self.decoder(x_q)

        quant_loss = (quant_losses.mean(dim=-1) * x_mask).sum() / (x_mask.sum() + 1e-8)
        recon_loss = ((decoded - x_in).pow(2).mean(dim=-1) * x_mask).sum() / (
            x_mask.sum() + 1e-8
        )

        if self.letter_mode and self.alpha > 0 and self.cf_embedding is not None:
            # Prepare cf_embedding batch
            cf_embed = torch.cat(
                [torch.zeros(1, self.codebook_dim), self.cf_embedding], dim=0
            )
            cf_embed_batch = cf_embed[x.cpu()].to(x.device)

            cf_loss = self.get_cf_loss(
                x_q.view(-1, self.codebook_dim),
                cf_embed_batch.view(-1, self.codebook_dim),
            )
            total_loss = (
                recon_loss + self.quant_loss_weight * quant_loss + self.alpha * cf_loss
            )
        else:
            total_loss = recon_loss + self.quant_loss_weight * quant_loss

        return hard_indices, soft_probs, total_loss

    def get_cf_loss(self, quantized_rep, encoded_rep):
        batch_size = quantized_rep.size(0)
        labels = torch.arange(batch_size, device=quantized_rep.device)
        similarities = quantized_rep @ encoded_rep.t()
        return F.cross_entropy(similarities, labels)

    def update_cluster_labels(
        self, n_clusters=10, size_min_limit=10, size_max_scale=6, kmeans_iters=10
    ):
        if self.letter_mode:
            with torch.no_grad():
                self.rq.update_cluster_labels(
                    n_clusters=n_clusters,
                    size_min_limit=size_min_limit,
                    size_max_scale=size_max_scale,
                    kmeans_iters=kmeans_iters,
                )

    def build_indices(self):
        with torch.no_grad():
            emb_weight = self.base_embedding.weight.data
            encoded = self.encoder(emb_weight)
            hard_indices, _, _, _ = self.rq(encoded)
            self.all_indices = hard_indices.cpu().numpy()

        self.full_all_indices = [[-1] * (len(self.codebook_size_list) + 1)]
        self.code2item = defaultdict(list)
        self.unique_indices = [0]
        for i, c in enumerate(self.all_indices[1:]):
            str_id = ",".join(map(str, c.tolist()))
            index = self.all_indices[i].tolist()
            self.code2item[str_id].append(i + 1)
            unique_index = len(self.code2item[str_id]) - 1
            self.full_all_indices.append(index + [unique_index])
            self.unique_indices.append(unique_index)

    def get_collision_rate(self):
        total_items = len(self.all_indices) - 1
        unique_items = len(self.code2item)
        collision_rate = 1 - (unique_items / total_items) if total_items > 0 else 0.0

        return unique_items, collision_rate
