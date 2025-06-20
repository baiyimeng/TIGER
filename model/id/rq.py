import torch
import torch.nn as nn
from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size_list,
        codebook_dim,
        letter_mode=False,
        beta=0.01,
        sk_epsilons=None,
        sk_iters=50,
    ):
        """
        Residual Vector Quantization Layer
        Args:
            codebook_size_list (List[int]): List of codebook sizes for each residual level
            codebook_dim (int): Embedding dimension per codebook
            letter_mode (bool): Enable letter mode
            beta (float): Diversity loss weight
            sk_epsilons (List[float]): Sinkhorn entropies per codebook
            sk_iters (int): Sinkhorn iteration count
        """
        super().__init__()
        self.codebook_size_list = codebook_size_list
        self.codebook_dim = codebook_dim
        self.letter_mode = letter_mode
        self.sk_epsilons = sk_epsilons or [0.0] * len(codebook_size_list)
        self.sk_iters = sk_iters

        self.vq_layers = nn.ModuleList(
            [
                VectorQuantizer(
                    codebook_size=size,
                    codebook_dim=codebook_dim,
                    letter_mode=letter_mode,
                    beta=beta,
                    sk_epsilon=epsilon,
                    sk_iters=sk_iters,
                )
                for size, epsilon in zip(codebook_size_list, self.sk_epsilons)
            ]
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [B, D] or [B, T, D]

        Returns:
            hard_indices (Tensor): Quantization indices [..., L]
            soft_probs (Tensor): Soft assignment [..., L, K]
            quantized (Tensor): Quantized embedding [..., D]
            quant_losses (Tensor): VQ loss [..., L]
        """
        residual = x
        quantized = 0

        all_indices, all_probs, all_losses = [], [], []

        for quantizer in self.vq_layers:
            indices, probs, quants, losses = quantizer(residual)
            residual = residual - quants
            quantized = quantized + quants

            all_indices.append(indices)
            all_probs.append(probs)
            all_losses.append(losses)

        # Stack outputs along new axis (last axis = quantization level)
        hard_indices = torch.stack(all_indices, dim=-1)
        soft_probs = torch.stack(all_probs, dim=-1)
        quant_losses = torch.stack(all_losses, dim=-1)

        return hard_indices, soft_probs, quantized, quant_losses

    def kmeans_init(self, x, kmeans_iters=10):
        """
        Initializes each codebook using residual k-means
        """
        residual = x
        for i, quantizer in enumerate(self.vq_layers):
            quantized = quantizer.kmeans_init(
                residual, self.codebook_size_list[i], kmeans_iters
            )
            residual = residual - quantized

    def update_cluster_labels(
        self, n_clusters=10, size_min_limit=10, size_max_scale=6, kmeans_iters=10
    ):
        """
        Updates cluster labels used for diversity loss
        """
        for quantizer in self.vq_layers:
            quantizer.update_cluster_labels(
                n_clusters=n_clusters,
                size_min_limit=size_min_limit,
                size_max_scale=size_max_scale,
                kmeans_iters=kmeans_iters,
            )

    def get_codebooks(self):
        """
        Returns:
            Tensor: Codebooks stacked as [L, K, D]
        """
        return torch.stack(
            [quantizer.get_codebook() for quantizer in self.vq_layers], dim=0
        )
