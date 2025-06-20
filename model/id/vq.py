import torch
import torch.nn as nn
import torch.nn.functional as F
from k_means_constrained import KMeansConstrained
from model.id.layers import sinkhorn_algorithm


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size=256,
        codebook_dim=32,
        mu=0.25,
        beta=0.01,
        letter_mode=False,
        sk_epsilon=0.0,
        sk_iters=50,
    ):
        """
        Vector Quantization with optional Sinkhorn-based soft assignment and diversity loss
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.mu = mu
        self.beta = beta
        self.letter_mode = letter_mode
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        self.cluster_label = list(range(codebook_size))
        self.cluster_labeled = False

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.embedding.weight)

    # === Forward Path ===
    def forward(self, x):
        is_seq = x.dim() == 3
        if is_seq:
            B, T, D = x.shape
            x = x.view(-1, D)

        d2 = self._compute_distances(x, self.embedding.weight)
        probs, indices = self._assign_code(d2)

        quants_hard = self.embedding(indices)
        commitment = F.mse_loss(quants_hard.detach(), x, reduction="none").mean(-1)
        codebook = F.mse_loss(quants_hard, x.detach(), reduction="none").mean(-1)

        # Optional: diversity loss
        if self.letter_mode and self.cluster_labeled and self.beta > 0:
            div_loss = self._diversity_loss(quants_hard, indices, self.cluster_label)
            loss = codebook + self.mu * commitment + self.beta * div_loss
        else:
            loss = codebook + self.mu * commitment

        # Straight-through estimator
        quants = (quants_hard - x).detach() + x

        if is_seq:
            loss = loss.view(B, T)
            indices = indices.view(B, T)
            probs = probs.view(B, T, self.codebook_size)
            quants = quants.view(B, T, self.codebook_dim)

        return indices, probs, quants, loss

    # === Quantization Utilities ===
    def _compute_distances(self, x, codebook):
        return (
            x.pow(2).sum(dim=-1, keepdim=True)
            + codebook.pow(2).sum(dim=-1, keepdim=True).t()
            - 2 * x @ codebook.t()
        )

    def _assign_code(self, d2):
        if self.sk_epsilon > 0.0:
            d2_normed = self._normalize_distances(d2).double()
            Q = sinkhorn_algorithm(d2_normed, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print("Warning: Sinkhorn returned NaN or Inf.")
            return Q, Q.argmax(dim=-1)
        else:
            logits = -d2
            probs = F.softmax(logits, dim=-1)
            return probs, logits.argmax(dim=-1)

    def _normalize_distances(self, d2):
        max_d, min_d = d2.max(), d2.min()
        mid = (max_d + min_d) / 2
        scale = max_d - mid + 1e-5
        return (d2 - mid) / scale

    # === Clustering & KMeans ===
    def kmeans_init(self, data, n_clusters, kmeans_iters=10):
        assert n_clusters == self.codebook_size
        centers, _ = self._constrained_kmeans(data, n_clusters, kmeans_iters)
        self.embedding.weight.data.copy_(centers)

        d2 = self._compute_distances(data, self.embedding.weight)
        if self.letter_mode and self.sk_epsilon > 0.0:
            d2_normed = self._normalize_distances(d2).double()
            Q = sinkhorn_algorithm(d2_normed, self.sk_epsilon, self.sk_iters)
            return self.embedding(Q.argmax(dim=-1))
        else:
            return self.embedding(d2.argmin(dim=-1))

    def update_cluster_labels(
        self, n_clusters=10, size_min_limit=10, size_max_scale=6, kmeans_iters=10
    ):
        centers, labels = self._constrained_kmeans(
            self.embedding.weight,
            n_clusters,
            size_min_limit,
            size_max_scale,
            kmeans_iters,
        )
        self.cluster_label = labels
        self.cluster_labeled = True

    def _constrained_kmeans(
        self, data, n_clusters, size_min_limit=50, size_max_scale=4, kmeans_iters=10
    ):
        x = data.detach().cpu().numpy()
        size_min = min(len(x) // (n_clusters * 2), size_min_limit)
        size_max = size_min * size_max_scale

        clf = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=size_max,
            max_iter=kmeans_iters,
            n_init=10,
            n_jobs=10,
            verbose=False,
        )
        try:
            clf.fit(x)
        except Exception:
            print("KMeans failed, retrying...")
            clf.fit(x)
        return torch.from_numpy(clf.cluster_centers_), clf.labels_.tolist()

    # === Diversity Loss ===
    def _diversity_loss(self, x_q, indices, labels):
        cluster_ids = [labels[idx.item()] for idx in indices]
        cluster_map = {i: [] for i in set(labels)}
        for i, lbl in enumerate(labels):
            cluster_map[lbl].append(i)

        all_candidates, ptr = [], []
        for i, cid in enumerate(cluster_ids):
            choices = [j for j in cluster_map[cid] if j != indices[i].item()]
            choices = choices or [indices[i].item()]
            start = len(all_candidates)
            all_candidates.extend(choices)
            end = len(all_candidates)
            ptr.append((start, end))

        all_candidates = torch.tensor(all_candidates, device=x_q.device)
        ptr = torch.tensor(ptr, device=x_q.device)
        rand_offsets = (
            torch.rand(len(ptr), device=x_q.device) * (ptr[:, 1] - ptr[:, 0]).float()
        ).long()
        sampled_neg = all_candidates[ptr[:, 0] + rand_offsets]

        logits = torch.matmul(x_q, self.embedding.weight.t())
        logits.scatter_(1, indices.unsqueeze(1), float("-inf"))
        return F.cross_entropy(logits, sampled_neg, reduction="none")

    def get_codebook(self):
        return self.embedding.weight
