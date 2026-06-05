import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------


def squared_euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Pairwise squared Euclidean distance.

    Args:
        x: [N, D]
        y: [M, D]

    Returns:
        dist: [N, M]
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    if y.size(1) != d:
        raise ValueError(f"Embedding dimensions do not match: {x.size()} vs {y.size()}")

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(dim=2)


class ConvBNLeakyReLU(nn.Module):
    """Conv1D + BatchNorm + LeakyReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding_mode: str = "valid",
        negative_slope: float = 0.01,
    ):
        super().__init__()

        if padding_mode not in {"valid", "same"}:
            raise ValueError("padding_mode must be either 'valid' or 'same'")

        if padding_mode == "valid":
            padding = 0
        else:
            # Same padding for stride 1. For even kernel sizes, PyTorch's symmetric
            # padding is not perfectly same-length, so we handle it manually.
            padding = 0

        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def _same_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Manual same padding along the temporal dimension."""
        if self.padding_mode != "same":
            return x

        length = x.size(-1)
        out_len = math.ceil(length / self.stride)
        pad_needed = max(0, (out_len - 1) * self.stride + self.kernel_size - length)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        return F.pad(x, (pad_left, pad_right), mode="constant", value=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._same_pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class FCBlock(nn.Module):
    """FC + BatchNorm + LeakyReLU block."""

    def __init__(self, in_features: int, out_features: int, negative_slope: float = 0.01):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# -----------------------------------------------------------------------------
# Random Dimension Permutation
# -----------------------------------------------------------------------------


class RandomDimensionPermutation(nn.Module):
    """
    Random Dimension Permutation (RDP) used in TapNet.

    Given m input dimensions, TapNet forms g random groups. The group size is

        phi = floor(m * alpha / g)

    where alpha is the scale factor.

    This module stores fixed random groups as a buffer, so the same groups are
    used throughout training/evaluation for reproducibility.
    """

    def __init__(
        self,
        in_channels: int,
        num_groups: int = 3,
        rdp_scale: float = 1.5,
        seed: int = 0,
    ):
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive")

        if num_groups <= 0:
            raise ValueError("num_groups must be positive")

        self.in_channels = int(in_channels)
        self.num_groups = int(max(1, min(num_groups, in_channels)))
        self.rdp_scale = float(rdp_scale)

        group_size = int(math.floor(self.in_channels * self.rdp_scale / self.num_groups))
        group_size = max(1, min(group_size, self.in_channels))
        self.group_size = group_size

        generator = torch.Generator()
        generator.manual_seed(seed)

        groups = []
        for _ in range(self.num_groups):
            perm = torch.randperm(self.in_channels, generator=generator)
            groups.append(perm[:self.group_size])

        group_indices = torch.stack(groups, dim=0).long()  # [G, phi]
        self.register_buffer("group_indices", group_indices, persistent=True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, C, L]

        Returns:
            list of grouped tensors, each with shape [B, phi, L]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input [B, C, L], got {x.shape}")

        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Input channel mismatch: expected {self.in_channels}, got {x.size(1)}"
            )

        grouped = []
        for g in range(self.num_groups):
            idx = self.group_indices[g]
            grouped.append(x.index_select(dim=1, index=idx))
        return grouped


class TapNetEncoder(nn.Module):
    """
    TapNet multivariate time-series encoder.

    This module returns the low-dimensional embedding f_theta(X).
    It does not classify by itself unless used with the optional episode head
    in TapNet_FSL.forward_episode().
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: Optional[int] = None,
        num_groups: int = 3,
        rdp_scale: float = 1.5,
        rdp_seed: int = 0,
        lstm_hidden_dim: int = 128,
        conv_filters: Tuple[int, int, int] = (256, 256, 128),
        conv_kernels: Tuple[int, int, int] = (8, 5, 3),
        fc_hidden_dim: int = 500,
        embedding_dim: int = 300,
        conv_padding: str = "valid",
        normalize_embedding: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.seq_len = seq_len
        self.normalize_embedding = normalize_embedding

        # For univariate data, RDP degenerates into one single group.
        if self.in_channels == 1:
            num_groups = 1
            rdp_scale = 1.0

        self.rdp = RandomDimensionPermutation(
            in_channels=self.in_channels,
            num_groups=num_groups,
            rdp_scale=rdp_scale,
            seed=rdp_seed,
        )

        self.num_groups = self.rdp.num_groups
        self.group_size = self.rdp.group_size

        c1, c2, c3 = conv_filters
        k1, k2, k3 = conv_kernels

        # LSTM branch over raw MTS: [B, C, L] -> [B, L, C]
        self.lstm = nn.LSTM(
            input_size=self.in_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # Conv branch:
        # - First conv is group-specific.
        # - Second and third conv are shared across groups.
        self.group_conv1 = nn.ModuleList([
            ConvBNLeakyReLU(
                in_channels=self.group_size,
                out_channels=c1,
                kernel_size=k1,
                padding_mode=conv_padding,
            )
            for _ in range(self.num_groups)
        ])

        self.shared_conv2 = ConvBNLeakyReLU(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k2,
            padding_mode=conv_padding,
        )

        self.shared_conv3 = ConvBNLeakyReLU(
            in_channels=c2,
            out_channels=c3,
            kernel_size=k3,
            padding_mode=conv_padding,
        )

        combo_dim = lstm_hidden_dim + c3 * self.num_groups

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc1 = FCBlock(combo_dim, fc_hidden_dim)
        self.fc2 = FCBlock(fc_hidden_dim, embedding_dim)

        self.out_put_channel_number = embedding_dim  # compatible naming with your OS-CNN style
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Args:
            x: [B, C, L]

        Returns:
            embedding: [B, embedding_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if x.dim() != 3:
            raise ValueError(f"Expected input shape [B, C, L], got {x.shape}")

        if debug:
            print("=" * 70)
            print(f"TapNet input: {x.shape}")
            print(f"RDP groups: {self.rdp.group_indices.detach().cpu().tolist()}")

        # ---------------------------------------------------------
        # LSTM branch
        # ---------------------------------------------------------
        lstm_in = x.transpose(1, 2)  # [B, L, C]
        lstm_out, _ = self.lstm(lstm_in)  # [B, L, H]
        lstm_feat = lstm_out.mean(dim=1)  # global average pooling over time

        # ---------------------------------------------------------
        # Conv branches with RDP groups
        # ---------------------------------------------------------
        grouped_x = self.rdp(x)
        conv_feats = []

        for g, x_g in enumerate(grouped_x):
            h = self.group_conv1[g](x_g)
            h = self.shared_conv2(h)
            h = self.shared_conv3(h)
            h = h.mean(dim=-1)  # global average pooling over time
            conv_feats.append(h)

        conv_feat = torch.cat(conv_feats, dim=1)  # [B, 128 * G]

        # ---------------------------------------------------------
        # Concatenate LSTM and Conv features, then project
        # ---------------------------------------------------------
        combo = torch.cat([lstm_feat, conv_feat], dim=1)
        combo = self.dropout(combo)

        emb = self.fc1(combo)
        emb = self.dropout(emb)
        emb = self.fc2(emb)

        if self.normalize_embedding:
            emb = F.normalize(emb, p=2, dim=1)

        if debug:
            print(f"LSTM feature: {lstm_feat.shape}")
            print(f"Conv feature: {conv_feat.shape}")
            print(f"TapNet embedding: {emb.shape}")
            print("=" * 70)

        return emb


# -----------------------------------------------------------------------------
# Optional attentional prototype head
# -----------------------------------------------------------------------------


class AttentionalPrototypeHead(nn.Module):
    """
    TapNet-style attentional prototype learning.

    For each local episode class k, the class prototype is computed by

        c_k = sum_i A_{k,i} H_{k,i}
        A_k = softmax(w_k^T tanh(V_k H_k^T))

    This implementation uses class-specific attention parameters for local
    episode classes 0, 1, ..., max_episode_classes - 1.

    In the existing OSSE-LSTM training loop we do not need this head because
    our code computes mean prototypes outside the model. However, this head is
    included so that the implementation remains faithful to TapNet.
    """

    def __init__(
        self,
        embedding_dim: int = 300,
        attention_hidden_dim: int = 128,
        max_episode_classes: int = 50,
        normalize_prototypes: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.max_episode_classes = max_episode_classes
        self.normalize_prototypes = normalize_prototypes

        self.V = nn.Parameter(
            torch.empty(max_episode_classes, attention_hidden_dim, embedding_dim)
        )
        self.w = nn.Parameter(
            torch.empty(max_episode_classes, attention_hidden_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.w.unsqueeze(-1))

    def compute_prototypes_from_grouped_indices(
        self,
        embeddings: torch.Tensor,
        grouped_s_idxs: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            embeddings: [B, D]
            grouped_s_idxs: list of tensors, each containing support indices for one class

        Returns:
            prototypes: [N_way, D]
            attention_weights: list of tensors, each [K]
        """
        device = embeddings.device
        n_way = len(grouped_s_idxs)

        if n_way > self.max_episode_classes:
            raise ValueError(
                f"n_way={n_way} exceeds max_episode_classes={self.max_episode_classes}. "
                f"Increase max_episode_classes when constructing TapNet_FSL."
            )

        prototypes = []
        attention_weights = []

        for local_cls_idx, idxs in enumerate(grouped_s_idxs):
            idxs = idxs.to(device)
            H_k = embeddings[idxs]  # [K, D]

            V_k = self.V[local_cls_idx]  # [U, D]
            w_k = self.w[local_cls_idx]  # [U]

            # scores: [K]
            hidden = torch.tanh(torch.matmul(H_k, V_k.t()))  # [K, U]
            scores = torch.matmul(hidden, w_k)  # [K]
            A_k = torch.softmax(scores, dim=0)  # [K]

            c_k = torch.sum(A_k.unsqueeze(1) * H_k, dim=0)  # [D]

            if self.normalize_prototypes:
                c_k = F.normalize(c_k.unsqueeze(0), p=2, dim=1).squeeze(0)

            prototypes.append(c_k)
            attention_weights.append(A_k)

        prototypes = torch.stack(prototypes, dim=0)
        return prototypes, attention_weights


# -----------------------------------------------------------------------------
# Full TapNet-FSL wrapper
# -----------------------------------------------------------------------------


class TapNet_FSL(nn.Module):
    """
    TapNet model adapted for your few-shot prototypical scenario.

    By default:
        model(x) returns embeddings, so your existing train.py can compute
        standard mean prototypes exactly as it does for OSSE-LSTM.

    Optional:
        model.forward_episode(...) computes TapNet-style attentional prototypes
        and returns logits directly.
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: Optional[int] = None,
        num_groups: int = 3,
        rdp_scale: float = 1.5,
        rdp_seed: int = 0,
        lstm_hidden_dim: int = 128,
        conv_filters: Tuple[int, int, int] = (256, 256, 128),
        conv_kernels: Tuple[int, int, int] = (8, 5, 3),
        fc_hidden_dim: int = 500,
        embedding_dim: int = 300,
        conv_padding: str = "valid",
        normalize_embedding: bool = False,
        dropout: float = 0.0,
        use_attention_prototypes: bool = True,
        attention_hidden_dim: int = 128,
        max_episode_classes: int = 50,
        normalize_prototypes: bool = False,
    ):
        super().__init__()

        self.encoder = TapNetEncoder(
            in_channels=in_channels,
            seq_len=seq_len,
            num_groups=num_groups,
            rdp_scale=rdp_scale,
            rdp_seed=rdp_seed,
            lstm_hidden_dim=lstm_hidden_dim,
            conv_filters=conv_filters,
            conv_kernels=conv_kernels,
            fc_hidden_dim=fc_hidden_dim,
            embedding_dim=embedding_dim,
            conv_padding=conv_padding,
            normalize_embedding=normalize_embedding,
            dropout=dropout,
        )

        self.use_attention_prototypes = use_attention_prototypes
        self.embedding_dim = embedding_dim
        self.out_put_channel_number = embedding_dim

        if use_attention_prototypes:
            self.prototype_head = AttentionalPrototypeHead(
                embedding_dim=embedding_dim,
                attention_hidden_dim=attention_hidden_dim,
                max_episode_classes=max_episode_classes,
                normalize_prototypes=normalize_prototypes,
            )
        else:
            self.prototype_head = None

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """Return embeddings, compatible with your current ProtoNet loop."""
        return self.encoder(x, debug=debug)

    def forward_episode(
        self,
        x: torch.Tensor,
        grouped_s_idxs: Sequence[torch.Tensor],
        query_idxs: Union[torch.Tensor, Sequence[int]],
        use_attention: Optional[bool] = None,
    ):
        """
        Optional TapNet-style episodic forward.

        Args:
            x: [B, C, L]
            grouped_s_idxs: list of support indices for each episode class
            query_idxs: indices of query samples
            use_attention:
                True  -> attentional prototypes
                False -> mean prototypes
                None  -> use self.use_attention_prototypes

        Returns:
            logits: [num_query, n_way]
            prototypes: [n_way, D]
            embeddings: [B, D]
            extra: dict with optional attention weights
        """
        if use_attention is None:
            use_attention = self.use_attention_prototypes

        embeddings = self.forward(x)
        device = embeddings.device

        if isinstance(query_idxs, torch.Tensor):
            query_idxs = query_idxs.to(device)
        else:
            query_idxs = torch.LongTensor(query_idxs).to(device)

        if use_attention:
            if self.prototype_head is None:
                raise RuntimeError("prototype_head is disabled. Set use_attention_prototypes=True.")
            prototypes, attention_weights = self.prototype_head.compute_prototypes_from_grouped_indices(
                embeddings,
                grouped_s_idxs,
            )
            extra = {"attention_weights": attention_weights}
        else:
            proto_lst = []
            for idxs in grouped_s_idxs:
                idxs = idxs.to(device)
                proto_lst.append(embeddings[idxs].mean(dim=0))
            prototypes = torch.stack(proto_lst, dim=0)
            extra = {"attention_weights": None}

        logits = -squared_euclidean_dist(embeddings[query_idxs], prototypes)
        return logits, prototypes, embeddings, extra


# Alias for convenient imports
TapNet = TapNet_FSL


if __name__ == "__main__":
    torch.manual_seed(0)

    # Example: 2-way, 2-shot, 2-query, multivariate with 6 channels
    B, C, L = 8, 6, 100
    x = torch.randn(B, C, L)

    model = TapNet_FSL(
        in_channels=C,
        seq_len=L,
        num_groups=3,
        rdp_scale=1.5,
        embedding_dim=300,
        normalize_embedding=True,
        conv_padding="valid",
        max_episode_classes=10,
    )

    emb = model(x, debug=True)
    print("Embedding:", emb.shape)

    grouped_s_idxs = [torch.LongTensor([0, 1]), torch.LongTensor([4, 5])]
    query_idxs = torch.LongTensor([2, 3, 6, 7])

    logits, proto, embs, extra = model.forward_episode(
        x,
        grouped_s_idxs=grouped_s_idxs,
        query_idxs=query_idxs,
        use_attention=True,
    )

    print("Logits:", logits.shape)
    print("Prototypes:", proto.shape)
