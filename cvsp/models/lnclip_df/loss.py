from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import logger

from .config import Loss as LossConfig


def alignment(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 2,
):
    """
    https://arxiv.org/pdf/2005.10242

    Label-aware Alignment loss.

    Calculates alignment for embeddings of samples with the SAME label
    within a batch, assuming embeddings are already unit-normalized.

    Args:
        embeddings: Tensor [N, D] - Batch of unit-normalized embeddings.
        labels: Tensor [N] - Corresponding labels.
        alpha: Power to raise squared distance (hyperparameter, default=2).

    Returns:
        Tensor: Label-aware Alignment loss (scalar). Returns 0 if no positive pairs.
    """
    assert embeddings.size(0) == labels.size(
        0
    ), "Embeddings and labels must have the same size."

    n_samples = embeddings.size(0)
    if n_samples < 2:
        return torch.tensor(0.0, device=embeddings.device)

    # Create a pairwise label comparison matrix (N x N), exclude self-pairs
    labels_equal_mask = (labels[:, None] == labels[None, :]).triu(diagonal=1)

    positive_indices = torch.nonzero(labels_equal_mask, as_tuple=False)
    if positive_indices.numel() == 0:
        return torch.tensor(0.0, device=embeddings.device)

    # Get embeddings of positive pairs
    x = embeddings[positive_indices[:, 0]]
    y = embeddings[positive_indices[:, 1]]

    # Calculate alignment loss
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(
    x: torch.Tensor,
    t: float = 2,
    clip_value: float = 1e-6,
):
    """
    https://arxiv.org/pdf/2005.10242

    Calculates the Uniformity loss.

    Args:
        x: [N, D] - Batch of feature embeddings.
        t: Temperature parameter (hyperparameter).

    Returns:
        Tensor: Uniformity loss value (scalar).
    """
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().clamp(min=clip_value).log()


@dataclass
class LossInputs:
    logits_labels: None | torch.Tensor = None
    labels: None | torch.Tensor = None
    embeddings: None | torch.Tensor = None


@dataclass
class LossOutputs:
    ce_labels: None | float = None
    bce_labels: None | float = None
    uniformity: None | float = None
    alignment_labels: None | float = None
    total: int | torch.Tensor = 0


class Loss(nn.Module):
    def __init__(self, loss_config: LossConfig):
        super().__init__()
        self.config = loss_config

    def forward(
        self,
        inputs: LossInputs,
    ) -> LossOutputs:
        loss_outputs = LossOutputs()

        if inputs.logits_labels is not None:
            if self.config.ce_labels:
                L = self.config.ce_labels * F.cross_entropy(
                    inputs.logits_labels,
                    inputs.labels,
                    label_smoothing=self.config.label_smoothing,
                )
                loss_outputs.ce_labels = L.item()
                loss_outputs.total += L

        if inputs.embeddings is not None:
            # L2 normalize embeddings
            # See 3.1  https://arxiv.org/pdf/2004.11362
            # embeddings = F.normalize(inputs.embeddings, p=2, dim=1)
            embeddings = inputs.embeddings

            # check that embeddings are normalized
            if not torch.allclose(
                embeddings.norm(p=2, dim=1),
                torch.ones(embeddings.size(0), device=embeddings.device),
            ):
                logger.print_warning_once("[yellow]Embeddings are not normalized")

            if inputs.labels is not None:
                if self.config.alignment_labels:
                    L = self.config.alignment_labels * alignment(
                        embeddings, inputs.labels
                    )
                    loss_outputs.alignment_labels = L.item()
                    loss_outputs.total += L

            if self.config.uniformity:
                L = self.config.uniformity * uniformity(embeddings)
                loss_outputs.uniformity = L.item()
                loss_outputs.total += L

        if isinstance(loss_outputs.total, int):
            logger.print_warning_once(
                "[yellow]Total loss is 0. Check if loss coefficients are set correctly."
            )

        if isinstance(loss_outputs.total, torch.Tensor) and loss_outputs.total.isnan():
            logger.print_warning("[yellow]Total loss is nan")
            loss_outputs.total = inputs.logits_labels.sum() * 0

        return loss_outputs

    def __call__(self, inputs: LossInputs) -> LossOutputs:
        return super().__call__(inputs)


if __name__ == "__main__":
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
    )
    embeddings /= embeddings.norm(p=2, dim=1, keepdim=True)

    labels = torch.tensor([0, 0, 0, 1, 1])

    print("Embeddings:")
    print(embeddings.numpy())

    print("\nLabels:")
    print(labels.numpy())

    alignment_loss = alignment(embeddings, labels, alpha=2)
    print("\nAlignment loss:", alignment_loss.item())

    uniformity_loss = uniformity(embeddings, t=2, clip_value=1e-6)
    print("Uniformity loss:", uniformity_loss.item())
