"""Small four-scale CNN with segmentation, row locations and presence heads."""
import torch
from torch import nn
from torch.nn import functional as F
from . import CLASSES


class Block(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                         nn.GroupNorm(4, outgoing), nn.SiLU(inplace=True),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                         nn.GroupNorm(4, outgoing), nn.SiLU(inplace=True))


class FishingVision(nn.Module):
    def __init__(self, channels=(16, 32, 64, 96)):
        super().__init__()
        a, b, c, d = channels
        self.channels = tuple(channels)
        self.e0, self.e1, self.e2, self.e3 = Block(3, a), Block(a, b), Block(b, c), Block(c, d)
        self.d2, self.d1, self.d0 = Block(d+c, c), Block(c+b, b), Block(b+a, a)
        self.seg = nn.Conv2d(a, len(CLASSES), 1)
        self.rows = nn.Conv2d(a, 4, 1)
        self.presence = nn.Linear(d, 4)

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(F.avg_pool2d(e0, 2))
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        up = lambda v, skip: F.interpolate(v, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat((up(e3, e2), e2), 1))
        d1 = self.d1(torch.cat((up(d2, e1), e1), 1))
        d0 = self.d0(torch.cat((up(d1, e0), e0), 1))
        return {"seg": self.seg(d0), "rows": torch.logsumexp(self.rows(d0), dim=-1),
                "presence": self.presence(e3.mean(dim=(-2, -1)))}


def objective(pred, batch):
    target = batch["mask"]
    valid = target >= 0
    weights = pred["seg"].new_tensor([0.3, 3, 3, 1, 1, 1.5], dtype=torch.float32)
    ce = F.cross_entropy(pred["seg"].float(), target, weight=weights,
                         ignore_index=-1, reduction="none")
    seg_loss = (ce * valid).sum() / valid.sum().clamp_min(1)
    probability = pred["seg"].float().softmax(1)
    onehot = F.one_hot(target.clamp_min(0), len(CLASSES)).permute(0, 3, 1, 2).float()
    v = valid[:, None]
    intersection = (probability * onehot * v).sum(dim=(0, 2, 3))
    denominator = ((probability + onehot) * v).sum(dim=(0, 2, 3))
    dice = (1 - (2*intersection + 1) / (denominator + 1))[1:].mean()
    y = torch.arange(pred["rows"].shape[-1], device=target.device).float()
    gaussian = torch.exp(-0.5 * ((y[None, None] - batch["rows"][:, :, None]) / 1.8)**2)
    gaussian = gaussian / gaussian.sum(-1, keepdim=True).clamp_min(1e-8)
    row_ce = -(gaussian * pred["rows"].float().log_softmax(-1)).sum(-1)
    row_loss = (row_ce * batch["row_valid"]).sum() / batch["row_valid"].sum().clamp_min(1)
    presence = F.binary_cross_entropy_with_logits(pred["presence"].float(), batch["presence"])
    total = seg_loss + 0.5*dice + 1.5*row_loss + presence
    return total, {"seg_ce": seg_loss.detach(), "dice": dice.detach(),
                   "row_ce": row_loss.detach(), "presence_bce": presence.detach()}
