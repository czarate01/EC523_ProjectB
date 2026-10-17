"""Untrained interface example. Replace this model for the project.

The public contract is intentionally small: build once, then transform one
[1,T] waveform without filenames, targets, SNRs, transcripts, or metadata.
"""

import torch
from torch import nn


class DemoSpectralGate(nn.Module):
    def __init__(self, sample_rate: int = 16_000, n_fft: int = 512, hop: int = 128):
        super().__init__()
        if sample_rate != 16_000:
            raise ValueError("the course interface is fixed at 16 kHz")
        self.n_fft = n_fft
        self.hop = hop
        # A learnable per-frequency sigmoid, initialized close to identity.
        self.mask_logits = nn.Parameter(torch.full((n_fft // 2 + 1,), 5.0))
        self.register_buffer("window", torch.hann_window(n_fft), persistent=True)

    def enhance(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate != 16_000 or waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError("expected one [1,T] waveform at 16 kHz")
        length = waveform.shape[-1]
        padded = waveform
        if length < self.n_fft:
            padded = torch.nn.functional.pad(waveform, (0, self.n_fft - length))
        spectrum = torch.stft(
            padded, self.n_fft, self.hop, window=self.window,
            return_complex=True, center=True,
        )
        mask = self.mask_logits.sigmoid().view(1, -1, 1)
        enhanced = torch.istft(
            spectrum * mask, self.n_fft, self.hop, window=self.window,
            length=padded.shape[-1], center=True,
        )
        return enhanced[..., :length]


def build_enhancer(sample_rate: int) -> nn.Module:
    return DemoSpectralGate(sample_rate=sample_rate)

