"""Utilities for temporal VAE traffic anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
ROI = tuple[int, int, int, int]


@dataclass
class DetectionResult:
    scores: np.ndarray
    raw_flags: np.ndarray
    smoothed_flags: np.ndarray


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray) -> None:
        self.sequences = torch.from_numpy(sequences).float()

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


class FrameEncoder(nn.Module):
    def __init__(self, emb_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrameDecoder(nn.Module):
    def __init__(self, emb_dim: int = 128, out_size: int = 64) -> None:
        super().__init__()
        self.out_size = out_size
        self.fc = nn.Linear(emb_dim, 64 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x).view(-1, 64, 4, 4)
        y = self.deconv(h)
        if y.shape[-1] != self.out_size or y.shape[-2] != self.out_size:
            y = torch.nn.functional.interpolate(
                y,
                size=(self.out_size, self.out_size),
                mode="bilinear",
                align_corners=False,
            )
        return y


class TemporalVAE(nn.Module):
    def __init__(self, seq_len: int = 8, latent_dim: int = 64, hidden_dim: int = 128, emb_dim: int = 128, img_size: int = 64) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.frame_encoder = FrameEncoder(emb_dim=emb_dim)
        self.temporal_encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc_dec_seed = nn.Linear(latent_dim, hidden_dim)
        self.temporal_decoder = nn.LSTM(hidden_dim, emb_dim, batch_first=True)
        self.frame_decoder = FrameDecoder(emb_dim=emb_dim, out_size=img_size)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.frame_encoder(x).view(b, t, -1)
        _, (h_n, _) = self.temporal_encoder(feats)
        h_last = h_n[-1]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        seed = self.fc_dec_seed(z)
        seq_seed = seed.unsqueeze(1).expand(b, self.seq_len, -1)
        dec_feats, _ = self.temporal_decoder(seq_seed)
        frames = self.frame_decoder(dec_feats.reshape(b * self.seq_len, -1))
        return frames.view(b, self.seq_len, 1, frames.shape[-2], frames.shape[-1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def _iter_videos(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        yield path
        return

    if not path.is_dir():
        return

    for p in sorted(path.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
            yield p


def _sanitize_roi(frame_shape: tuple[int, int, int], crop_roi: ROI | None) -> ROI | None:
    if crop_roi is None:
        return None

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in crop_roi]
    x1 = max(0, min(x1, w - 1))
    x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(1, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid crop ROI after clamping: {(x1, y1, x2, y2)} for frame size {(w, h)}")
    return x1, y1, x2, y2


def extract_frames(
    video_path: Path,
    img_size: int = 64,
    frame_step: int = 1,
    max_frames: int | None = None,
    crop_roi: ROI | None = None,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    idx = 0
    sanitized_roi: ROI | None = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if sanitized_roi is None and crop_roi is not None:
            sanitized_roi = _sanitize_roi(frame.shape, crop_roi)

        if idx % frame_step == 0:
            if sanitized_roi is not None:
                x1, y1, x2, y2 = sanitized_roi
                frame = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
            frames.append(gray.astype(np.float32) / 255.0)
            if max_frames is not None and len(frames) >= max_frames:
                break
        idx += 1

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    arr = np.stack(frames, axis=0)
    return arr[:, None, :, :]


def build_sequences(frames: np.ndarray, seq_len: int = 8, stride: int = 2) -> np.ndarray:
    sequences = []
    for i in range(0, max(0, len(frames) - seq_len + 1), stride):
        sequences.append(frames[i : i + seq_len])
    if not sequences:
        return np.empty((0, seq_len, 1, frames.shape[-2], frames.shape[-1]), dtype=np.float32)
    return np.stack(sequences, axis=0).astype(np.float32)


def collect_sequences(
    source: str | Path,
    img_size: int = 64,
    seq_len: int = 8,
    frame_step: int = 1,
    seq_stride: int = 2,
    max_frames_per_video: int | None = None,
    max_sequences: int | None = None,
    crop_roi: ROI | None = None,
) -> np.ndarray:
    source_path = Path(source)
    all_seqs = []
    for video in _iter_videos(source_path):
        frames = extract_frames(
            video,
            img_size=img_size,
            frame_step=frame_step,
            max_frames=max_frames_per_video,
            crop_roi=crop_roi,
        )
        seqs = build_sequences(frames, seq_len=seq_len, stride=seq_stride)
        if len(seqs) > 0:
            all_seqs.append(seqs)

    if not all_seqs:
        raise RuntimeError(
            "No usable video sequences found. Make sure source points to a video file or a folder with video clips."
        )

    merged = np.concatenate(all_seqs, axis=0)
    if max_sequences is not None and len(merged) > max_sequences:
        merged = merged[:max_sequences]
    return merged


def make_loader(sequences: np.ndarray, batch_size: int = 8, shuffle: bool = False) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(sequences).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _vae_loss_per_sample(x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kl_weight: float) -> torch.Tensor:
    recon_err = torch.mean((x - recon) ** 2, dim=(1, 2, 3, 4))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) / max(1, mu.shape[1])
    return recon_err + kl_weight * kl


def train_temporal_vae(model: TemporalVAE, loader: DataLoader, device: torch.device, epochs: int = 5, lr: float = 1e-3, kl_weight: float = 0.05) -> list[float]:
    model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for _ in range(max(1, epochs)):
        batch_losses = []
        for (x,) in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss = _vae_loss_per_sample(x, recon, mu, logvar, kl_weight).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            batch_losses.append(float(loss.detach().cpu()))

        losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)

    return losses


def sequence_anomaly_score(model: TemporalVAE, batch: torch.Tensor, device: torch.device, kl_weight: float = 0.05) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = batch.to(device)
        recon, mu, logvar = model(x)
        score = _vae_loss_per_sample(x, recon, mu, logvar, kl_weight)
        return score.detach().cpu().numpy()


def calibrate_threshold(model: TemporalVAE, normal_loader: DataLoader, device: torch.device, percentile: float = 99.5, kl_weight: float = 0.05) -> tuple[float, np.ndarray]:
    scores = []
    for (x,) in normal_loader:
        s = sequence_anomaly_score(model, x, device=device, kl_weight=kl_weight)
        scores.append(s)
    all_scores = np.concatenate(scores) if scores else np.array([], dtype=np.float32)
    if all_scores.size == 0:
        raise RuntimeError("No normal scores found during threshold calibration")
    threshold = float(np.percentile(all_scores, percentile))
    return threshold, all_scores


def smooth_flags(flags: np.ndarray, window: int = 3, min_ratio: float = 0.5) -> np.ndarray:
    flags = flags.astype(np.float32)
    if window <= 1:
        return flags.astype(bool)

    out = np.zeros_like(flags)
    half = window // 2
    for i in range(len(flags)):
        left = max(0, i - half)
        right = min(len(flags), i + half + 1)
        out[i] = 1.0 if np.mean(flags[left:right]) >= min_ratio else 0.0
    return out.astype(bool)


def detect_anomalies(model: TemporalVAE, test_loader: DataLoader, device: torch.device, threshold: float, kl_weight: float = 0.05, smooth_window: int = 3, smooth_min_ratio: float = 0.5) -> DetectionResult:
    scores = []
    for (x,) in test_loader:
        s = sequence_anomaly_score(model, x, device=device, kl_weight=kl_weight)
        scores.append(s)

    all_scores = np.concatenate(scores) if scores else np.array([], dtype=np.float32)
    if all_scores.size == 0:
        raise RuntimeError("No test scores generated for anomaly detection")

    raw = all_scores > threshold
    smoothed = smooth_flags(raw, window=smooth_window, min_ratio=smooth_min_ratio)
    return DetectionResult(scores=all_scores, raw_flags=raw, smoothed_flags=smoothed)


def anomaly_ratio(flags: np.ndarray) -> float:
    if flags.size == 0:
        return 0.0
    return float(np.mean(flags.astype(np.float32)))


def prepare_vae_context(
    normal_source: str | Path,
    test_source: str | Path | None = None,
    weights_path: str | Path = "temporal_vae_weights.pt",
    img_size: int = 64,
    seq_len: int = 8,
    frame_step: int = 1,
    seq_stride: int = 2,
    max_frames_per_video: int | None = 3000,
    max_sequences: int | None = 1500,
    batch_size: int = 8,
    latent_dim: int = 64,
    hidden_dim: int = 128,
    emb_dim: int = 128,
    train_if_missing: bool = True,
    train_epochs: int = 5,
    train_lr: float = 1e-3,
    train_kl_weight: float = 0.05,
    crop_roi: ROI | None = None,
    device: torch.device | None = None,
) -> tuple[TemporalVAE, torch.device, DataLoader, DataLoader]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normal_sequences = collect_sequences(
        normal_source,
        img_size=img_size,
        seq_len=seq_len,
        frame_step=frame_step,
        seq_stride=seq_stride,
        max_frames_per_video=max_frames_per_video,
        max_sequences=max_sequences,
        crop_roi=crop_roi,
    )

    if test_source is None:
        split = max(1, int(0.7 * len(normal_sequences)))
        split = min(split, len(normal_sequences) - 1) if len(normal_sequences) > 1 else len(normal_sequences)
        train_sequences = normal_sequences[:split]
        test_sequences = normal_sequences[split:] if len(normal_sequences) > 1 else normal_sequences.copy()
    else:
        train_sequences = normal_sequences
        test_sequences = collect_sequences(
            test_source,
            img_size=img_size,
            seq_len=seq_len,
            frame_step=frame_step,
            seq_stride=seq_stride,
            max_frames_per_video=max_frames_per_video,
            max_sequences=max_sequences,
            crop_roi=crop_roi,
        )

    normal_loader = make_loader(train_sequences, batch_size=batch_size, shuffle=True)
    test_loader = make_loader(test_sequences, batch_size=batch_size, shuffle=False)

    model = TemporalVAE(
        seq_len=seq_len,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        img_size=img_size,
    ).to(device)

    weights = Path(weights_path)
    if weights.exists():
        state = torch.load(weights, map_location=device)
        model.load_state_dict(state, strict=False)
    elif train_if_missing:
        train_temporal_vae(
            model,
            normal_loader,
            device=device,
            epochs=train_epochs,
            lr=train_lr,
            kl_weight=train_kl_weight,
        )
        torch.save(model.state_dict(), weights)
    else:
        raise RuntimeError(f"Weights not found at {weights} and train_if_missing=False")

    model.eval()
    return model, device, normal_loader, test_loader


# ============ PHASE 1: Synthetic Data Generation ============

def generate_synthetic_wrong_way_samples(
    normal_source: str | Path,
    output_dir: str | Path = "vae_wrong_way_samples",
) -> Path:
    """Generate synthetic wrong-way via temporal reverse + horizontal flip using ffmpeg."""
    import subprocess
    normal_path = Path(normal_source)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for side in ["left", "right"]:
        (output_path / side).mkdir(exist_ok=True)
    
    source_videos = []
    if normal_path.is_file() and normal_path.suffix.lower() in VIDEO_SUFFIXES:
        source_videos.append(normal_path)
    elif normal_path.is_dir():
        for p in sorted(normal_path.glob("**/*")):
            if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
                source_videos.append(p)
    
    if not source_videos:
        raise RuntimeError(f"No source videos found in {normal_source}")
    
    for src_video in source_videos:
        sides = ["left"] if "left" in src_video.parent.name.lower() else (["right"] if "right" in src_video.parent.name.lower() else ["left", "right"])
        for side in sides:
            out_file = output_path / side / src_video.name
            if out_file.exists():
                print(f"Skipping {out_file}")
                continue
            try:
                cmd = ["ffmpeg", "-i", str(src_video), "-vf", "reverse,hflip", "-c:v", "libx264", "-preset", "fast", "-crf", "23", str(out_file)]
                print(f"Generating: {side}/{src_video.name}")
                subprocess.run(cmd, check=True, capture_output=True)
            except Exception as e:
                print(f"Warning: {e}")
    return output_path


# ============ PHASE 2: GANomaly Architecture ============

class GANomaly(nn.Module):
    """Encoder-Generator-Discriminator for anomaly detection."""
    def __init__(self, seq_len: int = 8, latent_dim: int = 64, hidden_dim: int = 128, emb_dim: int = 128, img_size: int = 64):
        super().__init__()
        self.seq_len, self.latent_dim = seq_len, latent_dim
        self.frame_encoder = FrameEncoder(emb_dim=emb_dim)
        self.temporal_encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_gen = nn.Linear(latent_dim, hidden_dim)
        self.temporal_generator = nn.LSTM(hidden_dim, emb_dim, batch_first=True)
        self.frame_decoder = FrameDecoder(emb_dim=emb_dim, out_size=img_size)
        self.discriminator = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        feats = self.frame_encoder(x.view(b*t,c,h,w)).view(b,t,-1)
        _, (h_n, _) = self.temporal_encoder(feats)
        return self.fc_mu(h_n[-1])
    def generate(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        dec_feats, _ = self.temporal_generator(self.fc_gen(z).unsqueeze(1).expand(b, self.seq_len, -1))
        frames = self.frame_decoder(dec_feats.reshape(b*self.seq_len, -1))
        return frames.view(b, self.seq_len, 1, frames.shape[-2], frames.shape[-1])
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        return self.discriminator(x.view(b,c,t,h,w))
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.generate(z), z
    def anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        gen, _ = self.forward(x)
        return torch.mean((x - gen)**2, dim=(1,2,3,4)).detach().cpu().numpy()

def train_ganomaly(model: GANomaly, loader: DataLoader, device: torch.device, epochs: int = 10, lr_gen: float = 1e-4, lr_disc: float = 5e-5) -> dict:
    """Train GANomaly with alternating Generator/Discriminator updates."""
    model.to(device)
    encoder_params = (list(model.frame_encoder.parameters()) + list(model.temporal_encoder.parameters()) +
                      list(model.fc_mu.parameters()) + list(model.fc_gen.parameters()) + 
                      list(model.temporal_generator.parameters()) + list(model.frame_decoder.parameters()))
    optim_gen = torch.optim.Adam(encoder_params, lr=lr_gen)
    optim_disc = torch.optim.Adam(model.discriminator.parameters(), lr=lr_disc)
    criterion_bce, criterion_mse = nn.BCELoss(), nn.MSELoss()
    losses = {"gen": [], "disc": []}
    for epoch in range(epochs):
        gen_losses, disc_losses = [], []
        for (x,) in loader:
            x = x.to(device)
            b = x.shape[0]
            label_real, label_fake = torch.ones(b,1,device=device), torch.zeros(b,1,device=device)
            optim_disc.zero_grad(set_to_none=True)
            gen, _ = model(x)
            loss_disc = criterion_bce(model.discriminate(x), label_real) + criterion_bce(model.discriminate(gen.detach()), label_fake)
            loss_disc.backward()
            optim_disc.step()
            disc_losses.append(float(loss_disc.detach().cpu()))
            optim_gen.zero_grad(set_to_none=True)
            gen, _ = model(x)
            loss_gen = criterion_mse(gen, x) + criterion_bce(model.discriminate(gen), label_real)
            loss_gen.backward()
            optim_gen.step()
            gen_losses.append(float(loss_gen.detach().cpu()))
        losses["gen"].append(np.mean(gen_losses)); losses["disc"].append(np.mean(disc_losses))
        if (epoch+1) % max(1, epochs//5) == 0:
            print(f"GANomaly {epoch+1}/{epochs}: gen={losses['gen'][-1]:.6f}, disc={losses['disc'][-1]:.6f}")
    return losses

def ganomaly_anomaly_score(model: GANomaly, batch: torch.Tensor, device: torch.device) -> np.ndarray:
    """Compute anomaly score (reconstruction error) for batch."""
    model.eval()
    with torch.no_grad():
        return model.anomaly_score(batch.to(device))
