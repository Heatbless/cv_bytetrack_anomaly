#!/usr/bin/env python3
"""
Standalone diagnostic script for anomaly detection threshold and input consistency analysis.
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ks_2samp

# Setup paths
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from vae_anomaly_module import (
    TemporalVAE, GANomaly, sequence_anomaly_score, ganomaly_anomaly_score,
    collect_sequences, make_loader
)

print("=" * 70)
print("DIAGNOSTIC: Score Distributions, Thresholds & Input Consistency")
print("=" * 70)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAE_IMG_SIZE = 64
VAE_SEQ_LEN = 16
VAE_LATENT_DIM = 64
VAE_HIDDEN_DIM = 128
VAE_EMB_DIM = 128
VAE_BATCH_SIZE = 8
KL_WEIGHT = 0.05

VIDEO_SOURCE = "CCTV_Sleman.mp4"
VAE_NORMAL_SOURCE_LEFT = "vae_split_samples/left"
VAE_NORMAL_SOURCE_RIGHT = "vae_split_samples/right"

# Load ROI from video
cap_meta = cv2.VideoCapture(VIDEO_SOURCE)
frame_w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_meta.release()

divider_x = 1021  # From notebook
VAE_LEFT_ROI = (0, 0, divider_x, frame_h)
VAE_RIGHT_ROI = (divider_x, 0, frame_w, frame_h)

print(f"\nLoading models from checkpoint files...")

# Load VAE models
vae_model_left = TemporalVAE(VAE_SEQ_LEN, VAE_LATENT_DIM, VAE_HIDDEN_DIM, VAE_EMB_DIM, VAE_IMG_SIZE).to(DEVICE)
vae_model_left.load_state_dict(torch.load("temporal_vae_left_weights.pt", map_location=DEVICE))
vae_model_left.eval()

vae_model_right = TemporalVAE(VAE_SEQ_LEN, VAE_LATENT_DIM, VAE_HIDDEN_DIM, VAE_EMB_DIM, VAE_IMG_SIZE).to(DEVICE)
vae_model_right.load_state_dict(torch.load("temporal_vae_right_weights.pt", map_location=DEVICE))
vae_model_right.eval()

# Load GANomaly models
gan_model_left = GANomaly(VAE_SEQ_LEN, VAE_LATENT_DIM, VAE_HIDDEN_DIM, VAE_EMB_DIM, VAE_IMG_SIZE).to(DEVICE)
gan_model_left.load_state_dict(torch.load("gan_anomaly_weights_left.pt", map_location=DEVICE))
gan_model_left.eval()

gan_model_right = GANomaly(VAE_SEQ_LEN, VAE_LATENT_DIM, VAE_HIDDEN_DIM, VAE_EMB_DIM, VAE_IMG_SIZE).to(DEVICE)
gan_model_right.load_state_dict(torch.load("gan_anomaly_weights_right.pt", map_location=DEVICE))
gan_model_right.eval()

print("✓ Models loaded successfully")

# Load comparison results
comparison_scores = pd.read_csv("vae_gan_comparison_scores.csv")
print(f"✓ Loaded comparison scores ({len(comparison_scores)} frames)")

# Extract scores from comparison results
inference_vae_left = comparison_scores['vae_left_score'].values
inference_vae_right = comparison_scores['vae_right_score'].values
inference_gan_left = comparison_scores['gan_left_score'].values
inference_gan_right = comparison_scores['gan_right_score'].values

# Collect training scores
print("\nCollecting training data scores...")
normal_seqs_left = collect_sequences(
    VAE_NORMAL_SOURCE_LEFT, img_size=VAE_IMG_SIZE, seq_len=VAE_SEQ_LEN,
    frame_step=2, seq_stride=2, max_frames_per_video=2500, max_sequences=1800, crop_roi=VAE_LEFT_ROI
)
normal_loader_left = make_loader(normal_seqs_left, batch_size=VAE_BATCH_SIZE)

normal_seqs_right = collect_sequences(
    VAE_NORMAL_SOURCE_RIGHT, img_size=VAE_IMG_SIZE, seq_len=VAE_SEQ_LEN,
    frame_step=2, seq_stride=2, max_frames_per_video=2500, max_sequences=1800, crop_roi=VAE_RIGHT_ROI
)
normal_loader_right = make_loader(normal_seqs_right, batch_size=VAE_BATCH_SIZE)

train_scores_vae_left = []
train_scores_vae_right = []
train_scores_gan_left = []
train_scores_gan_right = []

with torch.no_grad():
    for (x,) in normal_loader_left:
        x = x.to(DEVICE)
        s_vae = sequence_anomaly_score(vae_model_left, x, DEVICE, kl_weight=KL_WEIGHT)
        s_gan = ganomaly_anomaly_score(gan_model_left, x, DEVICE)
        train_scores_vae_left.extend(s_vae)
        train_scores_gan_left.extend(s_gan)
    
    for (x,) in normal_loader_right:
        x = x.to(DEVICE)
        s_vae = sequence_anomaly_score(vae_model_right, x, DEVICE, kl_weight=KL_WEIGHT)
        s_gan = ganomaly_anomaly_score(gan_model_right, x, DEVICE)
        train_scores_vae_right.extend(s_vae)
        train_scores_gan_right.extend(s_gan)

train_scores_vae_left = np.array(train_scores_vae_left)
train_scores_vae_right = np.array(train_scores_vae_right)
train_scores_gan_left = np.array(train_scores_gan_left)
train_scores_gan_right = np.array(train_scores_gan_right)

print("✓ Training scores collected")

# Calibrate thresholds
thr_vae_left = float(np.percentile(train_scores_vae_left, 98.5))
thr_vae_right = float(np.percentile(train_scores_vae_right, 98.5))
thr_gan_left = float(np.percentile(train_scores_gan_left, 98.5))
thr_gan_right = float(np.percentile(train_scores_gan_right, 98.5))

# --- PRINT ANALYSIS ---
print("\n" + "=" * 70)
print("1. SCORE DISTRIBUTIONS (Training Data)")
print("=" * 70)

print(f"\nVAE Left:")
print(f"  Training:   μ={train_scores_vae_left.mean():.6f}, σ={train_scores_vae_left.std():.6f}")
print(f"              range=[{train_scores_vae_left.min():.6f}, {train_scores_vae_left.max():.6f}]")
print(f"  Inference:  μ={inference_vae_left.mean():.6f}, σ={inference_vae_left.std():.6f}")
print(f"              range=[{inference_vae_left.min():.6f}, {inference_vae_left.max():.6f}]")
print(f"  Threshold:  {thr_vae_left:.6f} (98.5th percentile of training)")
print(f"  % Anomalous (inference): {100 * (inference_vae_left > thr_vae_left).mean():.1f}%")

print(f"\nVAE Right:")
print(f"  Training:   μ={train_scores_vae_right.mean():.6f}, σ={train_scores_vae_right.std():.6f}")
print(f"              range=[{train_scores_vae_right.min():.6f}, {train_scores_vae_right.max():.6f}]")
print(f"  Inference:  μ={inference_vae_right.mean():.6f}, σ={inference_vae_right.std():.6f}")
print(f"              range=[{inference_vae_right.min():.6f}, {inference_vae_right.max():.6f}]")
print(f"  Threshold:  {thr_vae_right:.6f} (98.5th percentile of training)")
print(f"  % Anomalous (inference): {100 * (inference_vae_right > thr_vae_right).mean():.1f}%")

print(f"\nGANomaly Left:")
print(f"  Training:   μ={train_scores_gan_left.mean():.6f}, σ={train_scores_gan_left.std():.6f}")
print(f"              range=[{train_scores_gan_left.min():.6f}, {train_scores_gan_left.max():.6f}]")
print(f"  Inference:  μ={inference_gan_left.mean():.6f}, σ={inference_gan_left.std():.6f}")
print(f"              range=[{inference_gan_left.min():.6f}, {inference_gan_left.max():.6f}]")
print(f"  Threshold:  {thr_gan_left:.6f} (98.5th percentile of training)")
print(f"  % Anomalous (inference): {100 * (inference_gan_left > thr_gan_left).mean():.1f}%")

print(f"\nGANomaly Right:")
print(f"  Training:   μ={train_scores_gan_right.mean():.6f}, σ={train_scores_gan_right.std():.6f}")
print(f"              range=[{train_scores_gan_right.min():.6f}, {train_scores_gan_right.max():.6f}]")
print(f"  Inference:  μ={inference_gan_right.mean():.6f}, σ={inference_gan_right.std():.6f}")
print(f"              range=[{inference_gan_right.min():.6f}, {inference_gan_right.max():.6f}]")
print(f"  Threshold:  {thr_gan_right:.6f} (98.5th percentile of training)")
print(f"  % Anomalous (inference): {100 * (inference_gan_right > thr_gan_right).mean():.1f}%")

# --- VISUALIZE DISTRIBUTIONS ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.hist(train_scores_vae_left, bins=30, alpha=0.6, label='Training', color='blue')
ax.hist(inference_vae_left, bins=30, alpha=0.6, label='Inference', color='orange')
ax.axvline(thr_vae_left, color='red', linestyle='--', linewidth=2, label=f'Threshold: {thr_vae_left:.4f}')
ax.set_xlabel('Anomaly Score')
ax.set_ylabel('Frequency')
ax.set_title('VAE Left: Score Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.hist(train_scores_vae_right, bins=30, alpha=0.6, label='Training', color='blue')
ax.hist(inference_vae_right, bins=30, alpha=0.6, label='Inference', color='orange')
ax.axvline(thr_vae_right, color='red', linestyle='--', linewidth=2, label=f'Threshold: {thr_vae_right:.4f}')
ax.set_xlabel('Anomaly Score')
ax.set_ylabel('Frequency')
ax.set_title('VAE Right: Score Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.hist(train_scores_gan_left, bins=30, alpha=0.6, label='Training', color='blue')
ax.hist(inference_gan_left, bins=30, alpha=0.6, label='Inference', color='orange')
ax.axvline(thr_gan_left, color='red', linestyle='--', linewidth=2, label=f'Threshold: {thr_gan_left:.4f}')
ax.set_xlabel('Anomaly Score')
ax.set_ylabel('Frequency')
ax.set_title('GANomaly Left: Score Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.hist(train_scores_gan_right, bins=30, alpha=0.6, label='Training', color='blue')
ax.hist(inference_gan_right, bins=30, alpha=0.6, label='Inference', color='orange')
ax.axvline(thr_gan_right, color='red', linestyle='--', linewidth=2, label=f'Threshold: {thr_gan_right:.4f}')
ax.set_xlabel('Anomaly Score')
ax.set_ylabel('Frequency')
ax.set_title('GANomaly Right: Score Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('threshold_diagnostics.png', dpi=100, bbox_inches='tight')
print("\n✓ Saved threshold_diagnostics.png")
plt.close()

# --- INPUT CONSISTENCY CHECK ---
print("\n" + "=" * 70)
print("2. INPUT CONSISTENCY CHECK")
print("=" * 70)

cap_check = cv2.VideoCapture(VIDEO_SOURCE)
frame_samples_left = []
frame_samples_right = []

for i in range(50):
    ret, frame = cap_check.read()
    if not ret:
        break
    
    x1_l, y1_l, x2_l, y2_l = VAE_LEFT_ROI
    x1_r, y1_r, x2_r, y2_r = VAE_RIGHT_ROI
    
    left_crop = frame[y1_l:y2_l, x1_l:x2_l].astype(np.float32) / 255.0
    right_crop = frame[y1_r:y2_r, x1_r:x2_r].astype(np.float32) / 255.0
    
    left_gray = cv2.cvtColor((left_crop * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    right_gray = cv2.cvtColor((right_crop * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    left_resized = cv2.resize(left_gray, (VAE_IMG_SIZE, VAE_IMG_SIZE))
    right_resized = cv2.resize(right_gray, (VAE_IMG_SIZE, VAE_IMG_SIZE))
    
    frame_samples_left.append(left_resized)
    frame_samples_right.append(right_resized)

cap_check.release()

frame_samples_left = np.array(frame_samples_left)
frame_samples_right = np.array(frame_samples_right)

print(f"\nLeft ROI:  {VAE_LEFT_ROI}")
print(f"  Resized to: {VAE_IMG_SIZE}x{VAE_IMG_SIZE}")
print(f"  Sample stats (n=50 frames):")
print(f"    Mean: {frame_samples_left.mean():.4f}, Std: {frame_samples_left.std():.4f}")
print(f"    Range: [{frame_samples_left.min():.4f}, {frame_samples_left.max():.4f}]")

print(f"\nRight ROI: {VAE_RIGHT_ROI}")
print(f"  Resized to: {VAE_IMG_SIZE}x{VAE_IMG_SIZE}")
print(f"  Sample stats (n=50 frames):")
print(f"    Mean: {frame_samples_right.mean():.4f}, Std: {frame_samples_right.std():.4f}")
print(f"    Range: [{frame_samples_right.min():.4f}, {frame_samples_right.max():.4f}]")

# Plot samples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(5):
    axes[0, i].imshow(frame_samples_left[i * 10], cmap='gray')
    axes[0, i].set_title(f'Left Frame {i*10}')
    axes[0, i].axis('off')
    
    axes[1, i].imshow(frame_samples_right[i * 10], cmap='gray')
    axes[1, i].set_title(f'Right Frame {i*10}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('input_consistency_samples.png', dpi=100, bbox_inches='tight')
print("✓ Saved input_consistency_samples.png")
plt.close()

# --- DOMAIN SHIFT ANALYSIS ---
print("\n" + "=" * 70)
print("3. DOMAIN SHIFT ANALYSIS (Kolmogorov-Smirnov Test)")
print("=" * 70)

ks_vae_l, p_vae_l = ks_2samp(train_scores_vae_left, inference_vae_left)
ks_vae_r, p_vae_r = ks_2samp(train_scores_vae_right, inference_vae_right)
ks_gan_l, p_gan_l = ks_2samp(train_scores_gan_left, inference_gan_left)
ks_gan_r, p_gan_r = ks_2samp(train_scores_gan_right, inference_gan_right)

print(f"\nKS test (p < 0.05 indicates significant domain shift):")
print(f"  VAE Left:   KS={ks_vae_l:.4f}, p={p_vae_l:.4f} {'⚠️ DOMAIN SHIFT' if p_vae_l < 0.05 else '✓ Similar'}")
print(f"  VAE Right:  KS={ks_vae_r:.4f}, p={p_vae_r:.4f} {'⚠️ DOMAIN SHIFT' if p_vae_r < 0.05 else '✓ Similar'}")
print(f"  GAN Left:   KS={ks_gan_l:.4f}, p={p_gan_l:.4f} {'⚠️ DOMAIN SHIFT' if p_gan_l < 0.05 else '✓ Similar'}")
print(f"  GAN Right:  KS={ks_gan_r:.4f}, p={p_gan_r:.4f} {'⚠️ DOMAIN SHIFT' if p_gan_r < 0.05 else '✓ Similar'}")

# --- ROOT CAUSE SUMMARY ---
print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)

vae_left_fire_rate = (inference_vae_left > thr_vae_left).mean()
vae_right_fire_rate = (inference_vae_right > thr_vae_right).mean()
gan_left_fire_rate = (inference_gan_left > thr_gan_left).mean()
gan_right_fire_rate = (inference_gan_right > thr_gan_right).mean()

print("\n📊 ANOMALY FIRE RATES:")
print(f"  VAE Left:   {100*vae_left_fire_rate:.1f}%  {'❌ TOO HIGH' if vae_left_fire_rate > 0.5 else '✓ OK'}")
print(f"  VAE Right:  {100*vae_right_fire_rate:.1f}%  {'✓ OK' if vae_right_fire_rate > 0.01 else '❌ TOO LOW'}")
print(f"  GAN Left:   {100*gan_left_fire_rate:.1f}%  {'❌ TOO HIGH' if gan_left_fire_rate > 0.5 else '✓ OK'}")
print(f"  GAN Right:  {100*gan_right_fire_rate:.1f}%  {'✓ OK' if gan_right_fire_rate > 0.01 else '❌ TOO LOW'}")

print("\n🔍 ROOT CAUSES:")
brightness_diff = abs(frame_samples_left.mean() - frame_samples_right.mean())
contrast_diff = abs(frame_samples_left.std() - frame_samples_right.std())

if vae_left_fire_rate > 0.9:
    print("  ⚠️  LEFT side oversensitive (VAE >90% anomaly rate)")
if vae_right_fire_rate < 0.01:
    print("  ⚠️  RIGHT side insensitive (VAE <1% anomaly rate)")
if gan_left_fire_rate > 0.9:
    print("  ⚠️  LEFT side oversensitive (GAN >90% anomaly rate)")
if gan_right_fire_rate < 0.01:
    print("  ⚠️  RIGHT side insensitive (GAN <1% anomaly rate)")

if brightness_diff > 0.1:
    print(f"  ⚠️  Brightness mismatch: left={frame_samples_left.mean():.3f}, right={frame_samples_right.mean():.3f} (Δ={brightness_diff:.3f})")
else:
    print(f"  ✓ Brightness consistent: left={frame_samples_left.mean():.3f}, right={frame_samples_right.mean():.3f}")

if contrast_diff > 0.05:
    print(f"  ⚠️  Contrast mismatch: left_σ={frame_samples_left.std():.3f}, right_σ={frame_samples_right.std():.3f} (Δ={contrast_diff:.3f})")
else:
    print(f"  ✓ Contrast consistent: left_σ={frame_samples_left.std():.3f}, right_σ={frame_samples_right.std():.3f}")

if p_vae_l < 0.05 or p_vae_r < 0.05 or p_gan_l < 0.05 or p_gan_r < 0.05:
    print("  ⚠️  Domain shift detected: inference distribution differs from training")

print("\n✓ Diagnostic report complete!")
