#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recalibrate anomaly detection thresholds on actual inference data.
Uses first 20% of video as calibration set, applies to remaining 80%.
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import deque

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from vae_anomaly_module import (
    TemporalVAE, GANomaly, sequence_anomaly_score, ganomaly_anomaly_score
)

print("=" * 70)
print("RECALIBRATION: Two-Pass Threshold Estimation")
print("=" * 70)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAE_IMG_SIZE = 64
VAE_SEQ_LEN = 16
VAE_LATENT_DIM = 64
VAE_HIDDEN_DIM = 128
VAE_EMB_DIM = 128
KL_WEIGHT = 0.05

VIDEO_SOURCE = "CCTV_Sleman.mp4"
DIVIDER_X = 1021

VAE_LEFT_ROI = (0, 0, DIVIDER_X, 1080)
VAE_RIGHT_ROI = (DIVIDER_X, 0, 1920, 1080)

print(f"\nLoading models...")

# Load models
vae_model_left = TemporalVAE(VAE_SEQ_LEN, VAE_LATENT_DIM, VAE_HIDDEN_DIM, VAE_EMB_DIM, VAE_IMG_SIZE).to(DEVICE)
vae_model_left.load_state_dict(torch.load("temporal_vae_left_weights.pt", map_location=DEVICE))
vae_model_left.eval()

vae_model_right = TemporalVAE(VAE_SEQ_LEN, VAE_LATENT_DIM, VAE_HIDDEN_DIM, VAE_EMB_DIM, VAE_IMG_SIZE).to(DEVICE)
vae_model_right.load_state_dict(torch.load("temporal_vae_right_weights.pt", map_location=DEVICE))
vae_model_right.eval()

gan_model_left = GANomaly(VAE_SEQ_LEN, VAE_LATENT_DIM, VAE_HIDDEN_DIM, VAE_EMB_DIM, VAE_IMG_SIZE).to(DEVICE)
gan_model_left.load_state_dict(torch.load("gan_anomaly_weights_left.pt", map_location=DEVICE))
gan_model_left.eval()

gan_model_right = GANomaly(VAE_SEQ_LEN, VAE_LATENT_DIM, VAE_HIDDEN_DIM, VAE_EMB_DIM, VAE_IMG_SIZE).to(DEVICE)
gan_model_right.load_state_dict(torch.load("gan_anomaly_weights_right.pt", map_location=DEVICE))
gan_model_right.eval()

print("[OK] Models loaded")

# Get video info
cap = cv2.VideoCapture(VIDEO_SOURCE)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

cal_frames = int(0.20 * total_frames)  # First 20% for calibration
print(f"\nVideo: {total_frames} frames @ {fps} FPS")
print(f"Calibration set: frames 1-{cal_frames} (first 20%)")
print(f"Test set: frames {cal_frames+1}-{total_frames} (remaining 80%)")

# --- PASS 1: Calibration on first 20% ---
print(f"\nPass 1: Collecting scores from calibration set...")

cap = cv2.VideoCapture(VIDEO_SOURCE)
cal_scores_vae_left = []
cal_scores_vae_right = []
cal_scores_gan_left = []
cal_scores_gan_right = []

seq_buffer_left = deque(maxlen=VAE_SEQ_LEN)
seq_buffer_right = deque(maxlen=VAE_SEQ_LEN)
frame_counter = 0

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret or frame_counter >= cal_frames:
            break
        
        frame_counter += 1
        
        # Extract ROIs
        x1_l, y1_l, x2_l, y2_l = VAE_LEFT_ROI
        x1_r, y1_r, x2_r, y2_r = VAE_RIGHT_ROI
        
        left_crop = frame[y1_l:y2_l, x1_l:x2_l].astype(np.float32) / 255.0
        right_crop = frame[y1_r:y2_r, x1_r:x2_r].astype(np.float32) / 255.0
        
        left_gray = cv2.cvtColor((left_crop * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        right_gray = cv2.cvtColor((right_crop * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        left_resized = cv2.resize(left_gray, (VAE_IMG_SIZE, VAE_IMG_SIZE))
        right_resized = cv2.resize(right_gray, (VAE_IMG_SIZE, VAE_IMG_SIZE))
        
        seq_buffer_left.append(left_resized[np.newaxis, :, :])
        seq_buffer_right.append(right_resized[np.newaxis, :, :])
        
        # Score when buffer full
        if len(seq_buffer_left) == VAE_SEQ_LEN:
            seq_left_np = np.array(list(seq_buffer_left), dtype=np.float32)[np.newaxis, ...]
            seq_right_np = np.array(list(seq_buffer_right), dtype=np.float32)[np.newaxis, ...]
            
            seq_left_t = torch.from_numpy(seq_left_np).float().to(DEVICE)
            seq_right_t = torch.from_numpy(seq_right_np).float().to(DEVICE)
            
            vae_model_left.eval()
            vae_model_right.eval()
            vae_recon_l, vae_mu_l, vae_logvar_l = vae_model_left(seq_left_t)
            vae_recon_r, vae_mu_r, vae_logvar_r = vae_model_right(seq_right_t)
            
            vae_score_l = torch.mean((seq_left_t - vae_recon_l) ** 2).item()
            vae_score_r = torch.mean((seq_right_t - vae_recon_r) ** 2).item()
            
            kl_l = -0.5 * torch.mean(1 + vae_logvar_l - vae_mu_l**2 - vae_logvar_l.exp())
            kl_r = -0.5 * torch.mean(1 + vae_logvar_r - vae_mu_r**2 - vae_logvar_r.exp())
            
            vae_score_l += KL_WEIGHT * kl_l.item()
            vae_score_r += KL_WEIGHT * kl_r.item()
            
            gan_score_l = ganomaly_anomaly_score(gan_model_left, seq_left_t, DEVICE)[0]
            gan_score_r = ganomaly_anomaly_score(gan_model_right, seq_right_t, DEVICE)[0]
            
            cal_scores_vae_left.append(vae_score_l)
            cal_scores_vae_right.append(vae_score_r)
            cal_scores_gan_left.append(gan_score_l)
            cal_scores_gan_right.append(gan_score_r)
        
        if frame_counter % max(1, cal_frames // 10) == 0:
            print(f"  {100.0 * frame_counter / cal_frames:.0f}% complete")

cap.release()

# Compute calibration thresholds
cal_scores_vae_left = np.array(cal_scores_vae_left)
cal_scores_vae_right = np.array(cal_scores_vae_right)
cal_scores_gan_left = np.array(cal_scores_gan_left)
cal_scores_gan_right = np.array(cal_scores_gan_right)

thr_vae_left = float(np.percentile(cal_scores_vae_left, 98.5))
thr_vae_right = float(np.percentile(cal_scores_vae_right, 98.5))
thr_gan_left = float(np.percentile(cal_scores_gan_left, 98.5))
thr_gan_right = float(np.percentile(cal_scores_gan_right, 98.5))

print(f"\n[OK] Calibration thresholds (98.5th percentile of first 20%):")
print(f"  VAE Left:   {thr_vae_left:.6f}  (μ={cal_scores_vae_left.mean():.6f}, σ={cal_scores_vae_left.std():.6f})")
print(f"  VAE Right:  {thr_vae_right:.6f}  (μ={cal_scores_vae_right.mean():.6f}, σ={cal_scores_vae_right.std():.6f})")
print(f"  GAN Left:   {thr_gan_left:.6f}  (μ={cal_scores_gan_left.mean():.6f}, σ={cal_scores_gan_left.std():.6f})")
print(f"  GAN Right:  {thr_gan_right:.6f}  (μ={cal_scores_gan_right.mean():.6f}, σ={cal_scores_gan_right.std():.6f})")

# --- PASS 2: Detection on remaining 80% with recalibrated thresholds ---
print(f"\nPass 2: Running inference on test set with recalibrated thresholds...")

cap = cv2.VideoCapture(VIDEO_SOURCE)
frame_counter = 0
seq_buffer_left = deque(maxlen=VAE_SEQ_LEN)
seq_buffer_right = deque(maxlen=VAE_SEQ_LEN)
alert_rows = []

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("vae_gan_recalibrated_anomalies.mp4", fourcc, fps, (frame_w, frame_h))

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        
        # Skip first 20%
        if frame_counter <= cal_frames:
            continue
        
        # Extract ROIs
        x1_l, y1_l, x2_l, y2_l = VAE_LEFT_ROI
        x1_r, y1_r, x2_r, y2_r = VAE_RIGHT_ROI
        
        left_crop = frame[y1_l:y2_l, x1_l:x2_l].astype(np.float32) / 255.0
        right_crop = frame[y1_r:y2_r, x1_r:x2_r].astype(np.float32) / 255.0
        
        left_gray = cv2.cvtColor((left_crop * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        right_gray = cv2.cvtColor((right_crop * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        left_resized = cv2.resize(left_gray, (VAE_IMG_SIZE, VAE_IMG_SIZE))
        right_resized = cv2.resize(right_gray, (VAE_IMG_SIZE, VAE_IMG_SIZE))
        
        seq_buffer_left.append(left_resized[np.newaxis, :, :])
        seq_buffer_right.append(right_resized[np.newaxis, :, :])
        
        # Score and visualize
        if len(seq_buffer_left) == VAE_SEQ_LEN:
            seq_left_np = np.array(list(seq_buffer_left), dtype=np.float32)[np.newaxis, ...]
            seq_right_np = np.array(list(seq_buffer_right), dtype=np.float32)[np.newaxis, ...]
            
            seq_left_t = torch.from_numpy(seq_left_np).float().to(DEVICE)
            seq_right_t = torch.from_numpy(seq_right_np).float().to(DEVICE)
            
            vae_model_left.eval()
            vae_model_right.eval()
            vae_recon_l, vae_mu_l, vae_logvar_l = vae_model_left(seq_left_t)
            vae_recon_r, vae_mu_r, vae_logvar_r = vae_model_right(seq_right_t)
            
            vae_score_l = torch.mean((seq_left_t - vae_recon_l) ** 2).item()
            vae_score_r = torch.mean((seq_right_t - vae_recon_r) ** 2).item()
            
            kl_l = -0.5 * torch.mean(1 + vae_logvar_l - vae_mu_l**2 - vae_logvar_l.exp())
            kl_r = -0.5 * torch.mean(1 + vae_logvar_r - vae_mu_r**2 - vae_logvar_r.exp())
            
            vae_score_l += KL_WEIGHT * kl_l.item()
            vae_score_r += KL_WEIGHT * kl_r.item()
            
            gan_score_l = ganomaly_anomaly_score(gan_model_left, seq_left_t, DEVICE)[0]
            gan_score_r = ganomaly_anomaly_score(gan_model_right, seq_right_t, DEVICE)[0]
            
            vae_left_anom = vae_score_l > thr_vae_left
            vae_right_anom = vae_score_r > thr_vae_right
            gan_left_anom = gan_score_l > thr_gan_left
            gan_right_anom = gan_score_r > thr_gan_right
            
            any_anom = vae_left_anom or vae_right_anom or gan_left_anom or gan_right_anom
            
            alert_rows.append({
                'frame': frame_counter,
                'time_sec': frame_counter / fps,
                'vae_left_score': vae_score_l,
                'vae_right_score': vae_score_r,
                'gan_left_score': gan_score_l,
                'gan_right_score': gan_score_r,
                'vae_left_anom': int(vae_left_anom),
                'vae_right_anom': int(vae_right_anom),
                'gan_left_anom': int(gan_left_anom),
                'gan_right_anom': int(gan_right_anom),
                'any_anom': int(any_anom),
            })
            
            # Visualization
            color_vae_l = (0, 0, 255) if vae_left_anom else (0, 255, 0)
            color_vae_r = (0, 0, 255) if vae_right_anom else (0, 255, 0)
            color_gan_l = (0, 165, 255) if gan_left_anom else (200, 200, 200)
            color_gan_r = (0, 165, 255) if gan_right_anom else (200, 200, 200)
            
            cv2.rectangle(frame, (x1_l, y1_l), (x2_l, y2_l), color_vae_l, 2)
            cv2.rectangle(frame, (x1_r, y1_r), (x2_r, y2_r), color_vae_r, 2)
            
            text_l = f"VAE: {'ANOM' if vae_left_anom else 'OK'} {vae_score_l:.3f}"
            text_r = f"VAE: {'ANOM' if vae_right_anom else 'OK'} {vae_score_r:.3f}"
            cv2.putText(frame, text_l, (x1_l + 5, y1_l - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_vae_l, 1)
            cv2.putText(frame, text_r, (x1_r + 5, y1_r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_vae_r, 1)
            
            cv2.putText(frame, f"GAN: {'ANOM' if gan_left_anom else 'OK'} {gan_score_l:.3f}", 
                       (x1_l + 5, y1_l + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_gan_l, 1)
            cv2.putText(frame, f"GAN: {'ANOM' if gan_right_anom else 'OK'} {gan_score_r:.3f}", 
                       (x1_r + 5, y1_r + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_gan_r, 1)
        
        out.write(frame)
        
        if frame_counter % max(1, (total_frames - cal_frames) // 10) == 0:
            pct = 100.0 * (frame_counter - cal_frames) / (total_frames - cal_frames)
            print(f"  {pct:.0f}% complete")

cap.release()
out.release()

# Export results
alert_df = pd.DataFrame(alert_rows)
alert_df.to_csv("vae_gan_recalibrated_scores.csv", index=False)

print(f"\n[OK] Saved recalibrated inference video to vae_gan_recalibrated_anomalies.mp4")
print(f"[OK] Saved scores to vae_gan_recalibrated_scores.csv")

# Print statistics
if len(alert_rows) > 0:
    n_any_anom = sum(1 for r in alert_rows if r['any_anom'])
    n_vae_any = sum(1 for r in alert_rows if r['vae_left_anom'] or r['vae_right_anom'])
    n_gan_any = sum(1 for r in alert_rows if r['gan_left_anom'] or r['gan_right_anom'])
    
    print(f"\n--- Test Set Statistics (frames {cal_frames+1}-{total_frames}) ---")
    print(f"VAE anomalies: {n_vae_any}/{len(alert_rows)} ({100*n_vae_any/len(alert_rows):.1f}%)")
    print(f"GAN anomalies: {n_gan_any}/{len(alert_rows)} ({100*n_gan_any/len(alert_rows):.1f}%)")
    print(f"Either model:  {n_any_anom}/{len(alert_rows)} ({100*n_any_anom/len(alert_rows):.1f}%)")
    
    print(f"\n--- Score Ranges (Test Set) ---")
    print(f"VAE Left:   [{alert_df['vae_left_score'].min():.6f}, {alert_df['vae_left_score'].max():.6f}]  (threshold: {thr_vae_left:.6f})")
    print(f"VAE Right:  [{alert_df['vae_right_score'].min():.6f}, {alert_df['vae_right_score'].max():.6f}]  (threshold: {thr_vae_right:.6f})")
    print(f"GAN Left:   [{alert_df['gan_left_score'].min():.6f}, {alert_df['gan_left_score'].max():.6f}]  (threshold: {thr_gan_left:.6f})")
    print(f"GAN Right:  [{alert_df['gan_right_score'].min():.6f}, {alert_df['gan_right_score'].max():.6f}]  (threshold: {thr_gan_right:.6f})")

print("\n[OK] Recalibration complete!")
