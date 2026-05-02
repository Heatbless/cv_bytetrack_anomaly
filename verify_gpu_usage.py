#!/usr/bin/env python3
"""
Quick verification script to check if RT-DETR is using GPU
"""
import torch
from ultralytics import YOLO
import time

print("=" * 60)
print("GPU Verification for RT-DETR Inference")
print("=" * 60)

# Check CUDA
print(f"\nCUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load RT-DETR explicitly on GPU
print("\n--- Loading RT-DETR Model ---")
model = YOLO("rtdetr-l.pt")

# Check where model is loaded
print(f"\nModel device (after loading): {next(model.model.parameters()).device}")

# Dummy input for inference timing
import cv2
import numpy as np

test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

print("\n--- Timing Inference ---")
torch.cuda.reset_peak_memory_stats(0)
torch.cuda.synchronize()

start = time.time()
results = model(test_frame, verbose=False)
torch.cuda.synchronize()
elapsed = time.time() - start

peak_mem = torch.cuda.max_memory_allocated(0) / 1e9
current_mem = torch.cuda.memory_allocated(0) / 1e9

print(f"Inference time: {elapsed*1000:.2f} ms")
print(f"Peak VRAM used: {peak_mem:.2f} GB")
print(f"Current VRAM used: {current_mem:.2f} GB")

if elapsed < 0.1:  # GPU inference typically < 100ms for one frame
    print("\n✓ GPU inference confirmed (fast inference time)")
else:
    print(f"\n⚠️  Slow inference ({elapsed*1000:.0f}ms) - might be running on CPU")

# Check model parameters location
print(f"\nModel parameters device: {next(model.model.parameters()).device}")

print("\n" + "=" * 60)
