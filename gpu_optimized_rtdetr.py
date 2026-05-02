#!/usr/bin/env python3
"""
GPU-enabled RT-DETR inference wrapper
"""
import torch
from ultralytics import YOLO
import time
import cv2
import numpy as np

print("=" * 60)
print("GPU-Optimized RT-DETR Inference")
print("=" * 60)

print(f"\nCUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load model and EXPLICITLY move to GPU
print("\n--- Loading RT-DETR on GPU ---")
model = YOLO("rtdetr-l.pt")
model.to('cuda')  # <-- THE FIX

print(f"Model device: {next(model.model.parameters()).device}")

# Warm-up inference (first call is slower)
print("\nWarm-up inference...")
test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
_ = model(test_frame, verbose=False)

# Timed inference
print("\n--- Benchmarking Inference ---")
times = []
for i in range(5):
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
    
    start = time.time()
    results = model(test_frame, verbose=False)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    times.append(elapsed)
    peak_mem = torch.cuda.max_memory_allocated(0) / 1e9
    print(f"  Frame {i+1}: {elapsed*1000:.2f} ms (peak VRAM: {peak_mem:.2f} GB)")

print(f"\nAverage: {np.mean(times)*1000:.2f} ms")
print(f"Speedup vs CPU: ~{9114/np.mean(times):.0f}x faster")

if np.mean(times) < 0.2:
    print("\n✓ GPU inference confirmed!")
else:
    print("\n⚠️ Still slow - check model loading")

print("\n" + "=" * 60)
print("To enable GPU in YOLO.ipynb Cell 1, add after model initialization:")
print("  model.to('cuda')")
print("=" * 60)
