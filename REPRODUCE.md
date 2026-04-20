# AMX Benchmark

This document walks through reproducing the results in
[AMX blog post](https://www.tunbury.org/2026/04/08/intel-amx/).

# Hardware

The numbers in this document came from the following specific
hardware:

| Role | Hardware | Notes |
|---|---|---|
| AMX CPU | Intel Xeon Platinum 8573C | Emerald Rapids (5th Gen Scalable) |
| AMX cloud VM | Azure `Standard_D16s_v6` | 8 physical cores / 16 vCPU, UK West |
| T4 GPU VM | Azure `Standard_NC8as_T4_v3` | Tesla T4 (Turing), 16 GB GDDR6 |
| L4 GPU | NVIDIA L4 | Ada Lovelace, 24 GB GDDR6, local Monteverde host |

# Software

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch numpy onnxruntime
# For the full pipeline (not just the synthetic benchmark):
pip install dask rasterio planetary-computer pystac-client stackstac pandas
```

We used **PyTorch 2.11.0** (CPU wheels on the AMX VM; `+cu126` on the
GPU host). PyTorch's AMX support has evolved; earlier versions may
give different numbers.

For GPU use, install a PyTorch build matching your CUDA driver (see
[pytorch.org](https://pytorch.org)).

# Model Checkpoint

The Tessera encoder checkpoint
(`best_model_fsdp_20250608_220648_QAT.pt`, ~5.2 GB) is the one used
in the original pipeline.

# Input Data

For the synthetic benchmark, no input data is needed — the script
generates random tensors.

For the full pipeline, you need a preprocessed 0.1° tile cache in
Frank's dpixel format. This is produced by running `pipeline.py`
against a grid GeoTIFF with `--cache_dir` set. The cache directory
contains `s2/` and `s1/` subdirectories with the `.npy` arrays.

# Running the Synthetic Benchmark

This isolates pure inference time — no data loading, no sampling,
just the model being called repeatedly with dummy input.

```bash
# AMX CPU (8 physical cores, bfloat16 autocast)
python3 bin/bench_pytorch.py \
    --checkpoint best_model_fsdp_20250608_220648_QAT.pt \
    --device cpu --num_threads 8 \
    --batch_size 256 --num_batches 40

# GPU with bfloat16
python3 bin/bench_pytorch.py \
    --checkpoint best_model_fsdp_20250608_220648_QAT.pt \
    --device cuda --batch_size 256 --num_batches 40
```

Note: `bench_pytorch.py` as currently written does not have a
`--bf16` flag — the autocast is either added via editing the script
or by calling `torch.autocast` from a wrapper. The patched version
used in the blog post wraps the inference loop in:

```python
with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    output = model(s2_input, s1_input)
```

# Running the Full Pipeline

This is the end-to-end benchmark that produced the headline numbers
(17:31 on AMX, 2:46 on L4 GPU, etc.).

The `pipeline_bf16.py` script is a minimal patch to my single file
Tessera `pipeline.py` that adds a `--bf16` flag. The exact diff is
in `bin/pipeline_bf16.patch`. Three small changes:

1. Add `--bf16` command-line argument
2. Wrap the inference loop in `torch.autocast` when `--bf16` is set
3. Cast the model output to `float()` before `.cpu().numpy()` (bf16
   tensors can't convert directly to NumPy)

To run:

```bash
# AMX CPU, bfloat16 (8 physical cores)
python3 bin/pipeline_bf16.py \
    --dpixel_dir /path/to/cache/grid_-1.65_53.35 \
    --checkpoint best_model_fsdp_20250608_220648_QAT.pt \
    --output /tmp/bench_amx_bf16 \
    --device cpu --num_threads 8 \
    --batch_size 256 --repeat_times 1 --bf16

# GPU, bfloat16
python3 bin/pipeline_bf16.py \
    --dpixel_dir /path/to/cache/grid_-1.65_53.35 \
    --checkpoint best_model_fsdp_20250608_220648_QAT.pt \
    --output /tmp/bench_gpu_bf16 \
    --device cuda \
    --batch_size 256 --repeat_times 1 --bf16

# For float32 baseline, drop --bf16
```

The pipeline prints timing for each phase:

```
Inference complete: 15:08:29 (0:17:31.239215)
Done: 15:08:30 (total 0:17:32.675255)
  Download:  0:00:01.399394
  Inference: 0:17:31.239215
  Save:      0:00:00.036646
```

# Output Comparison

After running the same tile with multiple configurations, you can
compare output quality:

```python
import numpy as np

a_emb = np.load("/tmp/bench_gpu_f32/grid_-1.65_53.35.npy")
a_scales = np.load("/tmp/bench_gpu_f32/grid_-1.65_53.35_scales.npy")
b_emb = np.load("/tmp/bench_amx_bf16/grid_-1.65_53.35.npy")
b_scales = np.load("/tmp/bench_amx_bf16/grid_-1.65_53.35_scales.npy")

# Exact int8 match
print(f"Exact match: {np.mean(a_emb == b_emb) * 100:.1f}%")

# Cosine similarity on reconstructed float embeddings
mask = a_scales.flatten() > 0
a_f = (a_emb.astype(np.float32) * a_scales[..., np.newaxis]).reshape(-1, 128)[mask]
b_f = (b_emb.astype(np.float32) * b_scales[..., np.newaxis]).reshape(-1, 128)[mask]
cosine = np.sum(a_f * b_f, axis=1) / (
    np.linalg.norm(a_f, axis=1) * np.linalg.norm(b_f, axis=1) + 1e-10
)
print(f"Cosine similarity: mean={cosine.mean():.6f}, min={cosine.min():.6f}")
```

## Key Configuration Notes

- **Thread count**: Best performance came from 8 threads pinned to 8
  physical cores. Beyond that, scaling plateaus (16 physical cores
  was only 11% slower) and oversubscribing with hyperthreading is
  catastrophic — 32 threads on a 16-core VM ran ~15× slower than 8
  threads. Measured on the 16-physical-core VM:

  | Threads | bf16 ms/pixel | Notes |
  |---|---|---|
  | 8 | 1.08 | physical cores only |
  | 16 | 1.20 | all physical cores, no HT |
  | 32 | 17.71 | all vCPUs (HT enabled) |

- **Batch size**: 256 was optimal in our tests. Smaller batches lose
  efficiency; larger batches spill cache.
- **Framework**: PyTorch's `autocast` is the most straightforward way
  to drive AMX today — one line of code and the model runs through
  AMX tiles. We were unable to get ONNX Runtime to accelerate this
  f32-exported model on AMX: converting the graph to fp16 with
  `convert_float_to_float16` produced no speedup over f32, and
  `quantize_dynamic` to int8 failed on shape inference for the
  dynamic batch axis. Explicit bfloat16 model authoring may work but
  we did not verify it.
- **Device selection**: The patch detects device from
  `args.device` — `"cpu"` or `"cuda"`. The autocast context uses
  whichever device type PyTorch reports.

## Expected Results

For a 0.1° tile of ~773,000 valid pixels:

| Configuration | Inference Time |
|---|---|
| NVIDIA L4 GPU, bf16 | ~2:45 |
| NVIDIA L4 GPU, f32 | ~7:20 |
| Tesla T4 GPU, f32 | ~14:30 |
| Intel AMX CPU (8 threads), bf16 | ~17:30 |
| Tesla T4 GPU, bf16 | ~20:00 |
| Intel AMX CPU (8 threads), f32 | ~40:30 |

