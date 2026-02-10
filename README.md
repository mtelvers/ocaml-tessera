# Tessera Minimal Pipeline

A single-file reimplementation of the Tessera satellite imagery embedding pipeline. The original spans ~15 files across two codebases (preprocessing + inference); this consolidates everything into one `pipeline.py`.

## What it does

1. **Downloads** Sentinel-2 (multispectral) and Sentinel-1 (SAR) satellite data from Microsoft Planetary Computer
2. **Preprocesses** the data: cloud masking (SCL), tile selection mosaicking, harmonisation, amplitude-to-dB conversion
3. **Runs inference** through a pretrained transformer model to produce 128-dimensional embeddings per pixel
4. **Quantizes** to int8 with per-pixel symmetric scaling and saves as `.npy` files

## Usage

```bash
# GPU (recommended)
python pipeline.py \
  --input_tiff global_map_0.1_degree_tiff/grid_51.05_10.35.tiff \
  --checkpoint best_model_fsdp_20250608_220648_QAT.pt \
  --output /tmp/embeddings/2024 \
  --device cuda

# CPU with multiple threads
python pipeline.py \
  --input_tiff global_map_0.1_degree_tiff/grid_51.05_10.35.tiff \
  --checkpoint best_model_fsdp_20250608_220648_QAT.pt \
  --output /tmp/embeddings/2024 \
  --device cpu --num_threads 20

# Skip download, use existing preprocessed data
python pipeline.py \
  --dpixel_dir /path/to/dpixel/grid_51.05_10.35 \
  --checkpoint best_model_fsdp_20250608_220648_QAT.pt \
  --output /tmp/embeddings/2024 \
  --device cuda
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input_tiff` | | Grid GeoTIFF defining the ROI (region of interest) |
| `--checkpoint` | | Path to model checkpoint `.pt` file |
| `--output` | | Output directory for `.npy` files |
| `--start` | 2024-01-01 | Start date for satellite data search |
| `--end` | 2024-12-31 | End date for satellite data search |
| `--max_cloud` | 100.0 | Max cloud cover % for Sentinel-2 STAC search |
| `--device` | auto | `auto`, `cpu`, or `cuda` |
| `--num_threads` | 1 | CPU threads (when device=cpu) |
| `--batch_size` | 1024 | Inference batch size |
| `--repeat_times` | 1 | Number of stratified sampling passes to average |
| `--dpixel_dir` | | Load preprocessed `.npy` files instead of downloading |

## Dependencies

```
torch
numpy
rasterio
pystac-client
planetary-computer
stackstac
pandas
```

## Verification

The pipeline has been verified to produce identical output to the original multi-file pipeline when run on the same preprocessed data (0 differences across 155 million values).

When comparing end-to-end output against the published reference data (which used random sampling), the stratified deterministic sampling achieves:

- Mean cosine similarity: 0.943
- 99.5% of pixels have cosine similarity > 0.9
- Remaining differences are due to sampling strategy (deterministic vs random), not bugs

### PCA visualisation

PCA of the 128-dimensional embeddings projected to RGB for the test tile (grid_51.05_10.35 — coastal Somalia):

| Published | Original pipeline | Minimal pipeline |
|-----------|------------------|-----------------|
| ![Published](npy-pca/pca_official.png) | ![Original](npy-pca/pca_original_pipeline.png) | ![Minimal](npy-pca/pca_minimal_pipeline.png) |

## Input tiles

The `--input_tiff` is a single-band uint8 GeoTIFF acting as a binary mask (0=skip, 1=process). It defines the spatial extent, CRS, and resolution for the satellite data download. The grid naming convention is `grid_{longitude}_{latitude}.tiff` at 0.1-degree spacing.

## Performance

On an NVIDIA L4 GPU (monteverde.cl.cam.ac.uk), processing grid_51.05_10.35 (269,908 valid pixels):

| Phase | Time |
|-------|------|
| Download & preprocess | ~12 min |
| Inference (GPU) | ~2.5 min |
| **Total** | **~15 min** |

Download time is dominated by the synchronous dask scheduler (required to work around a PROJ/rasterio threading issue).
