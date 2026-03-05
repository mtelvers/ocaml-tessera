# Tessera OCaml Pipeline

An OCaml implementation of the Tessera satellite imagery embedding pipeline. Downloads Sentinel-2 and Sentinel-1 data, preprocesses it (cloud masking, mosaicking, harmonisation), runs ONNX model inference, and outputs quantized int8 embeddings.

Two tools are provided:

- **`bin/pipeline.ml`** — Full inference pipeline. Downloads Sentinel-2 and Sentinel-1 data per MGRS tile from Microsoft Planetary Computer (MPC) or AWS, processes spatial blocks, runs ONNX model inference, and outputs int8 embeddings.
- **`bin/dpixel.ml`** — Download-only tool. Fetches S2 and S1 satellite data and writes intermediate "dpixel" `.npy` files (bands, masks, doys, SAR ascending/descending) without running inference.

## Prerequisites

- OCaml 5.3.0 with opam
- GDAL (>= 3.10; 3.11+ recommended for Float16 support)
- ONNX Runtime shared library (`libonnxruntime.so`)

## External Repositories

These libraries must be cloned and pinned locally via `opam pin`:

| Library | Repository | Description |
|---------|-----------|-------------|
| `stac-client` | [mtelvers/stac-client](https://github.com/mtelvers/stac-client) | STAC API search + Planetary Computer token signing |
| `gdal` | [mtelvers/ocaml-gdal](https://github.com/mtelvers/ocaml-gdal) | GDAL bindings (ctypes-based) |
| `onnxruntime` | [mtelvers/ocaml-onnxruntime](https://github.com/mtelvers/ocaml-onnxruntime) | ONNX Runtime bindings (ctypes-based) |
| `npy` | [mtelvers/ocaml-npy](https://github.com/mtelvers/ocaml-npy) | NumPy `.npy` file reader/writer |

## Build

```bash
# Pin external dependencies (one-time setup)
opam pin add stac-client https://github.com/mtelvers/stac-client.git -n
opam pin add gdal https://github.com/mtelvers/ocaml-gdal.git -n
opam pin add onnxruntime https://github.com/mtelvers/ocaml-onnxruntime.git -n
opam pin add npy https://github.com/mtelvers/ocaml-npy.git -n
opam install stac-client gdal onnxruntime npy

# Build
eval $(opam env)
dune build bin/pipeline.exe
dune build bin/dpixel.exe
```

Note: GDAL ctypes bindings require relaxed C flags. The `dune-workspace` is configured with `-Wno-incompatible-pointer-types` for this reason.

## Usage

### pipeline.exe — Full inference

```bash
LD_LIBRARY_PATH=/path/to/onnxruntime/lib \
dune exec bin/pipeline.exe -- \
  --dpixel_dir /tmp/py_cache/grid_51.05_10.35 \
  --model tessera_model.onnx \
  --output /tmp/ocaml_out \
  --batch_size 1024 \
  --num_threads 20 \
  --repeat_times 1
```

### dpixel.exe — Download only

```bash
dune exec bin/dpixel.exe -- \
  --input_tiff global_map_0.1_degree_tiff/grid_51.05_10.35.tif \
  --output /tmp/dpixel_cache \
  --start 2024-01-01 \
  --end 2024-12-31
```

## Data Sources

- **Sentinel-2**: MPC (`planetarycomputer.microsoft.com`) or AWS (`earth-search.aws.element84.com`)
- **Sentinel-1 (SAR)**: NASA OPERA RTC-S1 via CMR (`cmr.earthdata.nasa.gov`)

## DPixel Output Format

The dpixel tool writes 7-8 `.npy` files per grid tile:

| File | Shape | Dtype | Description |
|------|-------|-------|-------------|
| `bands.npy` | (T,H,W,10) | uint16 | S2 spectral bands |
| `masks.npy` | (T,H,W) | uint8 | SCL-derived validity masks |
| `doys.npy` | (T,) | uint16 | Day-of-year for each S2 timestep |
| `sar_ascending.npy` | (T,H,W,2) | int16 | S1 ascending VV+VH |
| `sar_ascending_doy.npy` | (T,) | int16 | Day-of-year for ascending |
| `sar_descending.npy` | (T,H,W,2) | int16 | S1 descending VV+VH |
| `sar_descending_doy.npy` | (T,) | int16 | Day-of-year for descending |
| `mask_optimized.npy` | (T,H,W) | uint8 | OmniCloudMask (optional) |
