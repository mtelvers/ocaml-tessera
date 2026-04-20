#!/usr/bin/env python3
"""
Minimal single-file Tessera pipeline.
Download satellite data, preprocess, run deterministic inference, output int8 embeddings.

Usage:
    python pipeline.py --input_tiff grid_tiles/grid_51.05_10.35.tiff \
        --checkpoint checkpoints/best_model_fsdp_20250608_220648_QAT.pt \
        --output /tmp/embeddings/2024 --device cuda

    python pipeline.py --input_tiff grid_tiles/grid_51.05_10.35.tiff \
        --checkpoint checkpoints/best_model_fsdp_20250608_220648_QAT.pt \
        --output /tmp/embeddings/2024 --device cpu --num_threads 20

Dependencies: torch, numpy, rasterio, pystac-client, planetary-computer, stackstac, pandas
"""

import argparse
import math
import os

# Ensure PROJ can find its database (rasterio bundles its own proj.db).
# Use synchronous dask scheduler because rasterio's bundled PROJ fails
# to find proj.db in dask worker threads (a known packaging issue).
import importlib.util as _ilu
_spec = _ilu.find_spec("rasterio")
if _spec and _spec.origin:
    _d = os.path.dirname(_spec.origin)
    for _env, _sub in [("PROJ_LIB", "proj_data"), ("PROJ_DATA", "proj_data"),
                        ("GDAL_DATA", "gdal_data")]:
        _p = os.path.join(_d, _sub)
        if os.path.isdir(_p):
            os.environ[_env] = _p

import dask
dask.config.set(scheduler="synchronous")

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import stackstac
import torch
import torch.nn as nn
from datetime import datetime
from rasterio.warp import transform_bounds

# ======================== Constants ========================

S2_BANDS = ["B04", "B02", "B03", "B08", "B8A", "B05", "B06", "B07", "B11", "B12"]
SCL_INVALID = {0, 1, 2, 3, 8, 9}
HARMONISATION_DATE = np.datetime64("2022-01-25")
HARMONISATION_OFFSET = 1000
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

S2_BAND_MEAN = np.array([1711.0938, 1308.8511, 1546.4543, 3010.1293, 3106.5083,
                          2068.3044, 2685.0845, 2931.5889, 2514.6928, 1899.4922], dtype=np.float32)
S2_BAND_STD = np.array([1926.1026, 1862.9751, 1803.1792, 1741.7837, 1677.4543,
                         1888.7862, 1736.3090, 1715.8104, 1514.5199, 1398.4779], dtype=np.float32)
S1_BAND_MEAN = np.array([5484.0407, 3003.7812], dtype=np.float32)
S1_BAND_STD = np.array([1871.2334, 1726.0670], dtype=np.float32)

LATENT_DIM = 128
SAMPLE_SIZE_S2 = 40
SAMPLE_SIZE_S1 = 40
MIN_VALID_TIMESTEPS = 0
BATCH_SIZE = 1024


# ======================== Data Download & Preprocessing ========================

def load_roi(tiff_path):
    """Load tile geometry from a GeoTIFF."""
    with rasterio.open(tiff_path) as src:
        crs = src.crs
        bounds = src.bounds
        height, width = src.height, src.width
        resolution = src.res[0]
        mask = (src.read(1) > 0).astype(np.uint8)
    bbox = transform_bounds(crs, "EPSG:4326", *bounds)
    return crs, bounds, bbox, height, width, resolution, mask


def search_stac(collection, bbox, date_range, **extra_filters):
    """Search Planetary Computer STAC catalog."""
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[collection], bbox=bbox, datetime=date_range, **extra_filters,
    )
    return list(search.items())


def _stack(items, assets, crs, bounds, resolution):
    """Single stackstac call — returns a lazy xarray DataArray."""
    return stackstac.stack(
        items, assets=assets,
        epsg=crs.to_epsg(), bounds=bounds, resolution=resolution,
        snap_bounds=False, resampling=0,
    )


def process_s2(items, crs, bounds, resolution, mask):
    """Process Sentinel-2 items into bands, masks, and doys arrays.

    Matches the original tessera_preprocessing_end2end/s2_inmemory.py:
    - Pass 1: SCL analysis builds per-pixel tile_selection (first valid tile)
    - Coverage filter: >= 0.01% of ROI pixels valid
    - Pass 2: _smart_mosaic picks selected tile's pixel (not mean)
    - Harmonisation: subtract 1000 only where value >= 1000
    - Output mask: (tile_sel >= 0)
    """
    H, W = mask.shape
    if not items:
        print("  No S2 items found")
        return (np.empty((0, H, W, 10), dtype=np.uint16),
                np.empty((0, H, W), dtype=np.uint8),
                np.empty((0,), dtype=np.uint16))

    # --- Pass 1: bulk load SCL to build tile_selection per date ---
    print("  Loading SCL (cloud mask)...")
    scl_lazy = _stack(items, ["SCL"], crs, bounds, resolution)
    scl_all = scl_lazy.compute().values  # (N_items, 1, H', W')
    times = pd.DatetimeIndex(scl_lazy.coords["time"].values)
    H_out, W_out = scl_all.shape[2], scl_all.shape[3]

    H_use = min(mask.shape[0], H_out)
    W_use = min(mask.shape[1], W_out)
    roi = np.zeros((H_out, W_out), dtype=bool)
    roi[:H_use, :W_use] = mask[:H_use, :W_use].astype(bool)
    roi_total = int(roi.sum())

    dates = sorted(set(times.strftime("%Y-%m-%d")))
    valid_dates = []
    day_tile_sel = {}  # date_str -> tile_selection array
    # Map each item index to its date
    time_strs = times.strftime("%Y-%m-%d")

    for date_str in dates:
        date_mask = (time_strs == date_str)
        scl_day = scl_all[date_mask, 0, :, :]  # (N_tiles, H', W')
        n_tiles = scl_day.shape[0]

        # Build tile_selection: for each pixel, first tile with valid SCL
        valid_scl = ~np.isin(np.nan_to_num(scl_day, nan=0).astype(int),
                             list(SCL_INVALID))  # (N_tiles, H', W')
        tile_sel = np.full((H_out, W_out), -1, dtype=np.int16)
        assigned = np.zeros((H_out, W_out), dtype=bool)
        for t in range(n_tiles):
            cur = valid_scl[t] & roi & (~assigned)
            if np.any(cur):
                tile_sel[cur] = t
                assigned |= cur

        valid_cnt = int((tile_sel >= 0).sum())
        valid_pct = 100.0 * valid_cnt / roi_total if roi_total > 0 else 0.0
        # Original threshold: 0.01 (meaning 0.01% since valid_pct is 0-100)
        if valid_pct >= 0.01:
            valid_dates.append(date_str)
            day_tile_sel[date_str] = tile_sel

    print(f"  {len(valid_dates)}/{len(dates)} dates pass cloud filter")
    del scl_all

    if not valid_dates:
        return (np.empty((0, H_out, W_out, 10), dtype=np.uint16),
                np.empty((0, H_out, W_out), dtype=np.uint8),
                np.empty((0,), dtype=np.uint16))

    # --- Pass 2: bulk load spectral bands, smart mosaic ---
    print("  Loading spectral bands...")
    band_lazy = _stack(items, S2_BANDS, crs, bounds, resolution)
    band_all = band_lazy.compute().values  # (N_items, 10, H', W')
    band_times = pd.DatetimeIndex(band_lazy.coords["time"].values)
    band_time_strs = band_times.strftime("%Y-%m-%d")

    all_bands, all_masks, all_doys = [], [], []
    for date_str in valid_dates:
        doy = datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday
        date_mask = (band_time_strs == date_str)
        band_day = band_all[date_mask]  # (N_tiles, 10, H', W')
        tile_sel = day_tile_sel[date_str]

        out_bands = np.zeros((H, W, 10), dtype=np.uint16)
        for bi in range(10):
            data_arr = band_day[:, bi, :, :]  # (N_tiles, H', W')
            # _smart_mosaic: pick pixel from selected tile
            n = data_arr.shape[0]
            mosaicked = np.zeros((H, W), dtype=np.float64)
            hh = min(data_arr.shape[1], H, tile_sel.shape[0])
            ww = min(data_arr.shape[2], W, tile_sel.shape[1])
            roi_crop = roi[:hh, :ww]
            ts = tile_sel[:hh, :ww]
            # Pixels with a valid tile selection
            valid = roi_crop & (ts >= 0)
            if np.any(valid):
                y, x = np.where(valid)
                ti = np.clip(ts[y, x], 0, n - 1)
                mosaicked[y, x] = data_arr[ti, y, x]
            # Fallback: ROI pixels with no valid tile -> use first tile
            invalid = roi_crop & (ts < 0)
            if np.any(invalid) and n > 0:
                y2, x2 = np.where(invalid)
                mosaicked[y2, x2] = data_arr[0, y2, x2]

            # Harmonisation: subtract 1000 only where >= 1000
            if datetime.strptime(date_str, "%Y-%m-%d") > datetime(2022, 1, 25):
                harm_mask = ~np.isnan(mosaicked) & (mosaicked >= HARMONISATION_OFFSET)
                np.subtract(mosaicked, HARMONISATION_OFFSET, out=mosaicked, where=harm_mask)

            mosaicked = np.nan_to_num(mosaicked, nan=0.0, posinf=65535.0, neginf=0.0)
            out_bands[:, :, bi] = np.clip(mosaicked, 0, 65535).astype(np.uint16)

        out_mask = (tile_sel[:H, :W] >= 0).astype(np.uint8) if tile_sel.shape[0] >= H else \
            np.zeros((H, W), dtype=np.uint8)
        all_bands.append(out_bands)
        all_masks.append(out_mask)
        all_doys.append(doy)

    return (np.stack(all_bands).astype(np.uint16),
            np.stack(all_masks).astype(np.uint8),
            np.array(all_doys, dtype=np.uint16))


def amplitude_to_db(amp, mask=None):
    """Convert SAR amplitude to scaled dB, matching original pipeline.

    Formula: dB = (20*log10(amp) + 50) * 200, clipped to [0, 32767].
    If mask provided, zeros out pixels where mask <= 0.
    """
    amp = np.asarray(amp, dtype=np.float64)
    out = np.zeros_like(amp, dtype=np.int16)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = np.isfinite(amp) & (amp > 0)
    if np.any(valid):
        idx = np.where(valid)
        db = 20.0 * np.log10(amp[idx]) + 50.0
        scaled = db * 200.0
        out[idx] = np.clip(scaled, 0, 32767).astype(np.int16)
    if mask is not None:
        m = np.asarray(mask)
        hh = min(out.shape[0], m.shape[0])
        ww = min(out.shape[1], m.shape[1])
        out2 = np.zeros_like(out)
        out2[:hh, :ww] = np.where(m[:hh, :ww] > 0, out[:hh, :ww], 0)
        out = out2
    return out


def process_s1(items, crs, bounds, resolution, height, width, roi_mask=None):
    """Process Sentinel-1 items into ascending/descending arrays.

    Matches original: convert each tile to dB (with ROI mask) first,
    then mosaic dB values (mean where > 0).
    """
    H, W = height, width
    if not items:
        print("  No S1 items found")
        empty2 = np.empty((0, H, W, 2), dtype=np.int16)
        empty1 = np.empty((0,), dtype=np.int16)
        return empty2, empty1, empty2.copy(), empty1.copy()

    orbit_lookup = {}
    for item in items:
        orbit_lookup[item.id] = item.properties.get("sat:orbit_state", "unknown")

    print("  Loading SAR data...")
    sar_lazy = _stack(items, ["vv", "vh"], crs, bounds, resolution)
    sar_all = sar_lazy.compute().values  # (N_times, 2, H', W')
    H_out, W_out = sar_all.shape[2], sar_all.shape[3]

    # Build index mapping from stackstac's time coordinate (which may reorder items)
    stack_times = sar_lazy.coords["time"].values
    time_to_indices = {}
    for idx, t in enumerate(stack_times):
        day_str = str(np.datetime64(t, "D"))
        time_to_indices.setdefault(day_str, []).append(idx)

    by_date_orbit = {}
    for item in items:
        date_str = item.datetime.strftime("%Y-%m-%d")
        orbit = orbit_lookup[item.id]
        by_date_orbit.setdefault((date_str, orbit), None)

    # Map each (date, orbit) group to stackstac indices via time coordinate
    for (date_str, orbit) in by_date_orbit:
        by_date_orbit[(date_str, orbit)] = time_to_indices.get(date_str, [])

    asc_data, asc_doys = [], []
    desc_data, desc_doys = [], []

    for (date_str, orbit), indices in sorted(by_date_orbit.items()):
        doy = datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday
        sar_day = sar_all[indices]  # (N_tiles, 2, H', W')

        # For each polarisation: convert each tile to dB with mask, then mosaic
        def mosaic_pol(pol_arr):
            """pol_arr: (N_tiles, H', W') amplitude values. Returns None if no valid data."""
            db_list = []
            for t in range(pol_arr.shape[0]):
                db_list.append(amplitude_to_db(pol_arr[t], mask=roi_mask))
            if not db_list:
                return None
            arr = np.stack(db_list, axis=0)  # (N_tiles, H, W)
            valid = arr > 0
            if not np.any(valid):
                return None
            sumv = arr.astype(np.float64).sum(axis=0)
            cnt = valid.sum(axis=0)
            with np.errstate(invalid="ignore"):
                mean = np.divide(sumv, cnt, where=cnt > 0,
                                 out=np.zeros_like(sumv))
            out = np.zeros((H, W), dtype=np.int16)
            hh = min(mean.shape[0], H)
            ww = min(mean.shape[1], W)
            out[:hh, :ww] = mean[:hh, :ww].astype(np.int16)
            return out

        vv_out = mosaic_pol(sar_day[:, 0, :, :])
        vh_out = mosaic_pol(sar_day[:, 1, :, :])

        # Skip dates with no valid mosaicked data (e.g. orbit doesn't cover ROI)
        if vv_out is None and vh_out is None:
            continue
        if vv_out is None:
            vv_out = np.zeros((H, W), dtype=np.int16)
        if vh_out is None:
            vh_out = np.zeros((H, W), dtype=np.int16)

        out = np.zeros((H, W, 2), dtype=np.int16)
        out[:, :, 0] = vv_out
        out[:, :, 1] = vh_out

        if orbit == "ascending":
            asc_data.append(out)
            asc_doys.append(doy)
        else:
            desc_data.append(out)
            desc_doys.append(doy)

    def _stack_or_empty(data_list, doy_list):
        if data_list:
            return np.stack(data_list).astype(np.int16), np.array(doy_list, dtype=np.int16)
        return np.empty((0, H, W, 2), dtype=np.int16), np.empty((0,), dtype=np.int16)

    asc, asc_d = _stack_or_empty(asc_data, asc_doys)
    desc, desc_d = _stack_or_empty(desc_data, desc_doys)
    return asc, asc_d, desc, desc_d


# ======================== Model Architecture ========================

class TemporalPositionalEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, doy):
        position = doy.unsqueeze(-1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / self.d_model)
        )
        div_term = div_term.to(doy.device)
        pe = torch.zeros(doy.shape[0], doy.shape[1], self.d_model, device=doy.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class CustomGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ir = nn.Linear(input_size, hidden_size, bias=False)
        self.W_iz = nn.Linear(input_size, hidden_size, bias=False)
        self.W_ih = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_prev):
        r_t = torch.sigmoid(self.W_ir(x_t) + self.W_hr(h_prev) + self.b_r)
        z_t = torch.sigmoid(self.W_iz(x_t) + self.W_hz(h_prev) + self.b_z)
        h_tilde = torch.tanh(self.W_ih(x_t) + self.W_hh(r_t * h_prev) + self.b_h)
        return (1 - z_t) * h_prev + z_t * h_tilde


class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gru_cell = CustomGRUCell(input_size, hidden_size)

    def forward(self, x, h_0=None):
        if self.batch_first:
            batch_size, seq_len, _ = x.shape
        else:
            seq_len, batch_size, _ = x.shape
            x = x.transpose(0, 1)
        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size,
                               device=x.device, dtype=x.dtype)
        outputs = []
        h_t = h_0
        for t in range(seq_len):
            h_t = self.gru_cell(x[:, t, :], h_t)
            outputs.append(h_t)
        outputs = torch.stack(outputs, dim=1)
        if not self.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs, h_t


class CustomTemporalAwarePooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.temporal_context = CustomGRU(input_dim, input_dim, batch_first=True)
        self.query = nn.Linear(input_dim, 1)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        B, T, D = x.shape
        if T == 0:
            return torch.zeros(B, D, device=x.device, dtype=x.dtype)
        elif T == 1:
            return x.squeeze(1)
        x_context, _ = self.temporal_context(x)
        x_context = self.layer_norm(x_context)
        attn_weights = torch.softmax(self.query(x_context), dim=1)
        return (attn_weights * x).sum(dim=1)


class TransformerEncoder(nn.Module):
    def __init__(self, band_num, latent_dim, nhead=4, num_encoder_layers=4,
                 dim_feedforward=4096, dropout=0.1, max_seq_len=40):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(band_num, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 4),
        )
        self.temporal_encoder = TemporalPositionalEncoder(d_model=latent_dim * 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim * 4, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation="relu", batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers,
        )
        self.attn_pool = CustomTemporalAwarePooling(latent_dim * 4)

    def forward(self, x):
        bands = x[:, :, :-1]
        doy = x[:, :, -1]
        x = self.embedding(bands) + self.temporal_encoder(doy)
        x = self.transformer_encoder(x)
        return self.attn_pool(x)


class InferenceModel(nn.Module):
    """S2 backbone + S1 backbone -> concat -> dim_reducer -> 128-d embedding."""
    def __init__(self, s2_backbone, s1_backbone, dim_reducer):
        super().__init__()
        self.s2_backbone = s2_backbone
        self.s1_backbone = s1_backbone
        self.dim_reducer = dim_reducer

    def forward(self, s2_x, s1_x):
        s2_repr = self.s2_backbone(s2_x)
        s1_repr = self.s1_backbone(s1_x)
        fused = torch.cat([s2_repr, s1_repr], dim=-1)
        return self.dim_reducer(fused)


# ======================== Deterministic Sampling ========================

def stratified_select(valid_idx, sample_size, offset=0):
    """Stratified deterministic sampling: divide valid indices into sample_size
    equal bins and pick one element per bin. The offset (0..N-1) controls which
    position within each bin is selected, enabling multiple distinct passes."""
    n = len(valid_idx)
    if n == 0:
        return np.zeros(sample_size, dtype=valid_idx.dtype)
    if n <= sample_size:
        # Fewer valid indices than needed — tile to fill
        repeats = (sample_size // n) + 1
        return np.sort(np.tile(valid_idx, repeats)[:sample_size])
    # Divide into sample_size bins, pick one per bin using offset
    bins = np.linspace(0, n, sample_size + 1, dtype=np.float64)
    selected = np.empty(sample_size, dtype=valid_idx.dtype)
    for i in range(sample_size):
        lo = int(bins[i])
        hi = int(bins[i + 1])
        if hi <= lo:
            hi = lo + 1
        pos = lo + (offset % (hi - lo))
        selected[i] = valid_idx[min(pos, n - 1)]
    return selected


def sample_s2(s2_bands, s2_masks, s2_doys, offset=0):
    """Sample & normalise S2 data for one pixel. Returns (SAMPLE_SIZE_S2, 11)."""
    valid_idx = np.nonzero(s2_masks)[0]
    if len(valid_idx) == 0:
        valid_idx = np.arange(s2_bands.shape[0])
    idx = stratified_select(valid_idx, SAMPLE_SIZE_S2, offset)
    bands = (s2_bands[idx] - S2_BAND_MEAN) / (S2_BAND_STD + 1e-9)
    doys = s2_doys[idx].reshape(-1, 1)
    return np.hstack([bands, doys]).astype(np.float32)


def sample_s1(s1_asc_bands, s1_asc_doys, s1_desc_bands, s1_desc_doys, offset=0):
    """Sample & normalise S1 data for one pixel. Returns (SAMPLE_SIZE_S1, 3)."""
    s1_all = np.concatenate([s1_asc_bands, s1_desc_bands], axis=0)
    doys_all = np.concatenate([s1_asc_doys, s1_desc_doys], axis=0)
    valid_mask = np.any(s1_all != 0, axis=-1)
    valid_idx = np.nonzero(valid_mask)[0]
    if len(valid_idx) == 0:
        valid_idx = np.arange(s1_all.shape[0])
    idx = stratified_select(valid_idx, SAMPLE_SIZE_S1, offset)
    bands = (s1_all[idx] - S1_BAND_MEAN) / (S1_BAND_STD + 1e-9)
    doys = doys_all[idx].reshape(-1, 1)
    return np.hstack([bands, doys]).astype(np.float32)


# ======================== Quantization ========================

def quantize_symmetric(x_f32):
    """Per-sample symmetric int8 quantization. Returns (int8_array, scale)."""
    abs_max = np.max(np.abs(x_f32))
    scale = max(float(abs_max) / 127.0, 1e-8)
    quantized = np.clip(np.round(x_f32 / scale), -128, 127).astype(np.int8)
    return quantized, np.float32(scale)


# ======================== Main Pipeline ========================

def main():
    parser = argparse.ArgumentParser(description="Minimal Tessera pipeline")
    parser.add_argument("--input_tiff", help="Path to grid GeoTIFF (required unless --dpixel_dir)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--max_cloud", type=float, default=100.0, help="Max cloud cover %% for S2")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device: auto (detect GPU), cpu, or cuda")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="CPU threads (only used when device=cpu)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--repeat_times", type=int, default=1,
                        help="Number of stratified sampling passes to average (1=single pass)")
    parser.add_argument("--dpixel_dir", default=None,
                        help="Skip download; load existing preprocessed .npy files from this dir")
    parser.add_argument("--cache_dir", default=None,
                        help="Directory to cache preprocessed data. Downloads are saved here "
                             "and reused on subsequent runs (enables resumable batch processing)")
    parser.add_argument("--download_only", action="store_true",
                        help="Only download and preprocess (requires --cache_dir). Skip inference.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 autocast")
    args = parser.parse_args()

    if args.download_only and not args.cache_dir:
        parser.error("--download_only requires --cache_dir")

    # ---- Device setup ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.num_threads)
    print(f"Device: {device}")

    if args.input_tiff:
        grid_id = os.path.splitext(os.path.basename(args.input_tiff))[0]
    elif args.dpixel_dir:
        grid_id = os.path.basename(os.path.normpath(args.dpixel_dir))
    else:
        parser.error("--input_tiff is required unless --dpixel_dir is provided")
    date_range = f"{args.start}/{args.end}"

    # Skip if output already exists
    emb_path = os.path.join(args.output, f"{grid_id}.npy")
    scale_path = os.path.join(args.output, f"{grid_id}_scales.npy")
    if not args.download_only and os.path.isfile(emb_path) and os.path.isfile(scale_path):
        print(f"Output already exists: {emb_path}")
        print("Skipping. Use a different --output to rerun.")
        return

    t_start = datetime.now()
    print(f"\nStarted: {t_start:%Y-%m-%d %H:%M:%S}")

    # ================================================================
    # Step 1: Download & preprocess satellite data
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Step 1: Download & preprocess")
    print(f"{'='*60}")

    # Resolve where to load/save preprocessed data
    dpixel_load_dir = args.dpixel_dir
    dpixel_save_dir = None
    if args.cache_dir and not dpixel_load_dir:
        cache_tile_dir = os.path.join(args.cache_dir, grid_id)
        if os.path.isfile(os.path.join(cache_tile_dir, "s2", "bands.npy")):
            dpixel_load_dir = cache_tile_dir
            print(f"\nFound cached preprocessed data in {cache_tile_dir}")
        else:
            dpixel_save_dir = cache_tile_dir

    if dpixel_load_dir:
        print(f"\nLoading preprocessed data from {dpixel_load_dir}")
        # Support both flat layout (files in root) and subdirectory layout (s2/, s1/)
        d = dpixel_load_dir
        if os.path.isdir(os.path.join(d, "s2")):
            s2_bands = np.load(os.path.join(d, "s2", "bands.npy"))
            s2_masks = np.load(os.path.join(d, "s2", "masks.npy"))
            s2_doys = np.load(os.path.join(d, "s2", "doys.npy"))
            s1_asc = np.load(os.path.join(d, "s1", "sar_ascending.npy"))
            s1_asc_doy = np.load(os.path.join(d, "s1", "sar_ascending_doy.npy"))
            s1_desc = np.load(os.path.join(d, "s1", "sar_descending.npy"))
            s1_desc_doy = np.load(os.path.join(d, "s1", "sar_descending_doy.npy"))
        else:
            s2_bands = np.load(os.path.join(d, "bands.npy"))
            s2_masks = np.load(os.path.join(d, "masks.npy"))
            s2_doys = np.load(os.path.join(d, "doys.npy"))
            s1_asc = np.load(os.path.join(d, "sar_ascending.npy"))
            s1_asc_doy = np.load(os.path.join(d, "sar_ascending_doy.npy"))
            s1_desc = np.load(os.path.join(d, "sar_descending.npy"))
            s1_desc_doy = np.load(os.path.join(d, "sar_descending_doy.npy"))
        print(f"  S2: {s2_bands.shape}, S1 asc: {s1_asc.shape}, S1 desc: {s1_desc.shape}")
    else:
        print(f"\nLoading ROI from {args.input_tiff}")
        crs, bounds, bbox, H, W, res, roi_mask = load_roi(args.input_tiff)
        print(f"  Tile: {H}x{W}, resolution={res}, CRS={crs}")

        print(f"\nSearching Sentinel-2 ({date_range})...")
        s2_items = search_stac("sentinel-2-l2a", bbox, date_range,
                               query={"eo:cloud_cover": {"lt": args.max_cloud}})
        print(f"  Found {len(s2_items)} scenes")
        print("Processing Sentinel-2...")
        s2_bands, s2_masks, s2_doys = process_s2(s2_items, crs, bounds, res, roi_mask)
        print(f"  Result: {s2_bands.shape[0]} valid days, shape={s2_bands.shape}")

        print(f"\nSearching Sentinel-1 ({date_range})...")
        s1_items = search_stac("sentinel-1-rtc", bbox, date_range)
        print(f"  Found {len(s1_items)} scenes")
        print("Processing Sentinel-1...")
        s1_asc, s1_asc_doy, s1_desc, s1_desc_doy = process_s1(
            s1_items, crs, bounds, res, H, W, roi_mask=roi_mask)
        print(f"  Ascending: {s1_asc.shape[0]} passes, Descending: {s1_desc.shape[0]} passes")

        # Save preprocessed data to cache if requested
        if dpixel_save_dir:
            print(f"\nSaving preprocessed data to {dpixel_save_dir}")
            os.makedirs(os.path.join(dpixel_save_dir, "s2"), exist_ok=True)
            os.makedirs(os.path.join(dpixel_save_dir, "s1"), exist_ok=True)
            np.save(os.path.join(dpixel_save_dir, "s2", "bands.npy"), s2_bands)
            np.save(os.path.join(dpixel_save_dir, "s2", "masks.npy"), s2_masks)
            np.save(os.path.join(dpixel_save_dir, "s2", "doys.npy"), s2_doys)
            np.save(os.path.join(dpixel_save_dir, "s1", "sar_ascending.npy"), s1_asc)
            np.save(os.path.join(dpixel_save_dir, "s1", "sar_ascending_doy.npy"), s1_asc_doy)
            np.save(os.path.join(dpixel_save_dir, "s1", "sar_descending.npy"), s1_desc)
            np.save(os.path.join(dpixel_save_dir, "s1", "sar_descending_doy.npy"), s1_desc_doy)
            print("  Saved.")

    if args.download_only:
        print(f"\n--download_only: skipping inference.")
        print(f"Done: {datetime.now():%H:%M:%S} (total {datetime.now() - t_start})")
        return

    # Convert types to match the dataset expectations
    s2_bands = s2_bands.astype(np.float32)
    s2_masks = s2_masks.astype(np.int32)
    s2_doys = s2_doys.astype(np.int32)
    s1_asc = s1_asc.astype(np.float32)
    s1_asc_doy = s1_asc_doy.astype(np.int32)
    s1_desc = s1_desc.astype(np.float32)
    s1_desc_doy = s1_desc_doy.astype(np.int32)

    # Use the actual spatial dimensions from the S2 stackstac output
    _, H_actual, W_actual, _ = s2_bands.shape if s2_bands.ndim == 4 else (0, H, W, 10)

    s1_asc_empty = (s1_asc.shape[0] == 0)
    s1_desc_empty = (s1_desc.shape[0] == 0)

    t_download = datetime.now()
    print(f"\nDownload complete: {t_download:%H:%M:%S} ({t_download - t_start})")

    # ================================================================
    # Step 2: Find valid pixels
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Step 2: Find valid pixels (min_valid_timesteps={MIN_VALID_TIMESTEPS})")
    print(f"{'='*60}")

    valid_pixels = []  # list of (global_idx, i, j)
    for i in range(H_actual):
        for j in range(W_actual):
            global_idx = i * W_actual + j
            s2_mask_ij = s2_masks[:, i, j]
            s2_valid = int(s2_mask_ij.sum())
            s2_nonzero = bool(np.any(s2_bands[:, i, j, :] != 0))

            if not s1_asc_empty:
                s1_asc_valid = int(np.any(s1_asc[:, i, j, :] != 0, axis=-1).sum())
            else:
                s1_asc_valid = 0
            if not s1_desc_empty:
                s1_desc_valid = int(np.any(s1_desc[:, i, j, :] != 0, axis=-1).sum())
            else:
                s1_desc_valid = 0
            s1_total = s1_asc_valid + s1_desc_valid

            if s1_asc_empty and s1_desc_empty:
                if s2_nonzero and s2_valid >= MIN_VALID_TIMESTEPS:
                    valid_pixels.append((global_idx, i, j))
            else:
                if s2_nonzero and s2_valid >= MIN_VALID_TIMESTEPS and s1_total >= MIN_VALID_TIMESTEPS:
                    valid_pixels.append((global_idx, i, j))

    print(f"  {len(valid_pixels)} valid pixels out of {H_actual * W_actual}")
    if not valid_pixels:
        print("No valid pixels found. Exiting.")
        return

    # ================================================================
    # Step 3: Build model & load checkpoint
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Step 3: Build model & load checkpoint")
    print(f"{'='*60}")

    s2_backbone = TransformerEncoder(band_num=10, latent_dim=LATENT_DIM).to(device)
    s1_backbone = TransformerEncoder(band_num=2, latent_dim=LATENT_DIM).to(device)
    dim_reducer = nn.Sequential(nn.Linear(8 * LATENT_DIM, LATENT_DIM)).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_key = "model_state" if "model_state" in checkpoint else "model_state_dict"
    state_dict = checkpoint[state_key]

    s2_sd, s1_sd, dr_sd = {}, {}, {}
    for key, value in state_dict.items():
        clean = key.replace("_orig_mod.", "")
        if clean.startswith("s2_backbone."):
            s2_sd[clean.replace("s2_backbone.", "")] = value
        elif clean.startswith("s1_backbone."):
            s1_sd[clean.replace("s1_backbone.", "")] = value
        elif clean.startswith("dim_reducer."):
            dr_sd[clean.replace("dim_reducer.", "")] = value

    s2_backbone.load_state_dict(s2_sd, strict=True)
    s1_backbone.load_state_dict(s1_sd, strict=True)
    dim_reducer.load_state_dict(dr_sd, strict=True)
    print("  Checkpoint loaded successfully")

    model = InferenceModel(s2_backbone, s1_backbone, dim_reducer)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ================================================================
    # Step 4: Deterministic inference
    # ================================================================
    R = args.repeat_times
    print(f"\n{'='*60}")
    print(f"Step 4: Inference ({len(valid_pixels)} pixels, batch_size={args.batch_size}, repeat_times={R})")
    print(f"{'='*60}")

    out_embeddings = np.zeros((H_actual * W_actual, LATENT_DIM), dtype=np.int8)
    out_scales = np.zeros((H_actual * W_actual,), dtype=np.float32)
    n_batches = (len(valid_pixels) + args.batch_size - 1) // args.batch_size

    autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if args.bf16 else torch.no_grad()
    with torch.no_grad(), autocast_ctx:
        for batch_idx in range(n_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(valid_pixels))
            batch_pixels = valid_pixels[start:end]
            B = len(batch_pixels)
            global_indices = [gidx for gidx, _, _ in batch_pixels]

            sum_z = np.zeros((B, LATENT_DIM), dtype=np.float32)

            for r in range(R):
                s2_input_list = []
                s1_input_list = []

                for global_idx, i, j in batch_pixels:
                    s2_input_list.append(
                        sample_s2(s2_bands[:, i, j, :], s2_masks[:, i, j], s2_doys, offset=r))

                    asc_ij = s1_asc[:, i, j, :] if not s1_asc_empty else np.zeros((1, 2), dtype=np.float32)
                    asc_d = s1_asc_doy if not s1_asc_empty else np.array([180], dtype=np.int32)
                    desc_ij = s1_desc[:, i, j, :] if not s1_desc_empty else np.zeros((1, 2), dtype=np.float32)
                    desc_d = s1_desc_doy if not s1_desc_empty else np.array([180], dtype=np.int32)
                    s1_input_list.append(sample_s1(asc_ij, asc_d, desc_ij, desc_d, offset=r))

                s2_tensor = torch.tensor(np.stack(s2_input_list), dtype=torch.float32, device=device)
                s1_tensor = torch.tensor(np.stack(s1_input_list), dtype=torch.float32, device=device)
                sum_z += model(s2_tensor, s1_tensor).float().cpu().numpy()

            avg_z = sum_z / float(R)

            for b in range(B):
                q, s = quantize_symmetric(avg_z[b])
                out_embeddings[global_indices[b]] = q
                out_scales[global_indices[b]] = s

            if (batch_idx + 1) % 50 == 0 or batch_idx == n_batches - 1:
                pct = 100.0 * (batch_idx + 1) / n_batches
                print(f"  [{pct:5.1f}%] Batch {batch_idx + 1}/{n_batches}")

    t_inference = datetime.now()
    print(f"\nInference complete: {t_inference:%H:%M:%S} ({t_inference - t_download})")

    # ================================================================
    # Step 5: Save output
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Step 5: Save output")
    print(f"{'='*60}")

    out_embeddings = out_embeddings.reshape(H_actual, W_actual, LATENT_DIM)
    out_scales = out_scales.reshape(H_actual, W_actual)

    os.makedirs(args.output, exist_ok=True)
    np.save(emb_path, out_embeddings)
    np.save(scale_path, out_scales)

    print(f"  Embeddings: {emb_path}  shape={out_embeddings.shape} dtype={out_embeddings.dtype}")
    print(f"  Scales:     {scale_path}  shape={out_scales.shape} dtype={out_scales.dtype}")

    t_done = datetime.now()
    print(f"\nDone: {t_done:%H:%M:%S} (total {t_done - t_start})")
    print(f"  Download:  {t_download - t_start}")
    print(f"  Inference: {t_inference - t_download}")
    print(f"  Save:      {t_done - t_inference}")


if __name__ == "__main__":
    main()
