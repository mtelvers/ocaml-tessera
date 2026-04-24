(* Tessera pipeline — MGRS tile-based global processing.
   Downloads S2 and S1 satellite data per MGRS tile, processes in spatial blocks,
   runs deterministic ONNX inference, outputs int8 embeddings. *)

open Bigarray

(* ======================== Constants ======================== *)

let s2_bands = [| "B04"; "B02"; "B03"; "B08"; "B8A"; "B05"; "B06"; "B07"; "B11"; "B12" |]
let scl_invalid = [| 0; 1; 2; 3; 8; 9 |]
let harmonisation_date_ymd = (2022, 1, 25)
let harmonisation_offset = 1000
let stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"

type data_source = MPC | AWS

let aws_stac_url = "https://earth-search.aws.element84.com/v1"

(* AWS S2 band names, same order as s2_bands for MPC *)
let s2_bands_aws = [| "red"; "blue"; "green"; "nir"; "nir08";
                       "rededge1"; "rededge2"; "rededge3"; "swir16"; "swir22" |]

let cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
let opera_collection_id = "C2777436413-ASF"

let s2_band_names = function MPC -> s2_bands | AWS -> s2_bands_aws
let scl_asset_name = function MPC -> "SCL" | AWS -> "scl"

let s2_band_mean = [| 1711.0938; 1308.8511; 1546.4543; 3010.1293; 3106.5083;
                       2068.3044; 2685.0845; 2931.5889; 2514.6928; 1899.4922 |]
let s2_band_std  = [| 1926.1026; 1862.9751; 1803.1792; 1741.7837; 1677.4543;
                       1888.7862; 1736.3090; 1715.8104; 1514.5199; 1398.4779 |]
let s1_band_mean = [| 5484.0407; 3003.7812 |]
let s1_band_std  = [| 1871.2334; 1726.0670 |]

let latent_dim = 128
let sample_size_s2 = 40
let sample_size_s1 = 40
let min_valid_timesteps = 0
let default_batch_size = 2048

(* S2 bands at 10m resolution (indices 0-3 in s2_bands array) *)
let is_10m_band idx = idx < 4

(* ======================== Utility helpers ======================== *)

let is_scl_invalid v =
  Array.exists (fun x -> x = v) scl_invalid

(** Day-of-year from "YYYY-MM-DD" *)
let doy_of_date_str s =
  Scanf.sscanf s "%d-%d-%d" (fun y m d ->
    let tm = { Unix.tm_sec = 0; tm_min = 0; tm_hour = 12;
               tm_mday = d; tm_mon = m - 1; tm_year = y - 1900;
               tm_wday = 0; tm_yday = 0; tm_isdst = false } in
    let (_, tm') = Unix.mktime tm in
    tm'.Unix.tm_yday + 1)

(** Check if date_str > harmonisation_date *)
let is_after_harmonisation date_str =
  Scanf.sscanf date_str "%d-%d-%d" (fun y m d ->
    let (hy, hm, hd) = harmonisation_date_ymd in
    (y, m, d) > (hy, hm, hd))

(** Extract just YYYY-MM-DD from an ISO datetime string *)
let date_of_datetime dt =
  if String.length dt >= 10 then String.sub dt 0 10
  else dt

(** Sort and deduplicate a string list *)
let sort_uniq_strings lst =
  List.sort_uniq String.compare lst

(** Get asset href from a STAC item *)
let get_asset_href (item : Stac_client.item) key =
  match List.assoc_opt key item.assets with
  | Some a -> Some a.href
  | None -> None

(** Printf to stderr for progress *)
let eprintf fmt = Printf.eprintf fmt

let rec mkdir_p dir =
  if dir <> "/" && dir <> "." && not (Sys.file_exists dir) then begin
    mkdir_p (Filename.dirname dir);
    (try Unix.mkdir dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ())
  end

(* ======================== Flat bigarray types ======================== *)

(** Flat 4D array: [n_dates][height][width][channels] as contiguous uint16 bigarray.
    Index: ((d * h + i) * w + j) * c + b
    Uses uint16 (2 bytes) instead of float64 (8 bytes) — S2 data is 16-bit sensor
    values and S1 dB-scaled values fit in 0-32767. 4x memory reduction. *)
type flat_4d = {
  data : (int, int16_unsigned_elt, c_layout) Array1.t;
  n : int; h : int; w : int; c : int;
}

(** Flat 3D int array: [n_dates][height][width] as contiguous byte bigarray.
    Index: (d * h3 + i) * w3 + j
    Masks are 0 or 1 — byte (1 byte) instead of int (8 bytes). 8x reduction. *)
type flat_3d_int = {
  idata : (int, int8_unsigned_elt, c_layout) Array1.t;
  n3 : int; h3 : int; w3 : int;
}

let empty_flat_4d = { data = Array1.create int16_unsigned c_layout 0; n = 0; h = 0; w = 0; c = 0 }
let empty_flat_3d_int = { idata = Array1.create int8_unsigned c_layout 0; n3 = 0; h3 = 0; w3 = 0 }

(** S1 tile: URLs for a single SAR tile's VV and VH polarisations *)
type s1_tile = {
  st_id : string;
  st_vv : string option;
  st_vh : string option;
}

(** S1 group: one (date, orbit) combination with its tiles *)
type s1_group = {
  sg_date : string;
  sg_orbit : string;
  sg_tiles : s1_tile list;
}

(* ======================== ROI type ======================== *)

type roi = {
  crs_wkt : string;     (* WKT string of CRS *)
  bounds : float * float * float * float;  (* left, bottom, right, top *)
  bbox : float * float * float * float;    (* EPSG:4326 bbox *)
  height : int;
  width : int;
  resolution : float;
  geotransform : float * float * float * float;  (* origin_x, pixel_w, origin_y, pixel_h *)
}

(* ======================== Parallel map ======================== *)

let parallel_map ~n_workers f items =
  let n = Array.length items in
  if n = 0 || n_workers <= 1 then Array.map (fun x -> Some (f x)) items
  else begin
    let results = Array.make n None in
    let next = Atomic.make 0 in
    let actual = min n_workers n in
    let domains = Array.init actual (fun _ ->
      Domain.spawn (fun () ->
        let rec loop () =
          let i = Atomic.fetch_and_add next 1 in
          if i < n then begin
            results.(i) <- (try Some (f items.(i)) with exn ->
              Printf.eprintf "Warning: parallel task %d failed: %s\n%!" i
                (Printexc.to_string exn); None);
            loop ()
          end
        in loop ())
    ) in
    Array.iter Domain.join domains;
    results
  end

(* ======================== COG reading via GDAL warp ======================== *)

(** Convert a URL to a GDAL virtual filesystem path.
    s3://bucket/key -> /vsis3/bucket/key, https://... -> /vsicurl/https://... *)
let gdal_vsi_path url =
  if String.length url > 5 && String.sub url 0 5 = "s3://" then
    "/vsis3/" ^ String.sub url 5 (String.length url - 5)
  else
    "/vsicurl/" ^ url

let warp_id = Atomic.make 0

let warp_to_mem ?(resampling="near") ~dst_crs_wkt ~bounds:(left, bottom, right, top)
    ~width ~height src_path =
  let id = Atomic.fetch_and_add warp_id 1 in
  let dst = Printf.sprintf "/vsimem/warp_%d" id in
  let vsi_path = if String.length src_path > 5 &&
                    (String.sub src_path 0 5 = "s3://" || String.sub src_path 0 5 = "http:" ||
                     String.sub src_path 0 6 = "https:")
                 then gdal_vsi_path src_path
                 else src_path in
  match Gdal.Dataset.open_ex ~thread_safe:true vsi_path with
  | Error msg ->
    eprintf "Warning: open failed for %s: %s\n%!" src_path msg;
    Error msg
  | Ok src ->
    let opts = [
      "-t_srs"; dst_crs_wkt;
      "-te"; Printf.sprintf "%.15g" left;
             Printf.sprintf "%.15g" bottom;
             Printf.sprintf "%.15g" right;
             Printf.sprintf "%.15g" top;
      "-ts"; string_of_int width; string_of_int height;
      "-r"; resampling;
      "-of"; "MEM";
    ] in
    let result = Gdal.Dataset.warp src ~dst_filename:dst opts in
    Gdal.Dataset.close src;
    result

(* ======================== MGRS tile ROI ======================== *)

(** Derive ROI from an MGRS tile ID by querying STAC for one S2 item,
    then opening one of its band COGs to read the native geotransform. *)
let mgrs_tile_roi ~(client : Stac_client.t) ~data_source tile_id =
  let base_url, query_field = match data_source with
    | MPC -> stac_url,
             `Assoc ["s2:mgrs_tile", `Assoc ["eq", `String tile_id]]
    | AWS -> aws_stac_url,
             `Assoc ["grid:code", `Assoc ["eq", `String (Printf.sprintf "MGRS-%s" tile_id)]]
  in
  let params : Stac_client.search_params = {
    collections = ["sentinel-2-l2a"];
    bbox = [-180.0; -90.0; 180.0; 90.0];
    datetime = "2024-01-01/2024-12-31";
    query = Some query_field;
    limit = Some 1;
  } in
  let items = Stac_client.search client ~base_url params in
  match items with
  | [] -> failwith (Printf.sprintf "No S2 items found for MGRS tile %s" tile_id)
  | item :: _ ->
    (* Get EPSG from properties *)
    let epsg = match item.properties with
      | `Assoc fields ->
        (match List.assoc_opt "proj:epsg" fields with
         | Some (`Int i) -> i
         | Some (`Float f) -> Float.to_int f
         | _ -> failwith "Missing proj:epsg")
      | _ -> failwith "Bad properties"
    in
    (* Sign the item so we can access its COG *)
    let access_item = match data_source with
      | MPC -> Stac_client.sign_planetary_computer client item
      | AWS -> item
    in
    (* Open a 10m band COG to get the native geotransform *)
    let band_name = match data_source with MPC -> "B04" | AWS -> "red" in
    let href = match get_asset_href access_item band_name with
      | Some h -> h | None -> failwith "No B04/red asset in S2 item"
    in
    eprintf "  Reading geotransform from %s...\n%!" band_name;
    let ds = Gdal.Dataset.open_ex ~thread_safe:true (gdal_vsi_path href) |> Result.get_ok in
    let w = Gdal.Dataset.raster_x_size ds in
    let h = Gdal.Dataset.raster_y_size ds in
    let gt = Gdal.Dataset.get_geo_transform ds |> Result.get_ok in
    let crs_wkt_raw = Gdal.Dataset.projection ds in
    Gdal.Dataset.close ds;
    let origin_x = gt.origin_x in
    let pixel_w = gt.pixel_width in
    let origin_y = gt.origin_y in
    let pixel_h = gt.pixel_height in
    let left = origin_x in
    let top = origin_y in
    let right = left +. Float.of_int w *. pixel_w in
    let bottom = top +. Float.of_int h *. pixel_h in
    let resolution = Float.abs pixel_w in
    (* Use CRS WKT from dataset if available, else build from EPSG *)
    let crs_wkt = if String.length crs_wkt_raw > 0 then crs_wkt_raw
      else begin
        let srs = Gdal.SpatialReference.of_epsg epsg |> Result.get_ok in
        let wkt = Gdal.SpatialReference.to_wkt srs |> Result.get_ok in
        Gdal.SpatialReference.destroy srs;
        wkt
      end
    in
    (* Compute EPSG:4326 bbox *)
    let tile_srs = Gdal.SpatialReference.of_wkt crs_wkt |> Result.get_ok in
    let wgs84 = Gdal.SpatialReference.of_epsg 4326 |> Result.get_ok in
    let ct = Gdal.CoordinateTransformation.create tile_srs wgs84 |> Result.get_ok in
    let (xmin, ymin, xmax, ymax) =
      Gdal.CoordinateTransformation.transform_bounds ct
        ~xmin:(Float.min left right) ~ymin:(Float.min bottom top)
        ~xmax:(Float.max left right) ~ymax:(Float.max bottom top)
        ~density:21
      |> Result.get_ok
    in
    Gdal.CoordinateTransformation.destroy ct;
    Gdal.SpatialReference.destroy wgs84;
    Gdal.SpatialReference.destroy tile_srs;
    eprintf "  MGRS %s: EPSG:%d, %dx%d, res=%.1f\n%!" tile_id epsg w h resolution;
    eprintf "  Bounds: (%.2f, %.2f) - (%.2f, %.2f)\n%!" left bottom right top;
    eprintf "  BBOX (4326): (%.6f, %.6f) - (%.6f, %.6f)\n%!" xmin ymin xmax ymax;
    { crs_wkt;
      bounds = (left, bottom, right, top);
      bbox = (xmin, ymin, xmax, ymax);
      height = h;
      width = w;
      resolution;
      geotransform = (origin_x, pixel_w, origin_y, pixel_h) }

(* ======================== COG download to cache ======================== *)

(** Download a COG to local cache via GDAL translate. Skips if file exists. *)
let download_cog ~cache_path url =
  if Sys.file_exists cache_path then ()
  else begin
    mkdir_p (Filename.dirname cache_path);
    let vsi = gdal_vsi_path url in
    match Gdal.copy_file ~src:vsi ~dst:cache_path with
    | Ok () -> ()
    | Error msg -> eprintf "Warning: download failed for %s: %s\n%!" url msg
  end

(* ======================== S2 SCL analysis ======================== *)

(** Read a single-band GeoTIFF from local file as a flat byte bigarray.
    Returns (data, height, width).  Uses 1 byte/pixel instead of 16 bytes
    for boxed OCaml ints — critical for memory when reading 200+ SCL files. *)
let read_local_band_byte path =
  match Gdal.Dataset.open_ex ~thread_safe:true path with
  | Error msg -> failwith (Printf.sprintf "Failed to open %s: %s" path msg)
  | Ok ds ->
    let band = Gdal.Dataset.get_band ds 1 |> Result.get_ok in
    let w = Gdal.RasterBand.x_size band |> Result.get_ok in
    let h = Gdal.RasterBand.y_size band |> Result.get_ok in
    let ba = Gdal.RasterBand.read_byte band
               ~x_off:0 ~y_off:0 ~x_size:w ~y_size:h |> Result.get_ok in
    let flat = Array1.create int8_unsigned c_layout (h * w) in
    for i = 0 to h - 1 do
      for j = 0 to w - 1 do
        Array1.set flat (i * w + j) (Array2.get ba i j)
      done
    done;
    Gdal.Dataset.close ds;
    (flat, h, w)

(** Tile selection type: for each pixel at 20m, which tile index to use *)
type date_info = {
  di_date : string;
  di_doy : int;
  di_tile_sel : (int, int8_signed_elt, c_layout) Array1.t;  (* flat [scl_h * scl_w], tile index or -1; int8 = 1 byte/pixel *)
  di_item_indices : int list;  (* indices into items_arr for this date *)
}

(** Analyze SCL for all items, determine valid dates and tile selections.
    SCL is at 20m = (h/2) x (w/2). Returns (valid_dates, water_mask).
    water_mask is a bool array at 10m resolution: true = water (skip). *)
let analyze_scl ~scl_cache_dir ~roi_h ~roi_w items_arr item_dates =
  let scl_h = roi_h / 2 and scl_w = roi_w / 2 in
  let scl_total = scl_h * scl_w in
  let dates = sort_uniq_strings (Array.to_list item_dates) in

  (* Read all cached SCL files — byte bigarrays (1 byte/pixel, not 16) *)
  let n_items = Array.length items_arr in
  let scl_data = Array.init n_items (fun i ->
    let path = Filename.concat scl_cache_dir
                 (Printf.sprintf "%s.tif" items_arr.(i).Stac_client.id) in
    if Sys.file_exists path then
      let (arr, _h, _w) = read_local_band_byte path in arr
    else begin
      let z = Array1.create int8_unsigned c_layout scl_total in
      Array1.fill z 0; z
    end
  ) in

  (* Compute per-pixel water frequency at 20m resolution.
     SCL class 6 = water. Count how many scenes classify each pixel as water
     vs how many scenes have valid (non-invalid) data for that pixel. *)
  let water_count = Array.make scl_total 0 in
  let valid_scene_count = Array.make scl_total 0 in
  Array.iter (fun scl ->
    for px = 0 to scl_total - 1 do
      let v = Array1.get scl px in
      if not (is_scl_invalid v) then begin
        valid_scene_count.(px) <- valid_scene_count.(px) + 1;
        if v = 6 then
          water_count.(px) <- water_count.(px) + 1
      end
    done
  ) scl_data;

  (* Water mask at 10m: pixel is water if >80% of valid scenes say water *)
  let water_mask = Array.make (roi_h * roi_w) false in
  let water_px_count = ref 0 in
  for sy = 0 to scl_h - 1 do
    for sx = 0 to scl_w - 1 do
      let spx = sy * scl_w + sx in
      let is_water =
        valid_scene_count.(spx) > 0 &&
        Float.of_int water_count.(spx) /. Float.of_int valid_scene_count.(spx) > 0.8
      in
      if is_water then begin
        (* Expand 20m pixel to 2x2 block at 10m *)
        let y0 = sy * 2 and x0 = sx * 2 in
        for dy = 0 to 1 do
          for dx = 0 to 1 do
            let y = y0 + dy and x = x0 + dx in
            if y < roi_h && x < roi_w then begin
              water_mask.(y * roi_w + x) <- true;
              incr water_px_count
            end
          done
        done
      end
    done
  done;
  let water_pct = 100.0 *. Float.of_int !water_px_count /. Float.of_int (roi_h * roi_w) in
  Printf.printf "  Water mask: %d pixels (%.1f%%) classified as water\n%!" !water_px_count water_pct;

  let valid_dates = ref [] in
  List.iter (fun date_str ->
    let indices = ref [] in
    Array.iteri (fun i d -> if d = date_str then indices := i :: !indices) item_dates;
    let indices = List.rev !indices in

    (* Build tile_selection at 20m resolution — int8_signed bigarray *)
    let tile_sel = Array1.create int8_signed c_layout scl_total in
    Array1.fill tile_sel (-1);
    List.iteri (fun t_idx item_idx ->
      let scl = scl_data.(item_idx) in
      for px = 0 to scl_total - 1 do
        if Array1.get tile_sel px < 0 then begin
          let v = Array1.get scl px in
          if not (is_scl_invalid v) then
            Array1.set tile_sel px t_idx
        end
      done
    ) indices;

    (* Coverage check *)
    let valid_cnt = ref 0 in
    for px = 0 to scl_total - 1 do
      if Array1.get tile_sel px >= 0 then incr valid_cnt
    done;
    let valid_pct = 100.0 *. Float.of_int !valid_cnt /. Float.of_int scl_total in
    if valid_pct >= 0.01 then begin
      let doy = doy_of_date_str date_str in
      valid_dates := { di_date = date_str; di_doy = doy;
                       di_tile_sel = tile_sel; di_item_indices = indices } :: !valid_dates
    end
  ) dates;
  (List.rev !valid_dates, water_mask)

(* ======================== Block-based S2 reading ======================== *)

(** Read a band block from a local GeoTIFF directly into a uint16 output bigarray.
    For 10m bands, read directly. For 20m bands, read at half and upsample.
    Uses BA_uint16 — reads 2 bytes per pixel instead of 8 (BA_float64). *)
let read_band_into ~path ~x_off ~y_off ~bw ~bh ~is_10m
    ~(dst : (int, int16_unsigned_elt, c_layout) Array1.t) ~dst_offset =
  match Gdal.Dataset.open_ex ~thread_safe:true path with
  | Error msg ->
    eprintf "Warning: failed to open %s: %s\n%!" path msg
  | Ok ds ->
    let band = Gdal.Dataset.get_band ds 1 |> Result.get_ok in
    let ba = if is_10m then
      Gdal.RasterBand.read_region Gdal.BA_uint16 band
        ~x_off ~y_off ~x_size:bw ~y_size:bh
        ~buf_x:bw ~buf_y:bh |> Result.get_ok
    else
      Gdal.RasterBand.read_region Gdal.BA_uint16 band
        ~x_off:(x_off / 2) ~y_off:(y_off / 2) ~x_size:(bw / 2) ~y_size:(bh / 2)
        ~buf_x:bw ~buf_y:bh |> Result.get_ok
    in
    for i = 0 to bh - 1 do
      for j = 0 to bw - 1 do
        Array1.set dst (dst_offset + i * bw + j) (Array2.get ba i j)
      done
    done;
    Gdal.Dataset.close ds

(** Read S2 data for one spatial block across all valid dates.
    Returns (flat_4d [n_dates, bh, bw, 10], flat_3d_int [n_dates, bh, bw], int array doys).
    Applies harmonisation and tile selection mosaic.
    Reads bands one at a time directly into the output bigarray to avoid
    allocating billions of boxed floats. *)
let read_s2_block ~band_cache_dir ~valid_dates ~items_arr ~data_source
    ~x_off ~y_off ~bw ~bh ~roi_w =
  let full_scl_w = roi_w / 2 in
  let n_dates = List.length valid_dates in
  if n_dates = 0 then (empty_flat_4d, empty_flat_3d_int, [||])
  else begin
    let total_4d = n_dates * bh * bw * 10 in
    let total_3d = n_dates * bh * bw in
    let bands_data = Array1.create int16_unsigned c_layout total_4d in
    let masks_data = Array1.create int8_unsigned c_layout total_3d in
    Array1.fill bands_data 0;
    Array1.fill masks_data 0;
    let doys = Array.make n_dates 0 in
    let band_names = s2_band_names data_source in

    (* Temp buffer for reading one band at a time — uint16 *)
    let band_buf = Array1.create int16_unsigned c_layout (bh * bw) in

    List.iteri (fun d_idx di ->
      doys.(d_idx) <- di.di_doy;
      let n_tiles = List.length di.di_item_indices in
      let date_base = d_idx * bh * bw in
      let harmonise = is_after_harmonisation di.di_date in

      (* First pass: determine tile selection per pixel for this date *)
      let pixel_tile = Array.make (bh * bw) (-1) in
      for i = 0 to bh - 1 do
        let scl_i = (y_off + i) / 2 in
        let row_base = i * bw in
        for j = 0 to bw - 1 do
          let scl_j = (x_off + j) / 2 in
          let scl_px = scl_i * full_scl_w + scl_j in
          let ts = if scl_px >= 0 && scl_px < Array1.dim di.di_tile_sel
                   then Array1.get di.di_tile_sel scl_px else -1 in
          pixel_tile.(row_base + j) <- ts;
          if ts >= 0 then
            Array1.set masks_data (date_base + row_base + j) 1
        done
      done;

      (* Build set of used tile indices *)
      let used_tiles = Array.make n_tiles false in
      for px = 0 to bh * bw - 1 do
        let t = pixel_tile.(px) in
        if t >= 0 && t < n_tiles then used_tiles.(t) <- true
      done;

      (* For each used tile, read its bands and fill pixels assigned to it *)
      for t_idx = 0 to n_tiles - 1 do
        if used_tiles.(t_idx) then begin
          let item_idx = List.nth di.di_item_indices t_idx in
          let item = items_arr.(item_idx) in
          for bi = 0 to 9 do
            let bn = band_names.(bi) in
            let path = Filename.concat band_cache_dir
                         (Printf.sprintf "%s/%s.tif" item.Stac_client.id bn) in
            if Sys.file_exists path then begin
              read_band_into ~path ~x_off ~y_off ~bw ~bh
                ~is_10m:(is_10m_band bi) ~dst:band_buf ~dst_offset:0;
              for px = 0 to bh * bw - 1 do
                if pixel_tile.(px) = t_idx then begin
                  let v = Array1.get band_buf px in
                  let v = if harmonise && v >= harmonisation_offset
                          then v - harmonisation_offset else v in
                  let v = max 0 (min 65535 v) in
                  Array1.set bands_data ((date_base + px) * 10 + bi) v
                end
              done
            end
          done
        end
      done;

      (* Fallback: pixels with no tile selection get data from first tile *)
      if n_tiles > 0 then begin
        let need_fallback = ref false in
        for px = 0 to bh * bw - 1 do
          if pixel_tile.(px) < 0 then (need_fallback := true)
        done;
        if !need_fallback then begin
          let item_idx = List.nth di.di_item_indices 0 in
          let item = items_arr.(item_idx) in
          for bi = 0 to 9 do
            let bn = band_names.(bi) in
            let path = Filename.concat band_cache_dir
                         (Printf.sprintf "%s/%s.tif" item.Stac_client.id bn) in
            if Sys.file_exists path then begin
              read_band_into ~path ~x_off ~y_off ~bw ~bh
                ~is_10m:(is_10m_band bi) ~dst:band_buf ~dst_offset:0;
              for px = 0 to bh * bw - 1 do
                if pixel_tile.(px) < 0 then begin
                  let v = Array1.get band_buf px in
                  let v = if harmonise && v >= harmonisation_offset
                          then v - harmonisation_offset else v in
                  let v = max 0 (min 65535 v) in
                  Array1.set bands_data ((date_base + px) * 10 + bi) v
                end
              done
            end
          done
        end
      end
    ) valid_dates;

    ({ data = bands_data; n = n_dates; h = bh; w = bw; c = 10 },
     { idata = masks_data; n3 = n_dates; h3 = bh; w3 = bw },
     doys)
  end

(* ======================== S1 processing ======================== *)

(** Read S1 data for one spatial block from pre-warped cached files.
    Uses reusable scratch buffers to avoid per-tile allocations.
    Returns (asc flat_4d, asc_doys, desc flat_4d, desc_doys). *)
let read_s1_block ~s1_groups ~s1_cache_dir
    ~x_off ~y_off ~bw ~bh ~data_source =
  if s1_groups = [] then
    (empty_flat_4d, [||], empty_flat_4d, [||])
  else begin
    let npx = bh * bw in
    (* Reusable scratch buffers — allocated once, reused for all tiles *)
    let amp_buf = Array1.create float64 c_layout npx in
    let mosaic_sum = Array.make npx 0.0 in
    let mosaic_cnt = Array.make npx 0 in

    let process_group group =
      let doy = doy_of_date_str group.sg_date in

      let mosaic_pol suffix =
        let result = Array.make npx 0 in
        Array.fill mosaic_sum 0 npx 0.0;
        Array.fill mosaic_cnt 0 npx 0;
        let any_valid = ref false in

        List.iter (fun tile ->
          let cache_path = Filename.concat s1_cache_dir
            (tile.st_id ^ suffix ^ ".tif") in
          (* Read amplitude into reusable bigarray *)
          Array1.fill amp_buf 0.0;
          (if Sys.file_exists cache_path then
            match Gdal.Dataset.open_ex ~thread_safe:true cache_path with
            | Error _ -> ()
            | Ok ds ->
              let band = Gdal.Dataset.get_band ds 1 |> Result.get_ok in
              (match Gdal.RasterBand.read_region Gdal.BA_float64 band
                ~x_off ~y_off ~x_size:bw ~y_size:bh ~buf_x:bw ~buf_y:bh with
              | Ok ba ->
                for i = 0 to bh - 1 do
                  for j = 0 to bw - 1 do
                    Array1.set amp_buf (i * bw + j) (Array2.get ba i j)
                  done
                done
              | Error _ -> ());
              Gdal.Dataset.close ds);

          (* Amplitude → dB and accumulate mosaic in one pass *)
          for px = 0 to npx - 1 do
            let amp = Array1.get amp_buf px in
            if Float.is_finite amp && amp > 0.0 then begin
              let db = 20.0 *. Float.log10 amp +. 50.0 in
              let scaled = db *. 200.0 in
              let v = Float.to_int (Float.min 32767.0 (Float.max 0.0 scaled)) in
              if v > 0 then begin
                any_valid := true;
                match data_source with
                | MPC ->
                  mosaic_sum.(px) <- mosaic_sum.(px) +. Float.of_int v;
                  mosaic_cnt.(px) <- mosaic_cnt.(px) + 1
                | AWS ->
                  result.(px) <- v
              end
            end
          done
        ) group.sg_tiles;

        if not !any_valid then None
        else begin
          (match data_source with
           | MPC ->
             for px = 0 to npx - 1 do
               if mosaic_cnt.(px) > 0 then
                 result.(px) <- Float.to_int (mosaic_sum.(px) /. Float.of_int mosaic_cnt.(px))
             done
           | AWS -> ());
          Some result
        end
      in

      let vv_out = mosaic_pol "_vv" in
      let vh_out = mosaic_pol "_vh" in
      match vv_out, vh_out with
      | None, None -> None
      | _ ->
        let vv = match vv_out with Some v -> v | None -> Array.make npx 0 in
        let vh = match vh_out with Some v -> v | None -> Array.make npx 0 in
        Some (group.sg_orbit, doy, vv, vh)
    in

    let group_results = Array.map (fun g ->
      process_group g
    ) (Array.of_list s1_groups) in

    (* Partition into ascending/descending *)
    let asc_list = ref [] in
    let desc_list = ref [] in
    Array.iter (fun opt -> match opt with
      | Some (orbit, doy, vv, vh) ->
        if orbit = "ascending" then asc_list := (doy, vv, vh) :: !asc_list
        else desc_list := (doy, vv, vh) :: !desc_list
      | None -> ()
    ) group_results;

    let build_flat_4d lst =
      let items = Array.of_list (List.rev lst) in
      let n = Array.length items in
      if n = 0 then (empty_flat_4d, [||])
      else begin
        let total = n * npx * 2 in
        let data = Array1.create int16_unsigned c_layout total in
        let doys = Array.make n 0 in
        Array.iteri (fun d (doy, vv, vh) ->
          doys.(d) <- doy;
          let d_base = d * npx * 2 in
          for px = 0 to npx - 1 do
            Array1.set data (d_base + px * 2) vv.(px);
            Array1.set data (d_base + px * 2 + 1) vh.(px)
          done
        ) items;
        ({ data; n; h = bh; w = bw; c = 2 }, doys)
      end
    in
    let (asc_data, asc_doys) = build_flat_4d !asc_list in
    let (desc_data, desc_doys) = build_flat_4d !desc_list in
    (asc_data, asc_doys, desc_data, desc_doys)
  end

(* ======================== Flat sampling (zero-allocation hot path) ======================== *)

(** Like stratified_select but reads from buf[0..n_valid-1] instead of a full array.
    Returns an array of sample_size selected indices. *)
let stratified_select_sub buf n_valid sample_size offset =
  if n_valid = 0 then
    Array.make sample_size 0
  else if n_valid <= sample_size then begin
    let result = Array.init sample_size (fun i -> buf.(i mod n_valid)) in
    Array.sort compare result;
    result
  end else begin
    let bins = Array.init (sample_size + 1) (fun i ->
      Float.of_int n_valid *. Float.of_int i /. Float.of_int sample_size) in
    Array.init sample_size (fun i ->
      let lo = Float.to_int bins.(i) in
      let hi = Float.to_int bins.(i + 1) in
      let hi = if hi <= lo then lo + 1 else hi in
      let pos = lo + (offset mod (hi - lo)) in
      buf.(min pos (n_valid - 1)))
  end

(** Sample S2 data for one pixel, writing directly into the ONNX input bigarray.
    Reads from flat source bigarrays; no per-pixel allocations. *)
let sample_s2_flat (s2b : flat_4d) (s2m : flat_3d_int) s2_doys
    ~i ~j ~pass ~(dst : (float, float32_elt, c_layout) Array1.t) ~dst_offset =
  let n_dates = s2m.n3 in
  let valid_buf = Array.make n_dates 0 in
  let n_valid = ref 0 in
  for d = 0 to n_dates - 1 do
    if Array1.get s2m.idata ((d * s2m.h3 + i) * s2m.w3 + j) > 0 then begin
      valid_buf.(!n_valid) <- d;
      incr n_valid
    end
  done;
  let vn = if !n_valid = 0 then
    (for d = 0 to n_dates - 1 do valid_buf.(d) <- d done; n_dates)
  else !n_valid in
  let idx = stratified_select_sub valid_buf vn sample_size_s2 pass in
  for t = 0 to sample_size_s2 - 1 do
    let ti = idx.(t) in
    let src_base = ((ti * s2b.h + i) * s2b.w + j) * s2b.c in
    let dst_base = dst_offset + t * 11 in
    for b = 0 to 9 do
      let v = Float.of_int (Array1.get s2b.data (src_base + b)) in
      Array1.set dst (dst_base + b)
        ((v -. s2_band_mean.(b)) /. (s2_band_std.(b) +. 1e-9))
    done;
    Array1.set dst (dst_base + 10) (Float.of_int s2_doys.(ti))
  done

(** Sample S1 data for one pixel, writing directly into the ONNX input bigarray.
    Merges ascending + descending, reads from flat source bigarrays. *)
let sample_s1_flat (s1_asc : flat_4d) s1_asc_doy (s1_desc : flat_4d) s1_desc_doy
    ~i ~j ~pass ~(dst : (float, float32_elt, c_layout) Array1.t) ~dst_offset =
  let n_asc = s1_asc.n in
  let n_desc = s1_desc.n in
  let n_total = n_asc + n_desc in
  let valid_buf = Array.make n_total 0 in
  let n_valid = ref 0 in
  (* Check ascending *)
  for d = 0 to n_asc - 1 do
    let base = ((d * s1_asc.h + i) * s1_asc.w + j) * s1_asc.c in
    let vv = Array1.get s1_asc.data base in
    let vh = Array1.get s1_asc.data (base + 1) in
    if vv <> 0 || vh <> 0 then begin
      valid_buf.(!n_valid) <- d;
      incr n_valid
    end
  done;
  (* Check descending — indices offset by n_asc *)
  for d = 0 to n_desc - 1 do
    let base = ((d * s1_desc.h + i) * s1_desc.w + j) * s1_desc.c in
    let vv = Array1.get s1_desc.data base in
    let vh = Array1.get s1_desc.data (base + 1) in
    if vv <> 0 || vh <> 0 then begin
      valid_buf.(!n_valid) <- n_asc + d;
      incr n_valid
    end
  done;
  let vn = if !n_valid = 0 then
    (for k = 0 to n_total - 1 do valid_buf.(k) <- k done; n_total)
  else !n_valid in
  let idx = stratified_select_sub valid_buf vn sample_size_s1 pass in
  for t = 0 to sample_size_s1 - 1 do
    let ti = idx.(t) in
    let dst_base = dst_offset + t * 3 in
    (* Determine which array and local index *)
    if ti < n_asc then begin
      let src_base = ((ti * s1_asc.h + i) * s1_asc.w + j) * s1_asc.c in
      for b = 0 to 1 do
        let v = Float.of_int (Array1.get s1_asc.data (src_base + b)) in
        Array1.set dst (dst_base + b)
          ((v -. s1_band_mean.(b)) /. (s1_band_std.(b) +. 1e-9))
      done;
      Array1.set dst (dst_base + 2) (Float.of_int s1_asc_doy.(ti))
    end else begin
      let d = ti - n_asc in
      let src_base = ((d * s1_desc.h + i) * s1_desc.w + j) * s1_desc.c in
      for b = 0 to 1 do
        let v = Float.of_int (Array1.get s1_desc.data (src_base + b)) in
        Array1.set dst (dst_base + b)
          ((v -. s1_band_mean.(b)) /. (s1_band_std.(b) +. 1e-9))
      done;
      Array1.set dst (dst_base + 2) (Float.of_int s1_desc_doy.(d))
    end
  done

(* ======================== Quantisation ======================== *)

let quantize_symmetric x_f32 =
  let n = Array.length x_f32 in
  let abs_max = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0.0 x_f32 in
  let scale = Float.max (abs_max /. 127.0) 1e-8 in
  let quantized = Array.init n (fun i ->
    let v = Float.round (x_f32.(i) /. scale) in
    let v = Float.max (-128.0) (Float.min 127.0 v) in
    Float.to_int v) in
  (quantized, scale)

(* ======================== Block loading ======================== *)

(** Load one block's S2/S1 data and scan valid pixels.
    Returns (valid_pixels, s2_bands, s2_masks, s2_doys,
             s1_asc, s1_asc_doy, s1_desc, s1_desc_doy).
    The heavy I/O work is done here; sampling happens during inference. *)
let load_block ~band_cache_dir ~valid_dates ~items_arr ~data_source
    ~s1_groups ~s1_cache_dir ~roi_w
    ~water_mask
    ~x_off ~y_off ~bw ~bh =
  (* Read S2 block *)
  let (s2_bands_blk, s2_masks_blk, s2_doys_blk) =
    read_s2_block ~band_cache_dir ~valid_dates
      ~items_arr ~data_source ~x_off ~y_off ~bw ~bh ~roi_w in

  (* Read S1 block *)
  let (s1_asc_blk, s1_asc_doy_blk, s1_desc_blk, s1_desc_doy_blk) =
    read_s1_block ~s1_groups ~s1_cache_dir
      ~x_off ~y_off ~bw ~bh ~data_source in

  (* Find valid pixels — date-first iteration for cache-friendly access *)
  let s1_asc_empty = s1_asc_blk.n = 0 in
  let s1_desc_empty = s1_desc_blk.n = 0 in
  let total_px = bh * bw in

  (* S2: count valid dates per pixel *)
  let s2_valid_count = Array.make total_px 0 in
  for d = 0 to s2_masks_blk.n3 - 1 do
    let mask_base = d * total_px in
    for px = 0 to total_px - 1 do
      if Array1.get s2_masks_blk.idata (mask_base + px) > 0 then
        s2_valid_count.(px) <- s2_valid_count.(px) + 1
    done
  done;

  (* S2: check if any band in any date is nonzero *)
  let s2_has_nonzero = Array.make total_px false in
  let n_marked = ref 0 in
  let d = ref 0 in
  while !d < s2_bands_blk.n && !n_marked < total_px do
    let band_base = !d * total_px * 10 in
    for px = 0 to total_px - 1 do
      if not s2_has_nonzero.(px) then begin
        let base = band_base + px * 10 in
        let found = ref false in
        for bb = 0 to 9 do
          if Array1.get s2_bands_blk.data (base + bb) <> 0 then found := true
        done;
        if !found then begin
          s2_has_nonzero.(px) <- true;
          incr n_marked
        end
      end
    done;
    incr d
  done;

  (* S1: count valid dates per pixel *)
  let s1_asc_count = Array.make total_px 0 in
  if not s1_asc_empty then
    for d = 0 to s1_asc_blk.n - 1 do
      let band_base = d * total_px * 2 in
      for px = 0 to total_px - 1 do
        let base = band_base + px * 2 in
        if Array1.get s1_asc_blk.data base <> 0 ||
           Array1.get s1_asc_blk.data (base + 1) <> 0 then
          s1_asc_count.(px) <- s1_asc_count.(px) + 1
      done
    done;

  let s1_desc_count = Array.make total_px 0 in
  if not s1_desc_empty then
    for d = 0 to s1_desc_blk.n - 1 do
      let band_base = d * total_px * 2 in
      for px = 0 to total_px - 1 do
        let base = band_base + px * 2 in
        if Array1.get s1_desc_blk.data base <> 0 ||
           Array1.get s1_desc_blk.data (base + 1) <> 0 then
          s1_desc_count.(px) <- s1_desc_count.(px) + 1
      done
    done;

  (* Build valid pixel list *)
  let valid_pixels_list = ref [] in
  for px = 0 to total_px - 1 do
    let i = px / bw and j = px mod bw in
    let global_idx = (y_off + i) * roi_w + (x_off + j) in
    let is_water = water_mask.(global_idx) in
    let s1_total = s1_asc_count.(px) + s1_desc_count.(px) in
    let valid = not is_water &&
      if s1_asc_empty && s1_desc_empty then
        s2_has_nonzero.(px) && s2_valid_count.(px) >= min_valid_timesteps
      else
        s2_has_nonzero.(px) && s2_valid_count.(px) >= min_valid_timesteps && s1_total >= min_valid_timesteps
    in
    if valid then
      valid_pixels_list := (i, j) :: !valid_pixels_list
  done;
  let valid_pixels = Array.of_list (List.rev !valid_pixels_list) in

  (* Create effective S1 arrays (substitute dummies if empty) *)
  let make_dummy_s1 () =
    let total = 1 * bh * bw * 2 in
    let data = Array1.create int16_unsigned c_layout total in
    Array1.fill data 0;
    { data; n = 1; h = bh; w = bw; c = 2 }
  in
  let s1_asc_eff = if s1_asc_empty then make_dummy_s1 () else s1_asc_blk in
  let s1_asc_doy_eff = if s1_asc_empty then [| 180 |] else s1_asc_doy_blk in
  let s1_desc_eff = if s1_desc_empty then make_dummy_s1 () else s1_desc_blk in
  let s1_desc_doy_eff = if s1_desc_empty then [| 180 |] else s1_desc_doy_blk in

  (valid_pixels, s2_bands_blk, s2_masks_blk, s2_doys_blk,
   s1_asc_eff, s1_asc_doy_eff, s1_desc_eff, s1_desc_doy_eff)

(* ======================== Block inference ======================== *)

(** Run ONNX inference on a loaded block's data. Samples inline per batch.
    Writes results to the global output bigarrays. Called while holding GPU mutex. *)
let run_block_inference session ~valid_pixels
    ~s2_bands ~s2_masks ~s2_doys
    ~s1_asc ~s1_asc_doy ~s1_desc ~s1_desc_doy
    ~repeat_times ~batch_size =
  let n_valid = Array.length valid_pixels in
  let sum_z = Array.init n_valid (fun _ -> Array.make latent_dim 0.0) in
  let n_batches = (n_valid + batch_size - 1) / batch_size in

  (* Pre-allocate ONNX input tensors *)
  let s2_input = Array1.create float32 c_layout (batch_size * sample_size_s2 * 11) in
  let s1_input = Array1.create float32 c_layout (batch_size * sample_size_s1 * 3) in

  for pass = 0 to repeat_times - 1 do
    for batch_idx = 0 to n_batches - 1 do
      let start_px = batch_idx * batch_size in
      let actual_b = min batch_size (n_valid - start_px) in

      Array1.fill s2_input 0.0;
      Array1.fill s1_input 0.0;

      for px_in_batch = 0 to actual_b - 1 do
        let px = start_px + px_in_batch in
        let (i, j) = valid_pixels.(px) in

        sample_s2_flat s2_bands s2_masks s2_doys
          ~i ~j ~pass ~dst:s2_input
          ~dst_offset:(px_in_batch * sample_size_s2 * 11);

        sample_s1_flat s1_asc s1_asc_doy s1_desc s1_desc_doy
          ~i ~j ~pass ~dst:s1_input
          ~dst_offset:(px_in_batch * sample_size_s1 * 3)
      done;

      let s2_ba = if actual_b = batch_size then s2_input
        else Array1.sub s2_input 0 (actual_b * sample_size_s2 * 11) in
      let s1_ba = if actual_b = batch_size then s1_input
        else Array1.sub s1_input 0 (actual_b * sample_size_s1 * 3) in

      let outputs = Onnxruntime.Session.run_cached_ba session
        [|(s2_ba,
           [| Int64.of_int actual_b; Int64.of_int sample_size_s2; 11L |]);
          (s1_ba,
           [| Int64.of_int actual_b; Int64.of_int sample_size_s1; 3L |])|]
        ~output_sizes:[| actual_b * latent_dim |]
      in
      let output_ba = outputs.(0) in

      for px_in_batch = 0 to actual_b - 1 do
        let px = start_px + px_in_batch in
        let out_offset = px_in_batch * latent_dim in
        for k = 0 to latent_dim - 1 do
          sum_z.(px).(k) <- sum_z.(px).(k) +. Array1.get output_ba (out_offset + k)
        done
      done
    done
  done;

  (* Keep input tensors alive through inference *)
  ignore (Sys.opaque_identity s2_input);
  ignore (Sys.opaque_identity s1_input);

  (* Average, then quantize or keep as float depending on mode *)
  let r_f = Float.of_int repeat_times in
  let avgs = Array.init n_valid (fun px ->
    Array.init latent_dim (fun k -> sum_z.(px).(k) /. r_f)
  ) in
  (avgs, sum_z)


(* ======================== OPERA RTC-S1 (AWS) ======================== *)

(** Read the NASA Earthdata bearer token from ~/.edl_bearer_token. *)
let read_edl_token () =
  let token_path = Filename.concat (Sys.getenv "HOME") ".edl_bearer_token" in
  let ic = open_in token_path in
  let token = String.trim (input_line ic) in
  close_in ic;
  token

(** Set up GDAL HTTP auth for OPERA COG access via cumulus.
    Creates a header file with Bearer token and sets GDAL_HTTP_HEADER_FILE. *)
let setup_opera_gdal_auth token =
  let header_path = Filename.concat (Filename.get_temp_dir_name ()) "gdal_opera_headers.txt" in
  let oc = open_out header_path in
  Printf.fprintf oc "Authorization: Bearer %s\nCookie: asf-urs=%s\n" token token;
  close_out oc;
  Gdal.set_config_option "GDAL_HTTP_HEADER_FILE" header_path;
  eprintf "  GDAL OPERA auth configured\n%!"

type opera_granule = {
  og_date : string;
  og_vv_url : string;
  og_vh_url : string;
}

(** Search CMR for OPERA RTC-S1 granules. *)
let search_cmr_opera ~(client : Stac_client.t)
    ~bbox:(xmin, ymin, xmax, ymax) ~datetime =
  let temporal =
    match String.split_on_char '/' datetime with
    | [s; e] ->
      let fix d = if String.length d = 10 then d ^ "T00:00:00Z" else d in
      fix s ^ "," ^ fix e
    | _ -> datetime ^ "," ^ datetime
  in
  let bbox_str = Printf.sprintf "%.6f,%.6f,%.6f,%.6f" xmin ymin xmax ymax in
  let rec fetch_all page_num acc =
    let url = Printf.sprintf
      "%s?collection_concept_id=%s&bounding_box=%s&temporal=%s&page_size=2000&page_num=%d"
      cmr_url opera_collection_id bbox_str temporal page_num
    in
    let code, body = Stac_client.http_get client url in
    if code <> 200 then
      failwith (Printf.sprintf "CMR search failed (%d): %s" code body);
    let json = Yojson.Safe.from_string body in
    let entries = match json with
      | `Assoc fields ->
        (match List.assoc_opt "feed" fields with
         | Some (`Assoc feed_fields) ->
           (match List.assoc_opt "entry" feed_fields with
            | Some (`List entries) -> entries
            | _ -> [])
         | _ -> [])
      | _ -> []
    in
    if entries = [] then List.rev acc
    else begin
      let granules = List.filter_map (fun entry ->
        match entry with
        | `Assoc fields ->
          let time_start = match List.assoc_opt "time_start" fields with
            | Some (`String s) -> date_of_datetime s
            | _ -> ""
          in
          if time_start = "" then None
          else begin
            let title = match List.assoc_opt "title" fields with
              | Some (`String s) -> s
              | _ -> ""
            in
            if title = "" then begin
              eprintf "  Warning: skipping CMR entry without title\n%!";
              None
            end else
              let base = Printf.sprintf
                "https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/%s/%s"
                title title in
              Some { og_date = time_start;
                     og_vv_url = base ^ "_VV.tif";
                     og_vh_url = base ^ "_VH.tif" }
          end
        | _ -> None
      ) entries in
      let acc = List.rev_append granules acc in
      if List.length entries < 2000 then List.rev acc
      else fetch_all (page_num + 1) acc
    end
  in
  fetch_all 1 []

(** Read orbit direction from a GDAL dataset's metadata tags. *)
let read_orbit_direction_from_cog url =
  match Gdal.Dataset.open_ex ~thread_safe:true (gdal_vsi_path url) with
  | Error _ ->
    eprintf "  Warning: could not open COG for orbit metadata\n%!";
    "unknown"
  | Ok ds ->
    let result =
      match Gdal.Dataset.get_metadata_item ds ~key:"ORBIT_PASS_DIRECTION" ~domain:"" with
      | Some s -> String.lowercase_ascii s
      | None -> "unknown"
    in
    Gdal.Dataset.close ds;
    result

(** Prepare Sentinel-1 groups from MPC STAC items.
    Signs each item and extracts VV/VH asset hrefs. *)
let prepare_s1_mpc ~(client : Stac_client.t) items =
  if items = [] then []
  else begin
    let by_date_orbit = Hashtbl.create 64 in
    List.iter (fun (item : Stac_client.item) ->
      let dt = date_of_datetime (match Stac_client.get_datetime item with Some d -> d | None -> "") in
      let orbit = match Stac_client.get_string_prop item "sat:orbit_state" with
        | Some o -> o | None -> "unknown" in
      let key = (dt, orbit) in
      let prev = try Hashtbl.find by_date_orbit key with Not_found -> [] in
      Hashtbl.replace by_date_orbit key (item :: prev)
    ) items;
    let sorted_keys = List.sort compare (Hashtbl.fold (fun k _ acc -> k :: acc) by_date_orbit []) in
    List.map (fun (date_str, orbit) ->
      let group_items = List.rev (Hashtbl.find by_date_orbit (date_str, orbit)) in
      let tiles = List.map (fun (item : Stac_client.item) ->
        let signed = Stac_client.sign_planetary_computer client item in
        { st_id = item.Stac_client.id;
          st_vv = get_asset_href signed "vv";
          st_vh = get_asset_href signed "vh" }
      ) group_items in
      { sg_date = date_str; sg_orbit = orbit; sg_tiles = tiles }
    ) sorted_keys
  end

(** Prepare Sentinel-1 groups from OPERA RTC-S1 via CMR.
    GDAL auth must be configured before calling this. *)
let prepare_s1_opera ~(client : Stac_client.t) ~bbox ~datetime =
  eprintf "  Searching CMR for OPERA RTC-S1...\n%!";
  let granules = search_cmr_opera ~client ~bbox ~datetime in
  eprintf "  Found %d OPERA granules\n%!" (List.length granules);
  if granules = [] then []
  else begin
    let by_date = Hashtbl.create 64 in
    List.iter (fun g ->
      let prev = try Hashtbl.find by_date g.og_date with Not_found -> [] in
      Hashtbl.replace by_date g.og_date (g :: prev)
    ) granules;
    let sorted_dates = List.sort String.compare
      (Hashtbl.fold (fun k _ acc -> k :: acc) by_date []) in
    let groups = ref [] in
    List.iter (fun date_str ->
      let day_granules = List.rev (Hashtbl.find by_date date_str) in
      let by_orbit = Hashtbl.create 4 in
      List.iter (fun g ->
        let orbit = read_orbit_direction_from_cog g.og_vv_url in
        let prev = try Hashtbl.find by_orbit orbit with Not_found -> [] in
        Hashtbl.replace by_orbit orbit (g :: prev)
      ) day_granules;
      Hashtbl.iter (fun orbit orbit_granules ->
        let tiles = List.map (fun g ->
          { st_id = Filename.basename g.og_vv_url |> Filename.remove_extension;
            st_vv = Some g.og_vv_url;
            st_vh = Some g.og_vh_url }
        ) orbit_granules in
        groups := { sg_date = date_str; sg_orbit = orbit; sg_tiles = tiles } :: !groups
      ) by_orbit
    ) sorted_dates;
    List.rev !groups
  end

(* ======================== GeoTIFF output ======================== *)

(** Create an empty GeoTIFF with georeferencing. Closes immediately so the
    file exists on disk; subsequent writes reopen in Update mode. *)
type output_format = GeoTIFF | Zarr
type quant_mode = Int8Scale | Float16 | Float32

let create_output ~fmt ~path ~(roi : roi) ~h ~w ~bands dtype =
  let (origin_x, pixel_w, origin_y, pixel_h) = roi.geotransform in
  let (drv_name, opts) = match fmt with
    | GeoTIFF -> ("GTiff", ["COMPRESS=DEFLATE"; "TILED=YES"; "BLOCKXSIZE=256";
                             "BLOCKYSIZE=256"; "BIGTIFF=YES"])
    | Zarr ->
      let bs = if bands = 1 then "BLOCKSIZE=256,256" else "BLOCKSIZE=1,256,256" in
      ("Zarr", ["FORMAT=ZARR_V3"; "COMPRESS=BLOSC"; "BLOSC_CNAME=zstd"; bs])
  in
  let drv = Gdal.Driver.by_name drv_name |> Result.get_ok in
  let ds = Gdal.Driver.create drv ~filename:path ~width:w ~height:h
             ~bands ~options:opts dtype |> Result.get_ok in
  let gt : Gdal.geo_transform = {
    origin_x; pixel_width = pixel_w; row_rotation = 0.0;
    origin_y; col_rotation = 0.0; pixel_height = pixel_h
  } in
  ignore (Gdal.Dataset.set_geo_transform ds gt);
  ignore (Gdal.Dataset.set_projection ds roi.crs_wkt);
  Gdal.Dataset.close ds

(** Write one block of int8 embeddings: open in Update mode, write, close. *)
let write_embeddings_block_int8 ~path ~x_off ~y_off ~bw ~bh
    (block_emb : int array array) (valid_pixels : (int * int) array) =
  let ds = Gdal.Dataset.open_ ~access:Update path |> Result.get_ok in
  for b = 0 to latent_dim - 1 do
    let band = Gdal.Dataset.get_band ds (b + 1) |> Result.get_ok in
    let ba = Array2.create int8_unsigned c_layout bh bw in
    Array2.fill ba 0;
    Array.iteri (fun px (i, j) ->
      Array2.set ba i j (block_emb.(px).(b) land 0xFF)
    ) valid_pixels;
    ignore (Gdal.RasterBand.write_region Gdal.BA_byte band
              ~x_off ~y_off ~x_size:bw ~y_size:bh ba)
  done;
  Gdal.Dataset.close ds

(** Write one block of float16 embeddings: write float32 bigarrays,
    GDAL converts to float16 on disk. *)
let write_embeddings_block_f16 ~path ~x_off ~y_off ~bw ~bh
    (block_emb : float array array) (valid_pixels : (int * int) array) =
  let ds = Gdal.Dataset.open_ ~access:Update path |> Result.get_ok in
  for b = 0 to latent_dim - 1 do
    let band = Gdal.Dataset.get_band ds (b + 1) |> Result.get_ok in
    let ba = Array2.create float32 c_layout bh bw in
    Array2.fill ba 0.0;
    Array.iteri (fun px (i, j) ->
      Array2.set ba i j block_emb.(px).(b)
    ) valid_pixels;
    ignore (Gdal.RasterBand.write_region Gdal.BA_float32 band
              ~x_off ~y_off ~x_size:bw ~y_size:bh ba)
  done;
  Gdal.Dataset.close ds

(** Write one block of scales: open in Update mode, write, close. *)
let write_scales_block ~path ~x_off ~y_off ~bw ~bh
    (block_scales : float array) (valid_pixels : (int * int) array) =
  let ds = Gdal.Dataset.open_ ~access:Update path |> Result.get_ok in
  let band = Gdal.Dataset.get_band ds 1 |> Result.get_ok in
  let ba = Array2.create float32 c_layout bh bw in
  Array2.fill ba 0.0;
  Array.iteri (fun px (i, j) ->
    Array2.set ba i j block_scales.(px)
  ) valid_pixels;
  ignore (Gdal.RasterBand.write_region Gdal.BA_float32 band
            ~x_off ~y_off ~x_size:bw ~y_size:bh ba);
  Gdal.Dataset.close ds

(** Check if a block region has non-zero data in both TIFs. *)
let block_is_complete ~emb_path ~scale_path ~x_off ~y_off ~bw ~bh =
  let check_scales () =
    match Gdal.Dataset.open_ex scale_path with
    | Error _ -> false
    | Ok ds ->
      match Gdal.Dataset.get_band ds 1 with
      | Error _ -> Gdal.Dataset.close ds; false
      | Ok band ->
        match Gdal.RasterBand.read_region Gdal.BA_float32 band
                ~x_off ~y_off ~x_size:bw ~y_size:bh ~buf_x:bw ~buf_y:bh with
        | Error _ -> Gdal.Dataset.close ds; false
        | Ok ba ->
          let has_data = ref false in
          for i = 0 to bh - 1 do
            for j = 0 to bw - 1 do
              if Array2.get ba i j <> 0.0 then has_data := true
            done
          done;
          Gdal.Dataset.close ds;
          !has_data
  in
  let check_emb () =
    match Gdal.Dataset.open_ex emb_path with
    | Error _ -> false
    | Ok ds ->
      match Gdal.Dataset.get_band ds 1 with
      | Error _ -> Gdal.Dataset.close ds; false
      | Ok band ->
        match Gdal.RasterBand.read_region Gdal.BA_byte band
                ~x_off ~y_off ~x_size:bw ~y_size:bh ~buf_x:bw ~buf_y:bh with
        | Error _ -> Gdal.Dataset.close ds; false
        | Ok ba ->
          let has_data = ref false in
          for i = 0 to bh - 1 do
            for j = 0 to bw - 1 do
              if Array2.get ba i j <> 0 then has_data := true
            done
          done;
          Gdal.Dataset.close ds;
          !has_data
  in
  check_scales () && check_emb ()

(* ======================== Main pipeline ======================== *)

let () =
  (* CLI parsing *)
  let mgrs_tiles = ref [] in
  let model_path = ref "" in
  let output_dir = ref "" in
  let start_date = ref "2024-01-01" in
  let end_date = ref "2024-12-31" in
  let max_cloud = ref 100.0 in
  let num_threads = ref 1 in
  let batch_size = ref default_batch_size in
  let repeat_times = ref 1 in
  let cache_dir = ref "" in
  let cuda_device = ref (-1) in
  let data_source_str = ref "mpc" in
  let download_workers = ref 8 in
  let load_workers = ref 3 in
  let block_size = ref 1024 in
  let output_format_str = ref "geotiff" in
  let quantize_str = ref "int8" in

  let speclist = [
    ("--mgrs", Arg.String (fun id -> mgrs_tiles := id :: !mgrs_tiles), "MGRS tile ID (repeatable)");
    ("--model", Arg.Set_string model_path, "Path to ONNX model");
    ("--output", Arg.Set_string output_dir, "Output directory");
    ("--start", Arg.Set_string start_date, "Start date (YYYY-MM-DD)");
    ("--end", Arg.Set_string end_date, "End date (YYYY-MM-DD)");
    ("--max_cloud", Arg.Set_float max_cloud, "Max cloud cover %");
    ("--num_threads", Arg.Set_int num_threads, "CPU threads for ONNX");
    ("--batch_size", Arg.Set_int batch_size, "Inference batch size");
    ("--repeat_times", Arg.Set_int repeat_times, "Stratified sampling passes");
    ("--cache_dir", Arg.Set_string cache_dir, "Cache directory for downloaded COGs");
    ("--cuda", Arg.Set_int cuda_device, "CUDA device ID (e.g. 0) for GPU inference");
    ("--data_source", Arg.Set_string data_source_str, "Data source: mpc or aws (default: mpc)");
    ("--download_workers", Arg.Set_int download_workers, "Parallel GDAL download workers (default: 8)");
    ("--load_workers", Arg.Set_int load_workers, "Block prefetch workers for inference (default: 3)");
    ("--block_size", Arg.Set_int block_size, "Spatial block size for inference (default: 1024)");
    ("--format", Arg.Set_string output_format_str, "Output format: geotiff or zarr (default: geotiff)");
    ("--quantize", Arg.Set_string quantize_str, "Quantization: int8 or float16 (default: int8)");
  ] in
  Arg.parse speclist (fun _ -> ()) "Tessera MGRS tile pipeline";

  let data_source = match String.lowercase_ascii !data_source_str with
    | "mpc" -> MPC
    | "aws" -> AWS
    | s -> failwith (Printf.sprintf "Unknown data source: %s (expected mpc or aws)" s)
  in
  let out_fmt = match String.lowercase_ascii !output_format_str with
    | "geotiff" | "tif" -> GeoTIFF
    | "zarr" -> Zarr
    | s -> failwith (Printf.sprintf "Unknown format: %s (expected geotiff or zarr)" s)
  in
  let quant = match String.lowercase_ascii !quantize_str with
    | "int8" -> Int8Scale
    | "float16" | "f16" -> Float16
    | "float32" | "f32" -> Float32
    | s -> failwith (Printf.sprintf "Unknown quantize mode: %s (expected int8, float16 or float32)" s)
  in

  let mgrs_tiles = List.rev !mgrs_tiles in
  if mgrs_tiles = [] then
    failwith "--mgrs is required (provide one or more MGRS tile IDs)";
  if !cache_dir = "" then
    failwith "--cache_dir is required";

  let date_range = !start_date ^ "/" ^ !end_date in

  (* GDAL init *)
  Gdal.init ();

  (* Set GDAL COG optimization for AWS sources *)
  if data_source = AWS then begin
    Gdal.set_config_option "GDAL_DISABLE_READDIR_ON_OPEN" "EMPTY_DIR";
    Gdal.set_config_option "GDAL_HTTP_MULTIRANGE" "YES";
    Gdal.set_config_option "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES" "YES";
  end;

  let n_workers = !download_workers in
  let client = Stac_client.make () in
  let blk = !block_size in

  (* Load ONNX model once for all tiles *)
  Printf.printf "Loading model: %s\n%!" !model_path;
  let env = Onnxruntime.Env.create ~log_level:3 "tessera" in
  let cuda_opt = if !cuda_device >= 0 then Some !cuda_device else None in
  let session = Onnxruntime.Session.create env ~threads:!num_threads ?cuda_device:cuda_opt !model_path in
  if !cuda_device >= 0 then
    Printf.printf "  Using CUDA device %d\n%!" !cuda_device;
  Printf.printf "  Model loaded successfully\n%!";

  (* Process each MGRS tile *)
  List.iter (fun tile_id ->
    let t_tile_start = Unix.gettimeofday () in
    Printf.printf "\n%s\nProcessing MGRS tile: %s\n%s\n%!"
      (String.make 60 '=') tile_id (String.make 60 '=');

    (* Skip if output exists *)
    mkdir_p !output_dir;
    let ext = match out_fmt with GeoTIFF -> ".tif" | Zarr -> ".zarr" in
    let emb_tif_path = Filename.concat !output_dir (tile_id ^ ext) in
    let scale_tif_path = Filename.concat !output_dir (tile_id ^ "_scales" ^ ext) in
    begin

    (* ---- Step 1: Derive ROI from MGRS tile ---- *)
    Printf.printf "\nStep 1: Derive ROI\n%!";
    let roi = mgrs_tile_roi ~client ~data_source tile_id in
    let h = roi.height and w = roi.width in
    let (xmin, ymin, xmax, ymax) = roi.bbox in

    let tile_cache = Filename.concat !cache_dir tile_id in
    let scl_cache = Filename.concat tile_cache "scl" in
    let band_cache = Filename.concat tile_cache "bands" in
    let s1_cache = Filename.concat !cache_dir "s1" in
    mkdir_p scl_cache;
    mkdir_p band_cache;
    mkdir_p s1_cache;

    (* ---- Step 2: Search S2 and download SCL ---- *)
    Printf.printf "\nStep 2: Search S2 and download SCL\n%!";
    let s2_base_url = match data_source with MPC -> stac_url | AWS -> aws_stac_url in
    let s2_query = match data_source with
      | MPC -> `Assoc [
          "eo:cloud_cover", `Assoc ["lt", `Float !max_cloud];
          "s2:mgrs_tile", `Assoc ["eq", `String tile_id]
        ]
      | AWS -> `Assoc [
          "eo:cloud_cover", `Assoc ["lt", `Float !max_cloud];
          "grid:code", `Assoc ["eq", `String (Printf.sprintf "MGRS-%s" tile_id)]
        ]
    in
    let s2_params : Stac_client.search_params = {
      collections = ["sentinel-2-l2a"];
      bbox = [xmin; ymin; xmax; ymax];
      datetime = date_range;
      query = Some s2_query;
      limit = None;
    } in
    let s2_items = Stac_client.search client ~base_url:s2_base_url s2_params in
    Printf.printf "  Found %d S2 scenes\n%!" (List.length s2_items);

    let items_arr = Array.of_list s2_items in
    let item_dates = Array.map (fun (it : Stac_client.item) ->
      date_of_datetime (match Stac_client.get_datetime it with Some d -> d | None -> "")
    ) items_arr in

    (* Pre-sign all items *)
    let signed_items = Array.map (fun item -> match data_source with
      | MPC -> Stac_client.sign_planetary_computer client item
      | AWS -> item) items_arr in

    (* Download all SCL bands in parallel *)
    Printf.printf "  Downloading SCL bands...\n%!";
    let n_items = Array.length items_arr in
    let _scl_dl = parallel_map ~n_workers (fun i ->
      let access_item = signed_items.(i) in
      match get_asset_href access_item (scl_asset_name data_source) with
      | None -> ()
      | Some href ->
        let cache_path = Filename.concat scl_cache
          (Printf.sprintf "%s.tif" items_arr.(i).id) in
        download_cog ~cache_path href
    ) (Array.init n_items Fun.id) in

    (* ---- Step 3: Analyze SCL and filter dates ---- *)
    Printf.printf "\nStep 3: Analyze SCL\n%!";
    let (valid_dates, water_mask) = analyze_scl ~scl_cache_dir:scl_cache
      ~roi_h:h ~roi_w:w items_arr item_dates in
    Printf.printf "  %d valid dates pass cloud filter\n%!" (List.length valid_dates);

    if valid_dates = [] then
      Printf.printf "  No valid S2 dates. Skipping tile.\n%!"
    else begin

    (* ---- Step 4: Download spectral bands for valid dates ---- *)
    Printf.printf "\nStep 4: Download spectral bands (%d dates)\n%!" (List.length valid_dates);
    let band_names = s2_band_names data_source in
    let _band_dl = parallel_map ~n_workers (fun (di : date_info) ->
      List.iter (fun item_idx ->
        let access_item = signed_items.(item_idx) in
        let item_dir = Filename.concat band_cache items_arr.(item_idx).Stac_client.id in
        mkdir_p item_dir;
        Array.iter (fun bn ->
          match get_asset_href access_item bn with
          | None -> ()
          | Some href ->
            let cache_path = Filename.concat item_dir (bn ^ ".tif") in
            download_cog ~cache_path href
        ) band_names
      ) di.di_item_indices
    ) (Array.of_list valid_dates) in
    Printf.printf "  Band download complete\n%!";

    (* ---- Step 5: Search and prepare S1 ---- *)
    Printf.printf "\nStep 5: Search S1\n%!";
    let s1_groups = match data_source with
      | MPC ->
        let s1_params : Stac_client.search_params = {
          collections = ["sentinel-1-rtc"];
          bbox = [xmin; ymin; xmax; ymax];
          datetime = date_range;
          query = None;
          limit = None;
        } in
        let s1_items = Stac_client.search client ~base_url:stac_url s1_params in
        Printf.printf "  Found %d S1 scenes\n%!" (List.length s1_items);
        prepare_s1_mpc ~client s1_items
      | AWS ->
        let token = read_edl_token () in
        setup_opera_gdal_auth token;
        prepare_s1_opera ~client
          ~bbox:(xmin, ymin, xmax, ymax) ~datetime:date_range
    in
    Printf.printf "  %d S1 groups\n%!" (List.length s1_groups);

    (* ---- Step 5b: Pre-warp S1 to tile extent and cache ---- *)
    Printf.printf "  Warping S1 to tile extent...\n%!";
    let resampling = match data_source with MPC -> "near" | AWS -> "bilinear" in
    let (tile_left, tile_bottom, tile_right, tile_top) = roi.bounds in
    let tile_bounds = (tile_left, tile_bottom, tile_right, tile_top) in
    let s1_warp_count = ref 0 in
    let s1_warp_total = List.fold_left (fun acc g ->
      List.fold_left (fun acc2 tile ->
        acc2 + (if tile.st_vv <> None then 1 else 0)
             + (if tile.st_vh <> None then 1 else 0)
      ) acc g.sg_tiles
    ) 0 s1_groups in
    let do_warp_s1 url cache_path =
      if Sys.file_exists cache_path then begin
        incr s1_warp_count;
        if !s1_warp_count mod 20 = 0 then
          Printf.printf "    [%d/%d] cached\n%!" !s1_warp_count s1_warp_total
      end else begin
        (match warp_to_mem ~resampling ~dst_crs_wkt:roi.crs_wkt
                ~bounds:tile_bounds ~width:w ~height:h url with
        | Error msg ->
          incr s1_warp_count;
          eprintf "    Warning: S1 warp failed for %s: %s\n%!" (Filename.basename url) msg
        | Ok warped_ds ->
          (* Write as tiled GeoTIFF for efficient block reads *)
          (match Gdal.Dataset.translate warped_ds ~dst_filename:cache_path
             ["-co"; "TILED=YES"; "-co"; "BLOCKXSIZE=512"; "-co"; "BLOCKYSIZE=512"] with
           | Ok copy -> Gdal.Dataset.close copy
           | Error msg -> eprintf "    Warning: S1 cache write failed: %s\n%!" msg);
          Gdal.Dataset.close warped_ds;
          incr s1_warp_count;
          if !s1_warp_count mod 20 = 0 then
            Printf.printf "    [%d/%d] warped\n%!" !s1_warp_count s1_warp_total)
      end
    in
    let s1_warp_jobs = List.concat_map (fun g ->
      List.concat_map (fun tile ->
        let vv_job = match tile.st_vv with
          | Some url -> [(url, Filename.concat s1_cache (tile.st_id ^ "_vv.tif"))]
          | None -> []
        in
        let vh_job = match tile.st_vh with
          | Some url -> [(url, Filename.concat s1_cache (tile.st_id ^ "_vh.tif"))]
          | None -> []
        in
        vv_job @ vh_job
      ) g.sg_tiles
    ) s1_groups in
    parallel_map ~n_workers (fun (url, cache_path) ->
      do_warp_s1 url cache_path
    ) (Array.of_list s1_warp_jobs) |> ignore;
    Printf.printf "  S1 warp complete: %d/%d\n%!" !s1_warp_count s1_warp_total;

    let t_download = Unix.gettimeofday () in
    Printf.printf "\nDownload complete: %.1fs\n%!" (t_download -. t_tile_start);

    (* ---- Step 6: Block-based inference with worker pool ---- *)
    Printf.printf "\n%s\nStep 6: Block-based inference (block_size=%d, workers=%d)\n%s\n%!"
      (String.make 60 '-') blk !load_workers (String.make 60 '-');

    (* Create output files if they don't exist *)
    let emb_dtype = match quant with Int8Scale -> Gdal.Byte | Float16 -> Gdal.Float16 | Float32 -> Gdal.Float32 in
    if not (Sys.file_exists emb_tif_path) then
      create_output ~fmt:out_fmt ~path:emb_tif_path ~roi ~h ~w ~bands:latent_dim emb_dtype;
    if quant = Int8Scale && not (Sys.file_exists scale_tif_path) then
      create_output ~fmt:out_fmt ~path:scale_tif_path ~roi ~h ~w ~bands:1 Gdal.Float32;
    let tif_mu = Mutex.create () in

    let n_rows = (h + blk - 1) / blk in
    let n_cols = (w + blk - 1) / blk in
    let total_blocks = n_rows * n_cols in

    let r = !repeat_times in
    let b = !batch_size in

    (* Build the ordered list of blocks *)
    let blocks =
      let acc = ref [] in
      let num = ref 0 in
      for bi = 0 to n_rows - 1 do
        for bj = 0 to n_cols - 1 do
          incr num;
          let y_off = bi * blk in
          let x_off = bj * blk in
          let bh = min blk (h - y_off) in
          let bw = min blk (w - x_off) in
          acc := (!num, y_off, x_off, bh, bw) :: !acc
        done
      done;
      Array.of_list (List.rev !acc)
    in

    (* Worker pool: each domain grabs blocks, loads+preprocesses, then
       acquires GPU mutex for inference. Ramps up: starts with 1 worker,
       spawns the rest once first inference begins. *)
    let gpu_mu = Mutex.create () in
    let next_block = Atomic.make 0 in
    let n_blocks = Array.length blocks in
    let inference_started = Atomic.make false in

    let worker () =
      let rec loop () =
        let idx = Atomic.fetch_and_add next_block 1 in
        if idx < n_blocks then begin
          let (block_num, y_off, x_off, bh, bw) = blocks.(idx) in

          (* Resume: skip blocks that already have data *)
          let is_cached = match quant with
            | Int8Scale ->
              block_is_complete ~emb_path:emb_tif_path ~scale_path:scale_tif_path
                ~x_off ~y_off ~bw ~bh
            | Float16 | Float32 ->
              block_is_complete ~emb_path:emb_tif_path ~scale_path:emb_tif_path
                ~x_off ~y_off ~bw ~bh
          in
          if is_cached then begin
            Printf.printf "  Block %d/%d (%d,%d) — cached\n%!" block_num total_blocks y_off x_off;
            Atomic.set inference_started true;
            loop ()
          end else

          let t_block_start = Unix.gettimeofday () in
          Printf.printf "  Block %d/%d (%d,%d) %dx%d loading...\n%!"
            block_num total_blocks y_off x_off bh bw;

          let (valid_pixels, s2_bands, s2_masks, s2_doys,
               s1_asc, s1_asc_doy, s1_desc, s1_desc_doy) =
            load_block ~band_cache_dir:band_cache ~valid_dates
              ~items_arr ~data_source ~s1_groups ~s1_cache_dir:s1_cache
              ~roi_w:w ~water_mask ~x_off ~y_off ~bw ~bh in

          let n_valid = Array.length valid_pixels in
          let t_loaded = Unix.gettimeofday () in
          Printf.printf "    Block %d: %d valid pixels (load: %.1fs)\n%!"
            block_num n_valid (t_loaded -. t_block_start);

          if n_valid > 0 then begin
            Mutex.lock gpu_mu;
            Atomic.set inference_started true;
            let t_infer_start = Unix.gettimeofday () in
            let (avgs, _sum_z) =
              run_block_inference session ~valid_pixels
                ~s2_bands ~s2_masks ~s2_doys
                ~s1_asc ~s1_asc_doy ~s1_desc ~s1_desc_doy
                ~repeat_times:r ~batch_size:b in
            let t_infer_end = Unix.gettimeofday () in
            Mutex.unlock gpu_mu;

            (* Write block output (open, write, close each time) *)
            Mutex.lock tif_mu;
            (match quant with
             | Int8Scale ->
               let block_emb = Array.map (fun avg ->
                 let (q, _s) = quantize_symmetric avg in q) avgs in
               let block_scales = Array.map (fun avg ->
                 let (_q, s) = quantize_symmetric avg in s) avgs in
               write_embeddings_block_int8 ~path:emb_tif_path ~x_off ~y_off ~bw ~bh
                 block_emb valid_pixels;
               write_scales_block ~path:scale_tif_path ~x_off ~y_off ~bw ~bh
                 block_scales valid_pixels
             | Float16 | Float32 ->
               write_embeddings_block_f16 ~path:emb_tif_path ~x_off ~y_off ~bw ~bh
                 avgs valid_pixels);
            Mutex.unlock tif_mu;

            Printf.printf "    Block %d: inference %.1fs (total %.1fs)\n%!"
              block_num (t_infer_end -. t_infer_start) (t_infer_end -. t_block_start)
          end;

          loop ()
        end
      in loop ()
    in

    let actual_workers = min !load_workers n_blocks in
    (* Start with 1 worker so inference begins ASAP *)
    let first = Domain.spawn worker in
    (* Wait for first inference to start, then ramp up *)
    while not (Atomic.get inference_started) do Unix.sleepf 0.01 done;
    let extra = actual_workers - 1 in
    let extra_domains = Array.init extra (fun _ -> Domain.spawn worker) in
    Domain.join first;
    Array.iter Domain.join extra_domains;

    let t_inference = Unix.gettimeofday () in
    Printf.printf "\nInference complete: %.1fs\n%!" (t_inference -. t_download);
    (match quant with
     | Int8Scale ->
       Printf.printf "  Output: %s  (%d bands, int8)\n%!" emb_tif_path latent_dim;
       Printf.printf "  Output: %s  (1 band, float32)\n%!" scale_tif_path
     | Float16 ->
       Printf.printf "  Output: %s  (%d bands, float16)\n%!" emb_tif_path latent_dim
     | Float32 ->
       Printf.printf "  Output: %s  (%d bands, float32)\n%!" emb_tif_path latent_dim);

    let t_done = Unix.gettimeofday () in
    Printf.printf "\nTile %s done (total %.1fs)\n%!" tile_id (t_done -. t_tile_start);
    Printf.printf "  Download:  %.1fs\n%!" (t_download -. t_tile_start);
    Printf.printf "  Inference: %.1fs\n%!" (t_inference -. t_download)

    end (* valid_dates <> [] *)
    end (* begin *)
  ) mgrs_tiles;

  Printf.printf "\nAll tiles done.\n%!"
