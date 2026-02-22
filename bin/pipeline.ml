(* Minimal single-file Tessera pipeline (OCaml rewrite of pipeline.py).
   Download satellite data, preprocess, run deterministic ONNX inference,
   output int8 embeddings. *)

open Bigarray

(* ======================== Constants ======================== *)

let s2_bands = [| "B04"; "B02"; "B03"; "B08"; "B8A"; "B05"; "B06"; "B07"; "B11"; "B12" |]
let scl_invalid = [| 0; 1; 2; 3; 8; 9 |]
let harmonisation_date_ymd = (2022, 1, 25)
let harmonisation_offset = 1000.0
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
let default_batch_size = 1024

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

(** Flat 4D array: [n_dates][height][width][channels] as contiguous float bigarray.
    Index: ((d * h + i) * w + j) * c + b *)
type flat_4d = {
  data : (float, float64_elt, c_layout) Array1.t;
  n : int; h : int; w : int; c : int;
}

(** Flat 3D int array: [n_dates][height][width] as contiguous int bigarray.
    Index: (d * h3 + i) * w3 + j *)
type flat_3d_int = {
  idata : (int, int_elt, c_layout) Array1.t;
  n3 : int; h3 : int; w3 : int;
}

let empty_flat_4d = { data = Array1.create float64 c_layout 0; n = 0; h = 0; w = 0; c = 0 }
let empty_flat_3d_int = { idata = Array1.create int c_layout 0; n3 = 0; h3 = 0; w3 = 0 }

(** S1 tile: URLs for a single SAR tile's VV and VH polarisations *)
type s1_tile = {
  st_vv : string option;
  st_vh : string option;
}

(** S1 group: one (date, orbit) combination with its tiles *)
type s1_group = {
  sg_date : string;
  sg_orbit : string;
  sg_tiles : s1_tile list;
}

(* ======================== NPY helpers ======================== *)

(** Load npy as flat 4D bigarray [n_dates][height][width][bands] *)
let load_npy_4d_flat filename =
  let arr = Npy.load filename |> Result.get_ok in
  assert (Array.length arr.shape = 4);
  let n = arr.shape.(0) and h = arr.shape.(1) and w = arr.shape.(2) and c = arr.shape.(3) in
  let total = n * h * w * c in
  let data = Array1.create float64 c_layout total in
  for i = 0 to total - 1 do
    Array1.set data i (Npy.get_float arr i)
  done;
  { data; n; h; w; c }

(** Load npy as flat 3D int bigarray [n_dates][height][width] *)
let load_npy_3d_int_flat filename =
  let arr = Npy.load filename |> Result.get_ok in
  assert (Array.length arr.shape = 3);
  let n = arr.shape.(0) and h = arr.shape.(1) and w = arr.shape.(2) in
  let total = n * h * w in
  let idata = Array1.create int c_layout total in
  for i = 0 to total - 1 do
    Array1.set idata i (Npy.get_int arr i)
  done;
  { idata; n3 = n; h3 = h; w3 = w }

(** Load npy as int array [n] for doys *)
let load_npy_1d_int filename =
  let arr = Npy.load filename |> Result.get_ok in
  assert (Array.length arr.shape = 1);
  Npy.to_int_array arr

(* ======================== load_roi ======================== *)

type roi = {
  crs_wkt : string;     (* WKT string of CRS *)
  bounds : float * float * float * float;  (* left, bottom, right, top *)
  bbox : float * float * float * float;    (* EPSG:4326 bbox *)
  height : int;
  width : int;
  resolution : float;
  mask : int array array;  (* H x W, 1 if pixel > 0 else 0 *)
}

let load_roi tiff_path =
  Gdal.init ();
  let ds = Gdal.Dataset.open_ tiff_path |> Result.get_ok in
  let h = Gdal.Dataset.raster_y_size ds in
  let w = Gdal.Dataset.raster_x_size ds in
  let band = Gdal.Dataset.get_band ds 1 |> Result.get_ok in
  let data = Gdal.RasterBand.read_region Gdal.BA_float64 band
               ~x_off:0 ~y_off:0 ~x_size:w ~y_size:h
               ~buf_x:w ~buf_y:h |> Result.get_ok in
  let mask = Array.init h (fun i ->
    Array.init w (fun j ->
      if Array2.get data i j > 0.0 then 1 else 0)) in
  let gt = Gdal.Dataset.get_geo_transform ds |> Result.get_ok in
  let left = gt.origin_x in
  let pixel_w = gt.pixel_width in
  let top = gt.origin_y in
  let pixel_h = gt.pixel_height in  (* negative *)
  let right = left +. (Float.of_int w *. pixel_w) in
  let bottom = top +. (Float.of_int h *. pixel_h) in
  let resolution = Float.abs pixel_w in
  let crs_wkt = Gdal.Dataset.projection ds in
  (* Transform bounds to EPSG:4326 *)
  let src_srs = Gdal.SpatialReference.of_epsg 4326 |> Result.get_ok in
  let tile_srs = Gdal.SpatialReference.of_wkt crs_wkt |> Result.get_ok in
  let ct = Gdal.CoordinateTransformation.create tile_srs src_srs |> Result.get_ok in
  let (xmin4326, ymin4326, xmax4326, ymax4326) =
    Gdal.CoordinateTransformation.transform_bounds ct
      ~xmin:(Float.min left right) ~ymin:(Float.min bottom top)
      ~xmax:(Float.max left right) ~ymax:(Float.max bottom top)
      ~density:21
    |> Result.get_ok
  in
  Gdal.CoordinateTransformation.destroy ct;
  Gdal.SpatialReference.destroy tile_srs;
  Gdal.SpatialReference.destroy src_srs;
  Gdal.Dataset.close ds;
  { crs_wkt;
    bounds = (left, bottom, right, top);
    bbox = (xmin4326, ymin4326, xmax4326, ymax4326);
    height = h;
    width = w;
    resolution;
    mask }

(* ======================== Nested-to-flat converters (download path only) ======================== *)

(** Convert nested float array array array array [n][h][w][c] to flat_4d *)
let flatten_4d (arr : float array array array array) =
  let n = Array.length arr in
  if n = 0 then empty_flat_4d
  else
    let h = Array.length arr.(0) in
    let w = Array.length arr.(0).(0) in
    let c = Array.length arr.(0).(0).(0) in
    let total = n * h * w * c in
    let data = Array1.create float64 c_layout total in
    for d = 0 to n - 1 do
      for i = 0 to h - 1 do
        for j = 0 to w - 1 do
          let base = ((d * h + i) * w + j) * c in
          for b = 0 to c - 1 do
            Array1.set data (base + b) arr.(d).(i).(j).(b)
          done
        done
      done
    done;
    { data; n; h; w; c }

(** Convert nested int array array array [n][h][w] to flat_3d_int *)
let flatten_3d_int (arr : int array array array) =
  let n = Array.length arr in
  if n = 0 then empty_flat_3d_int
  else
    let h = Array.length arr.(0) in
    let w = Array.length arr.(0).(0) in
    let total = n * h * w in
    let idata = Array1.create int c_layout total in
    for d = 0 to n - 1 do
      for i = 0 to h - 1 do
        for j = 0 to w - 1 do
          Array1.set idata ((d * h + i) * w + j) arr.(d).(i).(j)
        done
      done
    done;
    { idata; n3 = n; h3 = h; w3 = w }

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
    s3://bucket/key → /vsis3/bucket/key, https://... → /vsicurl/https://... *)
let gdal_vsi_path url =
  if String.length url > 5 && String.sub url 0 5 = "s3://" then
    "/vsis3/" ^ String.sub url 5 (String.length url - 5)
  else
    "/vsicurl/" ^ url

let warp_id = Atomic.make 0

let warp_to_mem ~dst_crs_wkt ~bounds:(left, bottom, right, top)
    ~width ~height url =
  let id = Atomic.fetch_and_add warp_id 1 in
  let dst = Printf.sprintf "/vsimem/warp_%d" id in
  match Gdal.Dataset.open_ex (gdal_vsi_path url) with
  | Error msg ->
    eprintf "Warning: open failed for %s: %s\n%!" url msg;
    Error msg
  | Ok src ->
    let opts = [
      "-t_srs"; dst_crs_wkt;
      "-te"; Printf.sprintf "%.15g" left;
             Printf.sprintf "%.15g" bottom;
             Printf.sprintf "%.15g" right;
             Printf.sprintf "%.15g" top;
      "-ts"; string_of_int width; string_of_int height;
      "-r"; "near";
      "-of"; "MEM";
    ] in
    let result = Gdal.Dataset.warp src ~dst_filename:dst opts in
    Gdal.Dataset.close src;
    result

let read_cog_band ~dst_crs_wkt ~bounds ~width ~height url =
  match warp_to_mem ~dst_crs_wkt ~bounds ~width ~height url with
  | Error _ ->
    eprintf "Warning: warp failed for %s\n%!" url;
    Array.make (height * width) 0.0
  | Ok warped ->
    let band = Gdal.Dataset.get_band warped 1 |> Result.get_ok in
    let w_out = Gdal.RasterBand.x_size band |> Result.get_ok in
    let h_out = Gdal.RasterBand.y_size band |> Result.get_ok in
    match Gdal.RasterBand.read_region Gdal.BA_float64 band
            ~x_off:0 ~y_off:0 ~x_size:w_out ~y_size:h_out
            ~buf_x:w_out ~buf_y:h_out with
    | Ok ba ->
      let arr = Array.init (h_out * w_out) (fun i ->
        Array2.get ba (i / w_out) (i mod w_out)) in
      Gdal.Dataset.close warped; arr
    | Error _ ->
      eprintf "Warning: band read failed for %s\n%!" url;
      Gdal.Dataset.close warped;
      Array.make (height * width) 0.0

let read_cog_band_byte ~dst_crs_wkt ~bounds ~width ~height url =
  match warp_to_mem ~dst_crs_wkt ~bounds ~width ~height url with
  | Error _ ->
    eprintf "Warning: warp failed for %s\n%!" url;
    Array.make (height * width) 0
  | Ok warped ->
    let band = Gdal.Dataset.get_band warped 1 |> Result.get_ok in
    let w_out = Gdal.RasterBand.x_size band |> Result.get_ok in
    let h_out = Gdal.RasterBand.y_size band |> Result.get_ok in
    match Gdal.RasterBand.read_byte band
            ~x_off:0 ~y_off:0 ~x_size:w_out ~y_size:h_out with
    | Ok ba ->
      let arr = Array.init (h_out * w_out) (fun i ->
        Array2.get ba (i / w_out) (i mod w_out)) in
      Gdal.Dataset.close warped; arr
    | Error _ ->
      eprintf "Warning: band read failed for %s\n%!" url;
      Gdal.Dataset.close warped;
      Array.make (height * width) 0

(* ======================== process_s2 ======================== *)

(** Process Sentinel-2 items: SCL cloud masking, smart mosaic, harmonisation.
    Returns (bands[n_dates][H][W][10], masks[n_dates][H][W], doys[n_dates]) *)
let process_s2 ~(roi : roi) ~(client : Stac_client.t) ~data_source ~n_workers items =
  let h = roi.height and w = roi.width in
  let (left, bottom, right, top) = roi.bounds in
  if items = [] then begin
    eprintf "  No S2 items found\n%!";
    ([||], [||], [||])
  end else begin
    (* --- Pass 1: SCL analysis --- *)
    eprintf "  Loading SCL (cloud mask)...\n%!";
    (* Group items by date *)
    let item_dates = Array.of_list (List.map (fun (it : Stac_client.item) ->
      date_of_datetime (match Stac_client.get_datetime it with Some d -> d | None -> "")
    ) items) in
    let items_arr = Array.of_list items in
    let dates = sort_uniq_strings (Array.to_list item_dates) in
    let roi_total = Array.fold_left (fun acc row ->
      acc + Array.fold_left ( + ) 0 row) 0 roi.mask in

    (* Pre-sign all items *)
    let signed_items = Array.map (fun item -> match data_source with
      | MPC -> Stac_client.sign_planetary_computer client item
      | AWS -> item) items_arr in

    (* For each item, load SCL band (parallel) *)
    let n_items = Array.length items_arr in
    let scl_results = parallel_map ~n_workers (fun i ->
      let access_item = signed_items.(i) in
      match get_asset_href access_item (scl_asset_name data_source) with
      | None ->
        eprintf "  Warning: item %s has no SCL asset\n%!" items_arr.(i).id;
        Array.make (h * w) 0
      | Some href ->
        read_cog_band_byte ~dst_crs_wkt:roi.crs_wkt ~bounds:(left, bottom, right, top)
          ~width:w ~height:h href
    ) (Array.init n_items Fun.id) in
    let scl_data = Array.map (fun opt -> match opt with
      | Some d -> d
      | None -> Array.make (h * w) 0) scl_results in

    (* For each date, build tile_selection *)
    let valid_dates = ref [] in
    let day_tile_sel = Hashtbl.create 64 in
    List.iter (fun date_str ->
      (* Find item indices for this date *)
      let indices = ref [] in
      Array.iteri (fun i d -> if d = date_str then indices := i :: !indices) item_dates;
      let indices = List.rev !indices in
      let n_tiles = List.length indices in

      (* tile_selection: for each pixel, first tile with valid SCL *)
      let tile_sel = Array.init h (fun _ -> Array.make w (-1)) in
      let assigned = Array.init h (fun _ -> Array.make w false) in
      List.iteri (fun _t_idx item_idx ->
        let scl = scl_data.(item_idx) in
        for i = 0 to h - 1 do
          for j = 0 to w - 1 do
            if roi.mask.(i).(j) > 0 && not assigned.(i).(j) then begin
              let v = scl.(i * w + j) in
              if not (is_scl_invalid v) then begin
                tile_sel.(i).(j) <- _t_idx;
                assigned.(i).(j) <- true
              end
            end
          done
        done
      ) indices;
      ignore n_tiles;

      (* Coverage check *)
      let valid_cnt = ref 0 in
      for i = 0 to h - 1 do
        for j = 0 to w - 1 do
          if tile_sel.(i).(j) >= 0 then incr valid_cnt
        done
      done;
      let valid_pct = if roi_total > 0
        then 100.0 *. Float.of_int !valid_cnt /. Float.of_int roi_total
        else 0.0 in
      if valid_pct >= 0.01 then begin
        valid_dates := date_str :: !valid_dates;
        Hashtbl.replace day_tile_sel date_str tile_sel
      end
    ) dates;
    let valid_dates = List.rev !valid_dates in
    eprintf "  %d/%d dates pass cloud filter\n%!" (List.length valid_dates) (List.length dates);

    if valid_dates = [] then
      ([||], [||], [||])
    else begin
      (* --- Pass 2: Spectral bands (parallel per date) --- *)
      eprintf "  Loading spectral bands...\n%!";
      let valid_dates_arr = Array.of_list valid_dates in
      let date_results = parallel_map ~n_workers (fun date_str ->
        let doy = doy_of_date_str date_str in
        let tile_sel = Hashtbl.find day_tile_sel date_str in
        (* Find items for this date *)
        let indices = ref [] in
        Array.iteri (fun i d -> if d = date_str then indices := i :: !indices) item_dates;
        let indices = List.rev !indices in

        (* Load band COGs for each tile (using pre-signed items) *)
        let tile_band_data = List.map (fun item_idx ->
          let access_item = signed_items.(item_idx) in
          Array.map (fun band_name ->
            match get_asset_href access_item band_name with
            | None -> Array.make (h * w) 0.0
            | Some href ->
              read_cog_band ~dst_crs_wkt:roi.crs_wkt ~bounds:(left, bottom, right, top)
                ~width:w ~height:h href
          ) (s2_band_names data_source)
        ) indices in
        let tile_band_data = Array.of_list tile_band_data in
        let n_tiles = Array.length tile_band_data in

        let out_bands = Array.init h (fun _ -> Array.init w (fun _ -> Array.make 10 0.0)) in
        for bi = 0 to 9 do
          for i = 0 to h - 1 do
            for j = 0 to w - 1 do
              let ts = tile_sel.(i).(j) in
              let v =
                if ts >= 0 && ts < n_tiles then
                  tile_band_data.(ts).(bi).(i * w + j)
                else if roi.mask.(i).(j) > 0 && n_tiles > 0 then
                  tile_band_data.(0).(bi).(i * w + j)
                else
                  0.0
              in
              (* Harmonisation *)
              let v =
                if is_after_harmonisation date_str && v >= harmonisation_offset then
                  v -. harmonisation_offset
                else v
              in
              let v = Float.max 0.0 (Float.min 65535.0 v) in
              out_bands.(i).(j).(bi) <- v
            done
          done
        done;

        let out_mask = Array.init h (fun i ->
          Array.init w (fun j ->
            if tile_sel.(i).(j) >= 0 then 1 else 0)) in

        (out_bands, out_mask, doy)
      ) valid_dates_arr in

      (* Collect results, skipping failures *)
      let all_bands = ref [] in
      let all_masks = ref [] in
      let all_doys = ref [] in
      Array.iter (fun opt -> match opt with
        | Some (bands, mask, doy) ->
          all_bands := bands :: !all_bands;
          all_masks := mask :: !all_masks;
          all_doys := doy :: !all_doys
        | None -> ()
      ) date_results;

      (Array.of_list (List.rev !all_bands),
       Array.of_list (List.rev !all_masks),
       Array.of_list (List.rev !all_doys))
    end
  end

(* ======================== S1 processing ======================== *)

(** amplitude_to_db: (20*log10(amp) + 50) * 200, clipped to [0, 32767] *)
let amplitude_to_db amp_arr mask_arr h w =
  let out = Array.init h (fun _ -> Array.make w 0) in
  for i = 0 to h - 1 do
    for j = 0 to w - 1 do
      if mask_arr.(i).(j) > 0 then begin
        let amp = amp_arr.(i * w + j) in
        if Float.is_finite amp && amp > 0.0 then begin
          let db = 20.0 *. Float.log10 amp +. 50.0 in
          let scaled = db *. 200.0 in
          out.(i).(j) <- Float.to_int (Float.min 32767.0 (Float.max 0.0 scaled))
        end
      end
    done
  done;
  out

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
        { st_vv = get_asset_href signed "vv";
          st_vh = get_asset_href signed "vh" }
      ) group_items in
      { sg_date = date_str; sg_orbit = orbit; sg_tiles = tiles }
    ) sorted_keys
  end

(** Process prepared S1 groups into ascending/descending arrays.
    Common processing for both MPC and OPERA data sources. *)
let process_s1_groups ~(roi : roi) ~n_workers groups =
  let h = roi.height and w = roi.width in
  let (left, bottom, right, top) = roi.bounds in
  if groups = [] then begin
    eprintf "  No S1 data to process\n%!";
    ([||], [||], [||], [||])
  end else begin
    eprintf "  Loading SAR data (%d groups)...\n%!" (List.length groups);

    let group_results = parallel_map ~n_workers (fun group ->
      let doy = doy_of_date_str group.sg_date in

      let mosaic_pol get_url =
        let db_list = List.filter_map (fun tile ->
          match get_url tile with
          | None -> None
          | Some url ->
            let amp = read_cog_band ~dst_crs_wkt:roi.crs_wkt ~bounds:(left, bottom, right, top)
                        ~width:w ~height:h url in
            let db = amplitude_to_db amp roi.mask h w in
            let has_any = ref false in
            Array.iter (fun row ->
              Array.iter (fun v -> if v > 0 then has_any := true) row) db;
            if !has_any then Some db else None
        ) group.sg_tiles in
        if db_list = [] then None
        else begin
          let sum = Array.init h (fun _ -> Array.make w 0.0) in
          let cnt = Array.init h (fun _ -> Array.make w 0) in
          List.iter (fun db ->
            for i = 0 to h - 1 do
              for j = 0 to w - 1 do
                if db.(i).(j) > 0 then begin
                  sum.(i).(j) <- sum.(i).(j) +. Float.of_int db.(i).(j);
                  cnt.(i).(j) <- cnt.(i).(j) + 1
                end
              done
            done
          ) db_list;
          let out = Array.init h (fun i ->
            Array.init w (fun j ->
              if cnt.(i).(j) > 0
              then Float.to_int (sum.(i).(j) /. Float.of_int cnt.(i).(j))
              else 0)) in
          Some out
        end
      in

      let vv_out = mosaic_pol (fun t -> t.st_vv) in
      let vh_out = mosaic_pol (fun t -> t.st_vh) in
      match vv_out, vh_out with
      | None, None -> None
      | _ ->
        let vv = match vv_out with Some v -> v | None -> Array.init h (fun _ -> Array.make w 0) in
        let vh = match vh_out with Some v -> v | None -> Array.init h (fun _ -> Array.make w 0) in
        let combined = Array.init h (fun i ->
          Array.init w (fun j -> [| Float.of_int vv.(i).(j); Float.of_int vh.(i).(j) |])) in
        Some (group.sg_orbit, doy, combined)
    ) (Array.of_list groups) in

    (* Partition into ascending/descending *)
    let asc_data = ref [] in
    let asc_doys = ref [] in
    let desc_data = ref [] in
    let desc_doys = ref [] in
    Array.iter (fun opt -> match opt with
      | Some (Some (orbit, doy, combined)) ->
        if orbit = "ascending" then begin
          asc_data := combined :: !asc_data;
          asc_doys := doy :: !asc_doys
        end else begin
          desc_data := combined :: !desc_data;
          desc_doys := doy :: !desc_doys
        end
      | _ -> ()
    ) group_results;

    let desc_data_arr = Array.of_list (List.rev !desc_data) in
    let desc_doys_arr = Array.of_list (List.rev !desc_doys) in
    (* DIAGNOSTIC: check alignment of data and DOYs *)
    let nd = Array.length desc_data_arr in
    if nd > 0 then begin
      eprintf "  DIAG: %d desc dates, DOYs=[%d..%d]\n%!" nd
        desc_doys_arr.(0) desc_doys_arr.(nd - 1);
      for d = 0 to min 3 (nd - 1) do
        let nz = ref 0 in
        Array.iter (fun row -> Array.iter (fun col ->
          if col.(0) > 0.0 then incr nz) row) desc_data_arr.(d);
        eprintf "  DIAG: desc[%d] doy=%d nonzero=%d\n%!" d desc_doys_arr.(d) !nz
      done;
      if nd > 4 then begin
        let d = nd - 1 in
        let nz = ref 0 in
        Array.iter (fun row -> Array.iter (fun col ->
          if col.(0) > 0.0 then incr nz) row) desc_data_arr.(d);
        eprintf "  DIAG: desc[%d] doy=%d nonzero=%d\n%!" d desc_doys_arr.(d) !nz
      end
    end;
    (Array.of_list (List.rev !asc_data),
     Array.of_list (List.rev !asc_doys),
     desc_data_arr,
     desc_doys_arr)
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
      let v = Array1.get s2b.data (src_base + b) in
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
    if vv <> 0.0 || vh <> 0.0 then begin
      valid_buf.(!n_valid) <- d;
      incr n_valid
    end
  done;
  (* Check descending — indices offset by n_asc *)
  for d = 0 to n_desc - 1 do
    let base = ((d * s1_desc.h + i) * s1_desc.w + j) * s1_desc.c in
    let vv = Array1.get s1_desc.data base in
    let vh = Array1.get s1_desc.data (base + 1) in
    if vv <> 0.0 || vh <> 0.0 then begin
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
        let v = Array1.get s1_asc.data (src_base + b) in
        Array1.set dst (dst_base + b)
          ((v -. s1_band_mean.(b)) /. (s1_band_std.(b) +. 1e-9))
      done;
      Array1.set dst (dst_base + 2) (Float.of_int s1_asc_doy.(ti))
    end else begin
      let d = ti - n_asc in
      let src_base = ((d * s1_desc.h + i) * s1_desc.w + j) * s1_desc.c in
      for b = 0 to 1 do
        let v = Array1.get s1_desc.data (src_base + b) in
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
  match Gdal.Dataset.open_ex (gdal_vsi_path url) with
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
          { st_vv = Some g.og_vv_url;
            st_vh = Some g.og_vh_url }
        ) orbit_granules in
        groups := { sg_date = date_str; sg_orbit = orbit; sg_tiles = tiles } :: !groups
      ) by_orbit
    ) sorted_dates;
    List.rev !groups
  end

(* ======================== Main pipeline ======================== *)

let () =
  (* CLI parsing *)
  let input_tiff = ref "" in
  let model_path = ref "" in
  let output_dir = ref "" in
  let start_date = ref "2024-01-01" in
  let end_date = ref "2024-12-31" in
  let max_cloud = ref 100.0 in
  let num_threads = ref 1 in
  let batch_size = ref default_batch_size in
  let repeat_times = ref 1 in
  let dpixel_dir = ref "" in
  let cache_dir = ref "" in
  let download_only = ref false in
  let cuda_device = ref (-1) in
  let data_source_str = ref "mpc" in
  let download_workers = ref 8 in

  let speclist = [
    ("--input_tiff", Arg.Set_string input_tiff, "Path to grid GeoTIFF");
    ("--model", Arg.Set_string model_path, "Path to ONNX model");
    ("--output", Arg.Set_string output_dir, "Output directory");
    ("--start", Arg.Set_string start_date, "Start date (YYYY-MM-DD)");
    ("--end", Arg.Set_string end_date, "End date (YYYY-MM-DD)");
    ("--max_cloud", Arg.Set_float max_cloud, "Max cloud cover %");
    ("--num_threads", Arg.Set_int num_threads, "CPU threads for ONNX");
    ("--batch_size", Arg.Set_int batch_size, "Inference batch size");
    ("--repeat_times", Arg.Set_int repeat_times, "Stratified sampling passes");
    ("--dpixel_dir", Arg.Set_string dpixel_dir, "Load preprocessed .npy from dir");
    ("--cache_dir", Arg.Set_string cache_dir, "Cache preprocessed data");
    ("--download_only", Arg.Set download_only, "Only download, skip inference");
    ("--cuda", Arg.Set_int cuda_device, "CUDA device ID (e.g. 0) for GPU inference");
    ("--data_source", Arg.Set_string data_source_str, "Data source: mpc or aws (default: mpc)");
    ("--download_workers", Arg.Set_int download_workers, "Parallel GDAL download workers (default: 8)");
  ] in
  Arg.parse speclist (fun _ -> ()) "Minimal OCaml Tessera pipeline";

  let data_source = match String.lowercase_ascii !data_source_str with
    | "mpc" -> MPC
    | "aws" -> AWS
    | s -> failwith (Printf.sprintf "Unknown data source: %s (expected mpc or aws)" s)
  in

  if !download_only && !cache_dir = "" then
    failwith "--download_only requires --cache_dir";

  (* Grid ID *)
  let grid_id =
    if !input_tiff <> "" then
      Filename.remove_extension (Filename.basename !input_tiff)
    else if !dpixel_dir <> "" then
      Filename.basename !dpixel_dir
    else
      failwith "--input_tiff is required unless --dpixel_dir is provided"
  in
  let date_range = !start_date ^ "/" ^ !end_date in

  (* Skip if output exists *)
  let emb_path = Filename.concat !output_dir (grid_id ^ ".npy") in
  let scale_path = Filename.concat !output_dir (grid_id ^ "_scales.npy") in
  if not !download_only && Sys.file_exists emb_path && Sys.file_exists scale_path then begin
    Printf.printf "Output already exists: %s\nSkipping.\n%!" emb_path;
    exit 0
  end;

  let t_start = Unix.gettimeofday () in
  Printf.printf "\nStarted: %.0f\n%!" t_start;

  (* ================================================================ *)
  (* Step 1: Download & preprocess satellite data                     *)
  (* ================================================================ *)
  Printf.printf "\n%s\nStep 1: Download & preprocess\n%s\n%!"
    (String.make 60 '=') (String.make 60 '=');

  (* Resolve where to load/save preprocessed data *)
  let dpixel_load_dir = ref !dpixel_dir in
  let dpixel_save_dir = ref "" in
  if !cache_dir <> "" && !dpixel_load_dir = "" then begin
    let cache_tile_dir = Filename.concat !cache_dir grid_id in
    let check_subdir = Filename.concat (Filename.concat cache_tile_dir "s2") "bands.npy" in
    let check_flat = Filename.concat cache_tile_dir "bands.npy" in
    if Sys.file_exists check_subdir || Sys.file_exists check_flat then begin
      dpixel_load_dir := cache_tile_dir;
      Printf.printf "\nFound cached preprocessed data in %s\n%!" cache_tile_dir
    end else
      dpixel_save_dir := cache_tile_dir
  end;

  (* These will hold the satellite data as flat bigarrays *)
  let s2_bands_data = ref empty_flat_4d in
  let s2_masks_data = ref empty_flat_3d_int in
  let s2_doys_data = ref [||] in
  let s1_asc_data = ref empty_flat_4d in
  let s1_asc_doy_data = ref [||] in
  let s1_desc_data = ref empty_flat_4d in
  let s1_desc_doy_data = ref [||] in

  if !dpixel_load_dir <> "" then begin
    Printf.printf "\nLoading preprocessed data from %s\n%!" !dpixel_load_dir;
    let d = !dpixel_load_dir in
    let s2_dir = Filename.concat d "s2" in
    let has_subdirs = Sys.file_exists s2_dir && Sys.is_directory s2_dir in
    if has_subdirs then begin
      s2_bands_data := load_npy_4d_flat (Filename.concat s2_dir "bands.npy");
      s2_masks_data := load_npy_3d_int_flat (Filename.concat s2_dir "masks.npy");
      s2_doys_data := load_npy_1d_int (Filename.concat s2_dir "doys.npy");
      let s1_dir = Filename.concat d "s1" in
      s1_asc_data := load_npy_4d_flat (Filename.concat s1_dir "sar_ascending.npy");
      s1_asc_doy_data := load_npy_1d_int (Filename.concat s1_dir "sar_ascending_doy.npy");
      s1_desc_data := load_npy_4d_flat (Filename.concat s1_dir "sar_descending.npy");
      s1_desc_doy_data := load_npy_1d_int (Filename.concat s1_dir "sar_descending_doy.npy");
    end else begin
      s2_bands_data := load_npy_4d_flat (Filename.concat d "bands.npy");
      s2_masks_data := load_npy_3d_int_flat (Filename.concat d "masks.npy");
      s2_doys_data := load_npy_1d_int (Filename.concat d "doys.npy");
      s1_asc_data := load_npy_4d_flat (Filename.concat d "sar_ascending.npy");
      s1_asc_doy_data := load_npy_1d_int (Filename.concat d "sar_ascending_doy.npy");
      s1_desc_data := load_npy_4d_flat (Filename.concat d "sar_descending.npy");
      s1_desc_doy_data := load_npy_1d_int (Filename.concat d "sar_descending_doy.npy");
    end;
    Printf.printf "  S2: [%d] dates, S1 asc: [%d], S1 desc: [%d]\n%!"
      (!s2_bands_data).n
      (!s1_asc_data).n
      (!s1_desc_data).n
  end else begin
    (* Download via STAC + GDAL *)
    let n_workers = !download_workers in
    let client = Stac_client.make () in
    let roi = load_roi !input_tiff in
    Printf.printf "  Tile: %dx%d, resolution=%.6f\n%!" roi.height roi.width roi.resolution;
    let (xmin, ymin, xmax, ymax) = roi.bbox in

    (* Set GDAL COG optimization for AWS sources *)
    if data_source = AWS then begin
      Gdal.set_config_option "GDAL_DISABLE_READDIR_ON_OPEN" "EMPTY_DIR";
      Gdal.set_config_option "GDAL_HTTP_MULTIRANGE" "YES";
      Gdal.set_config_option "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES" "YES";
    end;

    let s2_base_url = match data_source with
      | MPC -> stac_url
      | AWS -> aws_stac_url
    in
    let t_s2_start = Unix.gettimeofday () in
    Printf.printf "\nSearching Sentinel-2 [%s] (%s)...\n%!"
      (match data_source with MPC -> "MPC" | AWS -> "AWS") date_range;
    let s2_params : Stac_client.search_params = {
      collections = ["sentinel-2-l2a"];
      bbox = [xmin; ymin; xmax; ymax];
      datetime = date_range;
      query = Some (`Assoc ["eo:cloud_cover", `Assoc ["lt", `Float !max_cloud]]);
      limit = None;
    } in
    let s2_items = Stac_client.search client ~base_url:s2_base_url s2_params in
    Printf.printf "  Found %d scenes\n%!" (List.length s2_items);
    Printf.printf "Processing Sentinel-2 (workers=%d)...\n%!" n_workers;
    let (s2b, s2m, s2d) = process_s2 ~roi ~client ~data_source ~n_workers s2_items in
    s2_bands_data := flatten_4d s2b;
    s2_masks_data := flatten_3d_int s2m;
    s2_doys_data := s2d;
    let t_s2_end = Unix.gettimeofday () in
    Printf.printf "  Result: %d valid days (%.1fs)\n%!" (Array.length s2b) (t_s2_end -. t_s2_start);

    let t_s1_start = Unix.gettimeofday () in
    Printf.printf "\nSearching Sentinel-1 [%s] (%s)...\n%!"
      (match data_source with MPC -> "MPC" | AWS -> "AWS/OPERA") date_range;
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
        Printf.printf "  Found %d scenes\n%!" (List.length s1_items);
        prepare_s1_mpc ~client s1_items
      | AWS ->
        let token = read_edl_token () in
        setup_opera_gdal_auth token;
        prepare_s1_opera ~client
          ~bbox:(xmin, ymin, xmax, ymax) ~datetime:date_range
    in
    Printf.printf "Processing Sentinel-1 (%d groups, workers=%d)...\n%!" (List.length s1_groups) n_workers;
    let (s1a, s1ad, s1de, s1dd) = process_s1_groups ~roi ~n_workers s1_groups in
    (if data_source = AWS then
      Gdal.set_config_option "GDAL_HTTP_HEADER_FILE" "");
    s1_asc_data := flatten_4d s1a;
    s1_asc_doy_data := s1ad;
    s1_desc_data := flatten_4d s1de;
    s1_desc_doy_data := s1dd;
    let t_s1_end = Unix.gettimeofday () in
    Printf.printf "  Ascending: %d passes, Descending: %d passes (%.1fs)\n%!"
      (Array.length s1a) (Array.length s1de) (t_s1_end -. t_s1_start);

    (* Save to cache if requested *)
    if !dpixel_save_dir <> "" then begin
      let d = !dpixel_save_dir in
      mkdir_p d;
      Printf.printf "\nSaving preprocessed data to %s\n%!" d;
      let save_flat_4d path (f : flat_4d) =
        let total = f.n * f.h * f.w * f.c in
        let arr = Array.init total (fun i -> Array1.get f.data i) in
        Npy.save path (Npy.of_float_array Npy.Float64 [| f.n; f.h; f.w; f.c |] arr)
      in
      let save_flat_3d_int path (f : flat_3d_int) =
        let total = f.n3 * f.h3 * f.w3 in
        let arr = Array.init total (fun i -> Array1.get f.idata i) in
        Npy.save path (Npy.of_int_array Npy.Int32 [| f.n3; f.h3; f.w3 |] arr)
      in
      let save_1d_int path arr =
        Npy.save path (Npy.of_int_array Npy.Int32 [| Array.length arr |] arr)
      in
      save_flat_4d (Filename.concat d "bands.npy") !s2_bands_data;
      save_flat_3d_int (Filename.concat d "masks.npy") !s2_masks_data;
      save_1d_int (Filename.concat d "doys.npy") !s2_doys_data;
      save_flat_4d (Filename.concat d "sar_ascending.npy") !s1_asc_data;
      save_1d_int (Filename.concat d "sar_ascending_doy.npy") !s1_asc_doy_data;
      save_flat_4d (Filename.concat d "sar_descending.npy") !s1_desc_data;
      save_1d_int (Filename.concat d "sar_descending_doy.npy") !s1_desc_doy_data;
      Printf.printf "  Saved 7 .npy files\n%!"
    end
  end;

  if !download_only then begin
    Printf.printf "\n--download_only: skipping inference.\n%!";
    Printf.printf "Done.\n%!";
    exit 0
  end;

  let s2_bands = !s2_bands_data in
  let s2_masks = !s2_masks_data in
  let s2_doys = !s2_doys_data in
  let s1_asc = !s1_asc_data in
  let s1_asc_doy = !s1_asc_doy_data in
  let s1_desc = !s1_desc_data in
  let s1_desc_doy = !s1_desc_doy_data in

  (* Get spatial dimensions from flat types *)
  let h_actual, w_actual =
    if s2_bands.n > 0 then (s2_bands.h, s2_bands.w)
    else failwith "No S2 data available"
  in
  let s1_asc_empty = s1_asc.n = 0 in
  let s1_desc_empty = s1_desc.n = 0 in

  let t_download = Unix.gettimeofday () in
  Printf.printf "\nDownload complete: %.1fs\n%!" (t_download -. t_start);

  (* ================================================================ *)
  (* Step 2: Find valid pixels                                        *)
  (* ================================================================ *)
  Printf.printf "\n%s\nStep 2: Find valid pixels (min_valid_timesteps=%d)\n%s\n%!"
    (String.make 60 '=') min_valid_timesteps (String.make 60 '=');

  let valid_pixels = ref [] in
  for i = 0 to h_actual - 1 do
    for j = 0 to w_actual - 1 do
      let global_idx = i * w_actual + j in
      (* S2 valid count — flat indexing *)
      let s2_valid = ref 0 in
      let s2_nonzero = ref false in
      for d = 0 to s2_bands.n - 1 do
        let mask_idx = (d * s2_masks.h3 + i) * s2_masks.w3 + j in
        if Array1.get s2_masks.idata mask_idx > 0 then incr s2_valid;
        if not !s2_nonzero then begin
          let base = ((d * s2_bands.h + i) * s2_bands.w + j) * s2_bands.c in
          let found = ref false in
          for b = 0 to s2_bands.c - 1 do
            if Array1.get s2_bands.data (base + b) <> 0.0 then found := true
          done;
          if !found then s2_nonzero := true
        end
      done;
      (* S1 valid count — flat indexing *)
      let s1_asc_valid = ref 0 in
      if not s1_asc_empty then
        for d = 0 to s1_asc.n - 1 do
          let base = ((d * s1_asc.h + i) * s1_asc.w + j) * s1_asc.c in
          if Array1.get s1_asc.data base <> 0.0 || Array1.get s1_asc.data (base + 1) <> 0.0 then
            incr s1_asc_valid
        done;
      let s1_desc_valid = ref 0 in
      if not s1_desc_empty then
        for d = 0 to s1_desc.n - 1 do
          let base = ((d * s1_desc.h + i) * s1_desc.w + j) * s1_desc.c in
          if Array1.get s1_desc.data base <> 0.0 || Array1.get s1_desc.data (base + 1) <> 0.0 then
            incr s1_desc_valid
        done;
      let s1_total = !s1_asc_valid + !s1_desc_valid in

      let valid =
        if s1_asc_empty && s1_desc_empty then
          !s2_nonzero && !s2_valid >= min_valid_timesteps
        else
          !s2_nonzero && !s2_valid >= min_valid_timesteps && s1_total >= min_valid_timesteps
      in
      if valid then
        valid_pixels := (global_idx, i, j) :: !valid_pixels
    done
  done;
  let valid_pixels = Array.of_list (List.rev !valid_pixels) in
  Printf.printf "  %d valid pixels out of %d\n%!" (Array.length valid_pixels) (h_actual * w_actual);
  if Array.length valid_pixels = 0 then begin
    Printf.printf "No valid pixels found. Exiting.\n%!";
    exit 0
  end;

  (* ================================================================ *)
  (* Step 3: Load ONNX model                                         *)
  (* ================================================================ *)
  Printf.printf "\n%s\nStep 3: Load ONNX model\n%s\n%!"
    (String.make 60 '=') (String.make 60 '=');

  Printf.printf "Loading model: %s\n%!" !model_path;
  let env = Onnxruntime.Env.create ~log_level:3 "tessera" in
  let cuda_opt = if !cuda_device >= 0 then Some !cuda_device else None in
  let session = Onnxruntime.Session.create env ~threads:!num_threads ?cuda_device:cuda_opt !model_path in
  if !cuda_device >= 0 then
    Printf.printf "  Using CUDA device %d\n%!" !cuda_device;
  Printf.printf "  Model loaded successfully\n%!";

  (* ================================================================ *)
  (* Step 4: Deterministic inference                                  *)
  (* ================================================================ *)
  let r = !repeat_times in
  let n_valid = Array.length valid_pixels in
  Printf.printf "\n%s\nStep 4: Inference (%d pixels, batch_size=%d, repeat_times=%d)\n%s\n%!"
    (String.make 60 '=') n_valid !batch_size r (String.make 60 '=');

  let out_embeddings = Array.make (h_actual * w_actual * latent_dim) 0 in
  let out_scales = Array.make (h_actual * w_actual) 0.0 in

  (* Accumulator for multi-pass averaging: sum of raw ONNX outputs per valid pixel *)
  let sum_z = Array.init n_valid (fun _ -> Array.make latent_dim 0.0) in

  let b = !batch_size in
  let n_batches = (n_valid + b - 1) / b in

  (* Pre-allocate input tensors at max batch size — reused every batch *)
  let s2_input = Array1.create float32 c_layout (b * sample_size_s2 * 11) in
  let s1_input = Array1.create float32 c_layout (b * sample_size_s1 * 3) in

  (* Create effective S1 arrays: dummy 1-date zero arrays when empty *)
  let make_dummy_s1 () =
    let total = 1 * h_actual * w_actual * 2 in
    let data = Array1.create float64 c_layout total in
    Array1.fill data 0.0;
    { data; n = 1; h = h_actual; w = w_actual; c = 2 }
  in
  let s1_asc_eff = if s1_asc_empty then make_dummy_s1 () else s1_asc in
  let s1_asc_doy_eff = if s1_asc_empty then [| 180 |] else s1_asc_doy in
  let s1_desc_eff = if s1_desc_empty then make_dummy_s1 () else s1_desc in
  let s1_desc_doy_eff = if s1_desc_empty then [| 180 |] else s1_desc_doy in

  for pass = 0 to r - 1 do
    Printf.printf "  Pass %d/%d\n%!" (pass + 1) r;

    for batch_idx = 0 to n_batches - 1 do
      let start_px = batch_idx * b in
      let end_px = min (start_px + b) n_valid in
      let actual_b = end_px - start_px in

      (* Zero out the input buffers *)
      Array1.fill s2_input 0.0;
      Array1.fill s1_input 0.0;

      (* Fill batch — flat sampling, no per-pixel allocations *)
      for px_in_batch = 0 to actual_b - 1 do
        let px = start_px + px_in_batch in
        let (_global_idx, i, j) = valid_pixels.(px) in

        sample_s2_flat s2_bands s2_masks s2_doys
          ~i ~j ~pass ~dst:s2_input
          ~dst_offset:(px_in_batch * sample_size_s2 * 11);

        sample_s1_flat s1_asc_eff s1_asc_doy_eff s1_desc_eff s1_desc_doy_eff
          ~i ~j ~pass ~dst:s1_input
          ~dst_offset:(px_in_batch * sample_size_s1 * 3)
      done;

      (* Use sub-views for the last (possibly smaller) batch *)
      let s2_ba = if actual_b = b then s2_input
        else Array1.sub s2_input 0 (actual_b * sample_size_s2 * 11) in
      let s1_ba = if actual_b = b then s1_input
        else Array1.sub s1_input 0 (actual_b * sample_size_s1 * 3) in

      (* Run ONNX inference — batched, using C-cached names *)
      let outputs = Onnxruntime.Session.run_cached_ba session
        [|(s2_ba,
           [| Int64.of_int actual_b; Int64.of_int sample_size_s2; 11L |]);
          (s1_ba,
           [| Int64.of_int actual_b; Int64.of_int sample_size_s1; 3L |])|]
        ~output_sizes:[| actual_b * latent_dim |]
      in
      let output_ba = outputs.(0) in

      (* Accumulate results *)
      for px_in_batch = 0 to actual_b - 1 do
        let px = start_px + px_in_batch in
        let out_offset = px_in_batch * latent_dim in
        for k = 0 to latent_dim - 1 do
          sum_z.(px).(k) <- sum_z.(px).(k) +. Array1.get output_ba (out_offset + k)
        done
      done;

      if (batch_idx + 1) mod (max 1 (n_batches / 20)) = 0 || batch_idx = n_batches - 1 then begin
        let pct = 100.0 *. Float.of_int end_px /. Float.of_int n_valid in
        Printf.printf "    [%5.1f%%] %d/%d pixels\n%!" pct end_px n_valid
      end
    done
  done;

  (* Keep pre-allocated tensors alive through inference *)
  ignore (Sys.opaque_identity s2_input);
  ignore (Sys.opaque_identity s1_input);

  (* Average and quantize all valid pixels *)
  let r_f = Float.of_int r in
  for px = 0 to n_valid - 1 do
    let (global_idx, _i, _j) = valid_pixels.(px) in
    let avg = Array.init latent_dim (fun k -> sum_z.(px).(k) /. r_f) in
    let (q, s) = quantize_symmetric avg in
    for k = 0 to latent_dim - 1 do
      out_embeddings.(global_idx * latent_dim + k) <- q.(k)
    done;
    out_scales.(global_idx) <- s
  done;

  let t_inference = Unix.gettimeofday () in
  Printf.printf "\nInference complete: %.1fs\n%!" (t_inference -. t_download);

  (* ================================================================ *)
  (* Step 5: Save output                                              *)
  (* ================================================================ *)
  Printf.printf "\n%s\nStep 5: Save output\n%s\n%!"
    (String.make 60 '=') (String.make 60 '=');

  (* Create output directory *)
  mkdir_p !output_dir;

  (* Save embeddings as int8 npy [H_actual, W_actual, LATENT_DIM] *)
  let emb_npy = Npy.of_int_array Npy.Int8 [| h_actual; w_actual; latent_dim |]
    (Array.init (h_actual * w_actual * latent_dim) (fun idx ->
       out_embeddings.(idx))) in
  Npy.save emb_path emb_npy;

  (* Save scales as float32 npy [H_actual, W_actual] *)
  let scale_npy = Npy.of_float_array Npy.Float32 [| h_actual; w_actual |]
    (Array.init (h_actual * w_actual) (fun idx ->
       out_scales.(idx))) in
  Npy.save scale_path scale_npy;

  Printf.printf "  Embeddings: %s  shape=(%d, %d, %d) dtype=int8\n%!"
    emb_path h_actual w_actual latent_dim;
  Printf.printf "  Scales:     %s  shape=(%d, %d) dtype=float32\n%!"
    scale_path h_actual w_actual;

  let t_done = Unix.gettimeofday () in
  Printf.printf "\nDone (total %.1fs)\n%!" (t_done -. t_start);
  Printf.printf "  Download:  %.1fs\n%!" (t_download -. t_start);
  Printf.printf "  Inference: %.1fs\n%!" (t_inference -. t_download);
  Printf.printf "  Save:      %.1fs\n%!" (t_done -. t_inference)
