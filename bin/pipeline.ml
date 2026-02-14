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

(* ======================== NPY helpers ======================== *)

(** Load npy as float array array array array [n_dates][height][width][bands] *)
let load_npy_4d filename =
  let arr = Npy.load filename |> Result.get_ok in
  assert (Array.length arr.shape = 4);
  let n = arr.shape.(0) and h = arr.shape.(1) and w = arr.shape.(2) and c = arr.shape.(3) in
  Array.init n (fun d ->
    Array.init h (fun i ->
      Array.init w (fun j ->
        Array.init c (fun b ->
          Npy.get_float arr (((d * h + i) * w + j) * c + b)))))

(** Load npy as int array array array [n_dates][height][width] for masks *)
let load_npy_3d_int filename =
  let arr = Npy.load filename |> Result.get_ok in
  assert (Array.length arr.shape = 3);
  let n = arr.shape.(0) and h = arr.shape.(1) and w = arr.shape.(2) in
  Array.init n (fun d ->
    Array.init h (fun i ->
      Array.init w (fun j ->
        Npy.get_int arr ((d * h + i) * w + j))))

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
  let ds = Gdal.Dataset.open' tiff_path Gdal_c.C.Type.Access.RO in
  let h = Gdal.Dataset.raster_y_size ds in
  let w = Gdal.Dataset.raster_x_size ds in
  let band = Gdal.Dataset.raster_band ds 1 |> Option.get in
  let data = Gdal.RasterBand.read ~x_off:0 ~y_off:0 ~x_size:w ~y_size:h
               ~buf_x_size:w ~buf_y_size:h band |> Result.get_ok in
  let mask = Array.init h (fun i ->
    Array.init w (fun j ->
      if Array1.get data (i * w + j) > 0.0 then 1 else 0)) in
  let gt = Gdal.Dataset.geo_transform ds |> Result.get_ok in
  (* gt: [0]=originX, [1]=pixelWidth, [2]=0, [3]=originY, [4]=0, [5]=pixelHeight (negative) *)
  let left = Array1.get gt 0 in
  let pixel_w = Array1.get gt 1 in
  let top = Array1.get gt 3 in
  let pixel_h = Array1.get gt 5 in  (* negative *)
  let right = left +. (Float.of_int w *. pixel_w) in
  let bottom = top +. (Float.of_int h *. pixel_h) in
  let resolution = Float.abs pixel_w in
  let crs_wkt = Gdal.Dataset.projection ds in
  (* Transform bounds to EPSG:4326 *)
  let src_srs = Gdal.SpatialReference.of_epsg 4326 |> Result.get_ok in (* dummy, will parse from WKT *)
  (* We need to create SRS from WKT. Use the dataset's projection string.
     For simplicity, extract EPSG from the WKT if possible, otherwise use transform. *)
  (* Actually, we need to create an SRS from the dataset projection.
     The GDAL bindings have of_epsg. Let's try to extract EPSG code from projection. *)
  let epsg_code =
    (* Try to find EPSG code in WKT *)
    try
      let wkt = crs_wkt in
      let pat = "EPSG" in
      let idx = ref 0 in
      let found = ref false in
      let code = ref 4326 in
      (try
         for k = String.length wkt - 10 downto 0 do
           if String.length wkt > k + 4 &&
              String.sub wkt k 4 = pat then begin
             (* Look for digits after "EPSG","<digits>" *)
             let start = k + 4 in
             (* Find first digit *)
             let dstart = ref start in
             while !dstart < String.length wkt && not (Char.code wkt.[!dstart] >= 48 && Char.code wkt.[!dstart] <= 57) do
               incr dstart
             done;
             let dend = ref !dstart in
             while !dend < String.length wkt && Char.code wkt.[!dend] >= 48 && Char.code wkt.[!dend] <= 57 do
               incr dend
             done;
             if !dend > !dstart then begin
               idx := !dstart;
               code := int_of_string (String.sub wkt !dstart (!dend - !dstart));
               found := true;
               raise Exit
             end
           end
         done
       with Exit -> ());
      ignore (idx, found);
      !code
    with _ -> 4326
  in
  let tile_srs = Gdal.SpatialReference.of_epsg epsg_code |> Result.get_ok in
  let ct = Gdal.CoordinateTransformation.create tile_srs src_srs |> Result.get_ok in
  let (xmin4326, ymin4326, xmax4326, ymax4326) =
    Gdal.CoordinateTransformation.transform_bounds ct
      ~xmin:(Float.min left right) ~ymin:(Float.min bottom top)
      ~xmax:(Float.max left right) ~ymax:(Float.max bottom top)
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

(* ======================== COG reading via GDAL warp ======================== *)

(** Read a single band from a COG URL, warped to target CRS/bounds/size.
    Returns float64 data as flat array of length width*height. *)
let read_cog_band ~dst_crs_wkt ~bounds:(left, bottom, right, top)
    ~width ~height url =
  let src = Gdal.Dataset.open_ex ("/vsicurl/" ^ url) in
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
  match Gdal.Dataset.warp ~options:opts ~src "/vsimem/tmp_warp" with
  | Error _ ->
    eprintf "Warning: warp failed for %s\n%!" url;
    Array.make (height * width) 0.0
  | Ok warped ->
    let band = Gdal.Dataset.raster_band warped 1 |> Option.get in
    let w_out = Gdal.RasterBand.x_size band in
    let h_out = Gdal.RasterBand.y_size band in
    (match Gdal.RasterBand.read ~x_off:0 ~y_off:0 ~x_size:w_out ~y_size:h_out
             ~buf_x_size:w_out ~buf_y_size:h_out band with
     | Ok ba ->
       let arr = Array.init (h_out * w_out) (fun i -> Array1.get ba i) in
       Gdal.Dataset.close warped;
       arr
     | Error _ ->
       eprintf "Warning: band read failed for %s\n%!" url;
       Gdal.Dataset.close warped;
       Array.make (height * width) 0.0)

(* ======================== process_s2 ======================== *)

(** Process Sentinel-2 items: SCL cloud masking, smart mosaic, harmonisation.
    Returns (bands[n_dates][H][W][10], masks[n_dates][H][W], doys[n_dates]) *)
let process_s2 ~(roi : roi) ~sw ~(client : Stac_client.t) items =
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

    (* For each item, load SCL band *)
    let n_items = Array.length items_arr in
    let scl_data = Array.init n_items (fun i ->
      let item = items_arr.(i) in
      (* Sign item for COG access *)
      let signed_item = Stac_client.sign_planetary_computer ~sw client item in
      match get_asset_href signed_item "SCL" with
      | None ->
        eprintf "  Warning: item %s has no SCL asset\n%!" item.id;
        Array.make (h * w) 0.0
      | Some href ->
        read_cog_band ~dst_crs_wkt:roi.crs_wkt ~bounds:(left, bottom, right, top)
          ~width:w ~height:h href
    ) in

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
              let v = Float.to_int (Float.round scl.(i * w + j)) in
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
      (* --- Pass 2: Spectral bands --- *)
      eprintf "  Loading spectral bands...\n%!";
      let all_bands = ref [] in
      let all_masks = ref [] in
      let all_doys = ref [] in

      List.iter (fun date_str ->
        let doy = doy_of_date_str date_str in
        let tile_sel = Hashtbl.find day_tile_sel date_str in
        (* Find items for this date *)
        let indices = ref [] in
        Array.iteri (fun i d -> if d = date_str then indices := i :: !indices) item_dates;
        let indices = List.rev !indices in

        (* Load and sign band COGs for each tile *)
        let tile_band_data = List.map (fun item_idx ->
          let item = items_arr.(item_idx) in
          let signed_item = Stac_client.sign_planetary_computer ~sw client item in
          Array.map (fun band_name ->
            match get_asset_href signed_item band_name with
            | None -> Array.make (h * w) 0.0
            | Some href ->
              read_cog_band ~dst_crs_wkt:roi.crs_wkt ~bounds:(left, bottom, right, top)
                ~width:w ~height:h href
          ) s2_bands
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

        all_bands := out_bands :: !all_bands;
        all_masks := out_mask :: !all_masks;
        all_doys := doy :: !all_doys
      ) valid_dates;

      (Array.of_list (List.rev !all_bands),
       Array.of_list (List.rev !all_masks),
       Array.of_list (List.rev !all_doys))
    end
  end

(* ======================== process_s1 ======================== *)

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

(** Process Sentinel-1 items into ascending/descending arrays. *)
let process_s1 ~(roi : roi) ~sw ~(client : Stac_client.t) items =
  let h = roi.height and w = roi.width in
  let (left, bottom, right, top) = roi.bounds in
  if items = [] then begin
    eprintf "  No S1 items found\n%!";
    ([||], [||], [||], [||])
  end else begin
    eprintf "  Loading SAR data...\n%!";
    (* Group by (date, orbit) *)
    let by_date_orbit = Hashtbl.create 64 in
    List.iter (fun (item : Stac_client.item) ->
      let dt = date_of_datetime (match Stac_client.get_datetime item with Some d -> d | None -> "") in
      let orbit = match Stac_client.get_string_prop item "sat:orbit_state" with
        | Some o -> o | None -> "unknown" in
      let key = (dt, orbit) in
      let prev = try Hashtbl.find by_date_orbit key with Not_found -> [] in
      Hashtbl.replace by_date_orbit key (item :: prev)
    ) items;

    let asc_data = ref [] in
    let asc_doys = ref [] in
    let desc_data = ref [] in
    let desc_doys = ref [] in

    let sorted_keys = List.sort compare (Hashtbl.fold (fun k _ acc -> k :: acc) by_date_orbit []) in
    List.iter (fun (date_str, orbit) ->
      let group_items = List.rev (Hashtbl.find by_date_orbit (date_str, orbit)) in
      let doy = doy_of_date_str date_str in

      (* For each polarisation, convert to dB per tile, then mosaic (mean where > 0) *)
      let mosaic_pol pol_name =
        let db_list = List.map (fun (item : Stac_client.item) ->
          let signed_item = Stac_client.sign_planetary_computer ~sw client item in
          match get_asset_href signed_item pol_name with
          | None -> None
          | Some href ->
            let amp = read_cog_band ~dst_crs_wkt:roi.crs_wkt ~bounds:(left, bottom, right, top)
                        ~width:w ~height:h href in
            Some (amplitude_to_db amp roi.mask h w)
        ) group_items in
        let db_list = List.filter_map Fun.id db_list in
        if db_list = [] then None
        else begin
          let n_tiles = List.length db_list in
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
          ignore n_tiles;
          (* Check if any pixel has valid data *)
          let has_any = ref false in
          Array.iter (fun row -> Array.iter (fun c -> if c > 0 then has_any := true) row) cnt;
          if not !has_any then None
          else begin
            let out = Array.init h (fun i ->
              Array.init w (fun j ->
                if cnt.(i).(j) > 0
                then Float.to_int (sum.(i).(j) /. Float.of_int cnt.(i).(j))
                else 0)) in
            Some out
          end
        end
      in

      let vv_out = mosaic_pol "vv" in
      let vh_out = mosaic_pol "vh" in
      match vv_out, vh_out with
      | None, None -> ()  (* skip *)
      | _ ->
        let vv = match vv_out with Some v -> v | None -> Array.init h (fun _ -> Array.make w 0) in
        let vh = match vh_out with Some v -> v | None -> Array.init h (fun _ -> Array.make w 0) in
        let combined = Array.init h (fun i ->
          Array.init w (fun j -> [| Float.of_int vv.(i).(j); Float.of_int vh.(i).(j) |])) in
        if orbit = "ascending" then begin
          asc_data := combined :: !asc_data;
          asc_doys := doy :: !asc_doys
        end else begin
          desc_data := combined :: !desc_data;
          desc_doys := doy :: !desc_doys
        end
    ) sorted_keys;

    (Array.of_list (List.rev !asc_data),
     Array.of_list (List.rev !asc_doys),
     Array.of_list (List.rev !desc_data),
     Array.of_list (List.rev !desc_doys))
  end

(* ======================== Stratified sampling ======================== *)

let stratified_select valid_idx sample_size offset =
  let n = Array.length valid_idx in
  if n = 0 then
    Array.make sample_size 0
  else if n <= sample_size then begin
    let repeats = (sample_size / n) + 1 in
    let tiled = Array.init (repeats * n) (fun i -> valid_idx.(i mod n)) in
    let result = Array.sub tiled 0 sample_size in
    Array.sort compare result;
    result
  end else begin
    let bins = Array.init (sample_size + 1) (fun i ->
      Float.of_int n *. Float.of_int i /. Float.of_int sample_size) in
    Array.init sample_size (fun i ->
      let lo = Float.to_int bins.(i) in
      let hi = Float.to_int bins.(i + 1) in
      let hi = if hi <= lo then lo + 1 else hi in
      let pos = lo + (offset mod (hi - lo)) in
      valid_idx.(min pos (n - 1)))
  end

(** Sample & normalise S2 data for one pixel. Returns flat float array of length sample_size_s2*11. *)
let sample_s2 s2_bands_pixel s2_masks_pixel s2_doys offset =
  (* s2_bands_pixel: float array array [n_dates][10]
     s2_masks_pixel: int array [n_dates]
     s2_doys: int array [n_dates] *)
  let n_dates = Array.length s2_masks_pixel in
  let valid_idx = Array.of_list (
    let acc = ref [] in
    for i = n_dates - 1 downto 0 do
      if s2_masks_pixel.(i) > 0 then acc := i :: !acc
    done;
    !acc) in
  let valid_idx = if Array.length valid_idx = 0 then
    Array.init n_dates Fun.id
  else valid_idx in
  let idx = stratified_select valid_idx sample_size_s2 offset in
  let result = Array.make (sample_size_s2 * 11) 0.0 in
  for t = 0 to sample_size_s2 - 1 do
    let ti = idx.(t) in
    for b = 0 to 9 do
      result.(t * 11 + b) <-
        (s2_bands_pixel.(ti).(b) -. s2_band_mean.(b)) /. (s2_band_std.(b) +. 1e-9)
    done;
    result.(t * 11 + 10) <- Float.of_int s2_doys.(ti)
  done;
  result

(** Sample & normalise S1 data for one pixel. Returns flat float array of length sample_size_s1*3. *)
let sample_s1 s1_asc_pixel s1_asc_doys s1_desc_pixel s1_desc_doys offset =
  (* s1_asc_pixel: float array array [n_asc][2]
     s1_desc_pixel: float array array [n_desc][2] *)
  let s1_all = Array.append s1_asc_pixel s1_desc_pixel in
  let doys_all = Array.append s1_asc_doys s1_desc_doys in
  let n = Array.length s1_all in
  let valid_idx = Array.of_list (
    let acc = ref [] in
    for i = n - 1 downto 0 do
      if Array.exists (fun v -> v <> 0.0) s1_all.(i) then acc := i :: !acc
    done;
    !acc) in
  let valid_idx = if Array.length valid_idx = 0 then
    Array.init n Fun.id
  else valid_idx in
  let idx = stratified_select valid_idx sample_size_s1 offset in
  let result = Array.make (sample_size_s1 * 3) 0.0 in
  for t = 0 to sample_size_s1 - 1 do
    let ti = idx.(t) in
    for b = 0 to 1 do
      result.(t * 3 + b) <-
        (s1_all.(ti).(b) -. s1_band_mean.(b)) /. (s1_band_std.(b) +. 1e-9)
    done;
    result.(t * 3 + 2) <- Float.of_int doys_all.(ti)
  done;
  result

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
  ] in
  Arg.parse speclist (fun _ -> ()) "Minimal OCaml Tessera pipeline";

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
    let check_path = Filename.concat (Filename.concat cache_tile_dir "s2") "bands.npy" in
    if Sys.file_exists check_path then begin
      dpixel_load_dir := cache_tile_dir;
      Printf.printf "\nFound cached preprocessed data in %s\n%!" cache_tile_dir
    end else
      dpixel_save_dir := cache_tile_dir
  end;

  (* These will hold the satellite data *)
  let s2_bands_data = ref [||] in  (* [n_dates][H][W][10] *)
  let s2_masks_data = ref [||] in  (* [n_dates][H][W] int *)
  let s2_doys_data = ref [||] in   (* [n_dates] *)
  let s1_asc_data = ref [||] in    (* [n_dates][H][W][2] *)
  let s1_asc_doy_data = ref [||] in
  let s1_desc_data = ref [||] in
  let s1_desc_doy_data = ref [||] in

  if !dpixel_load_dir <> "" then begin
    Printf.printf "\nLoading preprocessed data from %s\n%!" !dpixel_load_dir;
    let d = !dpixel_load_dir in
    let s2_dir = Filename.concat d "s2" in
    let has_subdirs = Sys.file_exists s2_dir && Sys.is_directory s2_dir in
    if has_subdirs then begin
      s2_bands_data := load_npy_4d (Filename.concat s2_dir "bands.npy");
      s2_masks_data := load_npy_3d_int (Filename.concat s2_dir "masks.npy");
      s2_doys_data := load_npy_1d_int (Filename.concat s2_dir "doys.npy");
      let s1_dir = Filename.concat d "s1" in
      s1_asc_data := load_npy_4d (Filename.concat s1_dir "sar_ascending.npy");
      s1_asc_doy_data := load_npy_1d_int (Filename.concat s1_dir "sar_ascending_doy.npy");
      s1_desc_data := load_npy_4d (Filename.concat s1_dir "sar_descending.npy");
      s1_desc_doy_data := load_npy_1d_int (Filename.concat s1_dir "sar_descending_doy.npy");
    end else begin
      s2_bands_data := load_npy_4d (Filename.concat d "bands.npy");
      s2_masks_data := load_npy_3d_int (Filename.concat d "masks.npy");
      s2_doys_data := load_npy_1d_int (Filename.concat d "doys.npy");
      s1_asc_data := load_npy_4d (Filename.concat d "sar_ascending.npy");
      s1_asc_doy_data := load_npy_1d_int (Filename.concat d "sar_ascending_doy.npy");
      s1_desc_data := load_npy_4d (Filename.concat d "sar_descending.npy");
      s1_desc_doy_data := load_npy_1d_int (Filename.concat d "sar_descending_doy.npy");
    end;
    Printf.printf "  S2: [%d] dates, S1 asc: [%d], S1 desc: [%d]\n%!"
      (Array.length !s2_bands_data)
      (Array.length !s1_asc_data)
      (Array.length !s1_desc_data)
  end else begin
    (* Download via STAC + GDAL *)
    Eio_main.run @@ fun env ->
    Mirage_crypto_rng_unix.use_default ();
    Eio.Switch.run @@ fun sw ->
    let net = Eio.Stdenv.net env in
    let client = Stac_client.make net in
    let roi = load_roi !input_tiff in
    Printf.printf "  Tile: %dx%d, resolution=%.6f\n%!" roi.height roi.width roi.resolution;
    let (xmin, ymin, xmax, ymax) = roi.bbox in

    Printf.printf "\nSearching Sentinel-2 (%s)...\n%!" date_range;
    let s2_params : Stac_client.search_params = {
      collections = ["sentinel-2-l2a"];
      bbox = [xmin; ymin; xmax; ymax];
      datetime = date_range;
      query = Some (`Assoc ["eo:cloud_cover", `Assoc ["lt", `Float !max_cloud]]);
      limit = None;
    } in
    let s2_items = Stac_client.search ~sw client ~base_url:stac_url s2_params in
    Printf.printf "  Found %d scenes\n%!" (List.length s2_items);
    Printf.printf "Processing Sentinel-2...\n%!";
    let (s2b, s2m, s2d) = process_s2 ~roi ~sw ~client s2_items in
    s2_bands_data := s2b;
    s2_masks_data := s2m;
    s2_doys_data := s2d;
    Printf.printf "  Result: %d valid days\n%!" (Array.length s2b);

    Printf.printf "\nSearching Sentinel-1 (%s)...\n%!" date_range;
    let s1_params : Stac_client.search_params = {
      collections = ["sentinel-1-rtc"];
      bbox = [xmin; ymin; xmax; ymax];
      datetime = date_range;
      query = None;
      limit = None;
    } in
    let s1_items = Stac_client.search ~sw client ~base_url:stac_url s1_params in
    Printf.printf "  Found %d scenes\n%!" (List.length s1_items);
    Printf.printf "Processing Sentinel-1...\n%!";
    let (s1a, s1ad, s1de, s1dd) = process_s1 ~roi ~sw ~client s1_items in
    s1_asc_data := s1a;
    s1_asc_doy_data := s1ad;
    s1_desc_data := s1de;
    s1_desc_doy_data := s1dd;
    Printf.printf "  Ascending: %d passes, Descending: %d passes\n%!"
      (Array.length s1a) (Array.length s1de);

    (* Save to cache if requested *)
    if !dpixel_save_dir <> "" then begin
      Printf.printf "\nSaving preprocessed data to %s\n%!" !dpixel_save_dir;
      (* Note: we'd need to implement npy save for multi-dim arrays.
         For now, just note the cache save is TODO for the download path. *)
      Printf.printf "  (cache save not yet implemented in OCaml pipeline)\n%!"
    end
  end;

  if !download_only then begin
    Printf.printf "\n--download_only: skipping inference.\n%!";
    Printf.printf "Done.\n%!";
    exit 0
  end;

  let s2_bands = !s2_bands_data in  (* [n_dates][H][W][10] *)
  let s2_masks = !s2_masks_data in  (* [n_dates][H][W] as float *)
  let s2_doys = !s2_doys_data in
  let s1_asc = !s1_asc_data in
  let s1_asc_doy = !s1_asc_doy_data in
  let s1_desc = !s1_desc_data in
  let s1_desc_doy = !s1_desc_doy_data in

  (* Get spatial dimensions *)
  let h_actual, w_actual =
    if Array.length s2_bands > 0 then
      (Array.length s2_bands.(0), Array.length s2_bands.(0).(0))
    else
      failwith "No S2 data available"
  in
  let s1_asc_empty = Array.length s1_asc = 0 in
  let s1_desc_empty = Array.length s1_desc = 0 in

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
      (* S2 valid count *)
      let s2_valid = ref 0 in
      let s2_nonzero = ref false in
      for d = 0 to Array.length s2_bands - 1 do
        if s2_masks.(d).(i).(j) > 0 then
          incr s2_valid;
        if not !s2_nonzero then
          if Array.exists (fun v -> v <> 0.0) s2_bands.(d).(i).(j) then
            s2_nonzero := true
      done;
      (* S1 valid count *)
      let s1_asc_valid = ref 0 in
      if not s1_asc_empty then
        for d = 0 to Array.length s1_asc - 1 do
          if Array.exists (fun v -> v <> 0.0) s1_asc.(d).(i).(j) then
            incr s1_asc_valid
        done;
      let s1_desc_valid = ref 0 in
      if not s1_desc_empty then
        for d = 0 to Array.length s1_desc - 1 do
          if Array.exists (fun v -> v <> 0.0) s1_desc.(d).(i).(j) then
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
  let session = Onnxruntime.Session.create env ~threads:!num_threads !model_path in
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

  for pass = 0 to r - 1 do
    Printf.printf "  Pass %d/%d\n%!" (pass + 1) r;

    for batch_idx = 0 to n_batches - 1 do
      let start_px = batch_idx * b in
      let end_px = min (start_px + b) n_valid in
      let actual_b = end_px - start_px in

      (* Zero out the input buffers *)
      Array1.fill s2_input 0.0;
      Array1.fill s1_input 0.0;

      (* Fill batch *)
      for px_in_batch = 0 to actual_b - 1 do
        let px = start_px + px_in_batch in
        let (_global_idx, i, j) = valid_pixels.(px) in

        (* S2 per-pixel data *)
        let s2_bands_pixel = Array.init (Array.length s2_bands) (fun d ->
          s2_bands.(d).(i).(j)) in
        let s2_masks_pixel = Array.init (Array.length s2_masks) (fun d ->
          s2_masks.(d).(i).(j)) in
        let s2_sampled = sample_s2 s2_bands_pixel s2_masks_pixel s2_doys pass in
        let s2_offset = px_in_batch * sample_size_s2 * 11 in
        for k = 0 to sample_size_s2 * 11 - 1 do
          Array1.set s2_input (s2_offset + k) s2_sampled.(k)
        done;

        (* S1 per-pixel data *)
        let asc_pixel = if not s1_asc_empty then
          Array.init (Array.length s1_asc) (fun d -> s1_asc.(d).(i).(j))
        else [| [| 0.0; 0.0 |] |] in
        let asc_d = if not s1_asc_empty then s1_asc_doy
        else [| 180 |] in
        let desc_pixel = if not s1_desc_empty then
          Array.init (Array.length s1_desc) (fun d -> s1_desc.(d).(i).(j))
        else [| [| 0.0; 0.0 |] |] in
        let desc_d = if not s1_desc_empty then s1_desc_doy
        else [| 180 |] in
        let s1_sampled = sample_s1 asc_pixel asc_d desc_pixel desc_d pass in
        let s1_offset = px_in_batch * sample_size_s1 * 3 in
        for k = 0 to sample_size_s1 * 3 - 1 do
          Array1.set s1_input (s1_offset + k) s1_sampled.(k)
        done
      done;

      (* Use sub-views for the last (possibly smaller) batch *)
      let s2_ba = if actual_b = b then s2_input
        else Array1.sub s2_input 0 (actual_b * sample_size_s2 * 11) in
      let s1_ba = if actual_b = b then s1_input
        else Array1.sub s1_input 0 (actual_b * sample_size_s1 * 3) in

      (* Run ONNX inference — batched, using C-cached names *)
      let outputs = Onnxruntime.Session.run_cached_ba session
        [(s2_ba,
          [| Int64.of_int actual_b; Int64.of_int sample_size_s2; 11L |]);
         (s1_ba,
          [| Int64.of_int actual_b; Int64.of_int sample_size_s1; 3L |])]
        ~output_shapes:[[| actual_b * latent_dim |]]
      in
      let output_ba = List.hd outputs in

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
  let rec mkdir_p dir =
    if dir <> "/" && dir <> "." && not (Sys.file_exists dir) then begin
      mkdir_p (Filename.dirname dir);
      Unix.mkdir dir 0o755
    end
  in
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
