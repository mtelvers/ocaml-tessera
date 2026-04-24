(* Tessera dpixel download tool — downloads S2 and S1 satellite data
   and writes the 7 .npy files in Frank's dpixel format.
   Based on the download half of the original pipeline. *)

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

(* ======================== Nested-to-flat converters ======================== *)

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

(** Convert a URL to a GDAL virtual filesystem path. *)
let gdal_vsi_path url =
  if String.length url > 5 && String.sub url 0 5 = "s3://" then
    "/vsis3/" ^ String.sub url 5 (String.length url - 5)
  else
    "/vsicurl/" ^ url

let warp_id = Atomic.make 0

let warp_to_mem ?(resampling="near") ~dst_crs_wkt ~bounds:(left, bottom, right, top)
    ~width ~height url =
  let id = Atomic.fetch_and_add warp_id 1 in
  let dst = Printf.sprintf "/vsimem/warp_%d" id in
  match Gdal.Dataset.open_ex ~thread_safe:true (gdal_vsi_path url) with
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
      "-r"; resampling;
      "-of"; "MEM";
      "-srcnodata"; "none";
    ] in
    let result = Gdal.Dataset.warp src ~dst_filename:dst opts in
    Gdal.Dataset.close src;
    result

let read_cog_band ?(resampling="near") ~dst_crs_wkt ~bounds ~width ~height url =
  match warp_to_mem ~resampling ~dst_crs_wkt ~bounds ~width ~height url with
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

(* ======================== load_roi ======================== *)

type roi = {
  crs_wkt : string;
  bounds : float * float * float * float;
  bbox : float * float * float * float;
  height : int;
  width : int;
  resolution : float;
  mask : int array array;
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
  let pixel_h = gt.pixel_height in
  let right = left +. (Float.of_int w *. pixel_w) in
  let bottom = top +. (Float.of_int h *. pixel_h) in
  let resolution = Float.abs pixel_w in
  let crs_wkt = Gdal.Dataset.projection ds in
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

(* ======================== process_s2 ======================== *)

let process_s2 ~(roi : roi) ~(client : Stac_client.t) ~data_source ~n_workers items =
  let h = roi.height and w = roi.width in
  let (left, bottom, right, top) = roi.bounds in
  if items = [] then begin
    eprintf "  No S2 items found\n%!";
    ([||], [||], [||])
  end else begin
    eprintf "  Loading SCL (cloud mask)...\n%!";
    let item_dates = Array.of_list (List.map (fun (it : Stac_client.item) ->
      date_of_datetime (match Stac_client.get_datetime it with Some d -> d | None -> "")
    ) items) in
    let items_arr = Array.of_list items in
    let dates = sort_uniq_strings (Array.to_list item_dates) in
    let roi_total = Array.fold_left (fun acc row ->
      acc + Array.fold_left ( + ) 0 row) 0 roi.mask in

    let signed_items = Array.map (fun item -> match data_source with
      | MPC -> Stac_client.sign_planetary_computer client item
      | AWS -> item) items_arr in

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

    let valid_dates = ref [] in
    let day_tile_sel = Hashtbl.create 64 in
    List.iter (fun date_str ->
      let indices = ref [] in
      Array.iteri (fun i d -> if d = date_str then indices := i :: !indices) item_dates;
      let indices = List.rev !indices in
      let n_tiles = List.length indices in

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
      eprintf "  Loading spectral bands...\n%!";
      let valid_dates_arr = Array.of_list valid_dates in
      let date_results = parallel_map ~n_workers (fun date_str ->
        let doy = doy_of_date_str date_str in
        let tile_sel = Hashtbl.find day_tile_sel date_str in
        let indices = ref [] in
        Array.iteri (fun i d -> if d = date_str then indices := i :: !indices) item_dates;
        let indices = List.rev !indices in

        let tile_band_data = List.map (fun item_idx ->
          let access_item = signed_items.(item_idx) in
          Array.map (fun band_name ->
            match get_asset_href access_item band_name with
            | None -> Array.make (h * w) 0.0
            | Some href ->
              read_cog_band ~resampling:"cubic" ~dst_crs_wkt:roi.crs_wkt ~bounds:(left, bottom, right, top)
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

let process_s1_groups ~(roi : roi) ~n_workers ~data_source groups =
  let h = roi.height and w = roi.width in
  let (left, bottom, right, top) = roi.bounds in
  let resampling = match data_source with MPC -> "near" | AWS -> "bilinear" in
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
            let amp = read_cog_band ~resampling ~dst_crs_wkt:roi.crs_wkt ~bounds:(left, bottom, right, top)
                        ~width:w ~height:h url in
            let db = amplitude_to_db amp roi.mask h w in
            let has_any = ref false in
            Array.iter (fun row ->
              Array.iter (fun v -> if v > 0 then has_any := true) row) db;
            if !has_any then Some db else None
        ) group.sg_tiles in
        if db_list = [] then None
        else begin
          match data_source with
          | MPC ->
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
          | AWS ->
            let out = Array.init h (fun _ -> Array.make w 0) in
            List.iter (fun db ->
              for i = 0 to h - 1 do
                for j = 0 to w - 1 do
                  if db.(i).(j) > 0 then
                    out.(i).(j) <- db.(i).(j)
                done
              done
            ) db_list;
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

    (Array.of_list (List.rev !asc_data),
     Array.of_list (List.rev !asc_doys),
     Array.of_list (List.rev !desc_data),
     Array.of_list (List.rev !desc_doys))
  end

(* ======================== OPERA RTC-S1 (AWS) ======================== *)

let read_edl_token () =
  let token_path = Filename.concat (Sys.getenv "HOME") ".edl_bearer_token" in
  let ic = open_in token_path in
  let token = String.trim (input_line ic) in
  close_in ic;
  token

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

(* ======================== NPY save helpers ======================== *)

(* Save bands.npy as uint16 (S2 sensor values fit in 0-65535) *)
let save_bands path (f : flat_4d) =
  let total = f.n * f.h * f.w * f.c in
  let arr = Array.init total (fun i ->
    let v = Array1.get f.data i in
    Float.to_int (Float.max 0.0 (Float.min 65535.0 v))) in
  Npy.save path (Npy.of_int_array Npy.Uint16 [| f.n; f.h; f.w; f.c |] arr)

(* Save masks.npy as uint8 (0 or 1) *)
let save_masks path (f : flat_3d_int) =
  let total = f.n3 * f.h3 * f.w3 in
  let arr = Array.init total (fun i -> Array1.get f.idata i) in
  Npy.save path (Npy.of_int_array Npy.Uint8 [| f.n3; f.h3; f.w3 |] arr)

(* Save S2 doys as uint16 (day-of-year 1-366) *)
let save_doys_uint16 path arr =
  Npy.save path (Npy.of_int_array Npy.Uint16 [| Array.length arr |] arr)

(* Save SAR data as int16 (dB-scaled values fit in 0-32767) *)
let save_sar path (f : flat_4d) =
  let total = f.n * f.h * f.w * f.c in
  let arr = Array.init total (fun i ->
    Float.to_int (Array1.get f.data i)) in
  Npy.save path (Npy.of_int_array Npy.Int16 [| f.n; f.h; f.w; f.c |] arr)

(* Save SAR doys as int16 *)
let save_doys_int16 path arr =
  Npy.save path (Npy.of_int_array Npy.Int16 [| Array.length arr |] arr)

(* ======================== OmniCloudMask ======================== *)

(** Per-channel normalize a (3, H, W) float32 bigarray in-place style.
    For each channel: compute mean/std of non-zero pixels, normalize non-zero,
    leave zeros as 0.0. Returns a new bigarray. *)
let channel_norm_3chw ~h ~w
    (src : (float, float32_elt, c_layout) Array1.t) =
  let out = Array1.create float32 c_layout (3 * h * w) in
  Array1.fill out 0.0;
  for c = 0 to 2 do
    let base = c * h * w in
    (* First pass: mean of non-zero *)
    let sum = ref 0.0 in
    let cnt = ref 0 in
    for k = 0 to h * w - 1 do
      let v = Array1.get src (base + k) in
      if v <> 0.0 then begin
        sum := !sum +. v;
        incr cnt
      end
    done;
    if !cnt > 0 then begin
      let mean = !sum /. Float.of_int !cnt in
      (* Second pass: std *)
      let sq_sum = ref 0.0 in
      for k = 0 to h * w - 1 do
        let v = Array1.get src (base + k) in
        if v <> 0.0 then begin
          let d = v -. mean in
          sq_sum := !sq_sum +. d *. d
        end
      done;
      let std = Float.sqrt (!sq_sum /. Float.of_int !cnt) in
      let std = if std = 0.0 then 1.0 else std in
      (* Third pass: normalize *)
      for k = 0 to h * w - 1 do
        let v = Array1.get src (base + k) in
        if v <> 0.0 then
          Array1.set out (base + k) ((v -. mean) /. std)
      done
    end
  done;
  out

(** Generate overlapping patch coordinates.
    Returns list of (top, bottom, left, right). *)
let make_patch_indexes ~array_height ~array_width ~patch_size ~patch_overlap =
  let stride = patch_size - patch_overlap in
  let max_bottom = array_height - patch_size in
  let max_right = array_width - patch_size in
  let patches = ref [] in
  let top = ref 0 in
  while !top < array_height do
    let t = if !top > max_bottom then max_bottom else !top in
    let bottom = t + patch_size in
    let left = ref 0 in
    while !left < array_width do
      let l = if !left > max_right then max_right else !left in
      let right = l + patch_size in
      patches := (t, bottom, l, right) :: !patches;
      left := !left + stride
    done;
    top := !top + stride
  done;
  List.rev !patches

(** Create gradient blending mask (patch_size x patch_size) as flat float32 bigarray.
    Matches Python create_gradient_mask exactly. *)
let create_gradient_mask ~patch_size ~patch_overlap =
  let ps = patch_size in
  let po = if patch_overlap * 2 > ps then ps / 2 else patch_overlap in
  let grad = Array1.create float32 c_layout (ps * ps) in
  if po > 0 then begin
    (* horizontal gradient: columns *)
    let fpo = Float.of_int po in
    for i = 0 to ps - 1 do
      for j = 0 to ps - 1 do
        let h_val =
          if j < po then Float.of_int (j + 1) /. fpo
          else if j >= ps - po then Float.of_int (ps - j) /. fpo
          else 1.0
        in
        (* vertical gradient: rows (= rotated horizontal) *)
        let v_val =
          if i < po then Float.of_int (i + 1) /. fpo
          else if i >= ps - po then Float.of_int (ps - i) /. fpo
          else 1.0
        in
        Array1.set grad (i * ps + j) (h_val *. v_val)
      done
    done
  end else
    Array1.fill grad 1.0;
  grad

(** Run both OCM sessions on one (3, H, W) normalized image.
    Returns flat (H, W) int array of class predictions (0=clear, 1/2/3=cloud). *)
let run_ocm_on_image ~sess1 ~sess2 ~h ~w ~patch_size ~patch_overlap ~batch_size
    (norm_img : (float, float32_elt, c_layout) Array1.t) =
  let ps = patch_size in
  let patches = make_patch_indexes ~array_height:h ~array_width:w
      ~patch_size:ps ~patch_overlap in
  let gradient = create_gradient_mask ~patch_size:ps ~patch_overlap in

  (* Accumulators: (4, H, W) predictions and (H, W) weight sums *)
  let pred = Array1.create float32 c_layout (4 * h * w) in
  Array1.fill pred 0.0;
  let wsum = Array1.create float32 c_layout (h * w) in
  Array1.fill wsum 0.0;

  (* Deduplicate patches *)
  let seen = Hashtbl.create 256 in
  let unique_patches = List.filter (fun idx ->
    if Hashtbl.mem seen idx then false
    else begin Hashtbl.replace seen idx (); true end
  ) patches in

  (* Process in batches *)
  let patch_arr = Array.of_list unique_patches in
  let n_patches = Array.length patch_arr in
  let batch_idx = ref 0 in
  while !batch_idx < n_patches do
    let actual_b = min batch_size (n_patches - !batch_idx) in

    (* Extract patches into (B, 3, ps, ps) flat bigarray *)
    let patch_ba = Array1.create float32 c_layout (actual_b * 3 * ps * ps) in
    Array1.fill patch_ba 0.0;
    for b = 0 to actual_b - 1 do
      let (top, _bottom, left, _right) = patch_arr.(!batch_idx + b) in
      for c = 0 to 2 do
        for pi = 0 to ps - 1 do
          for pj = 0 to ps - 1 do
            let si = top + pi and sj = left + pj in
            let src_idx = c * h * w + si * w + sj in
            let dst_idx = ((b * 3 + c) * ps + pi) * ps + pj in
            Array1.set patch_ba dst_idx (Array1.get norm_img src_idx)
          done
        done
      done
    done;

    (* Check if entire batch is zeros — skip if so *)
    let all_zero = ref true in
    for k = 0 to Array1.dim patch_ba - 1 do
      if Array1.get patch_ba k <> 0.0 then all_zero := false
    done;

    if not !all_zero then begin
      (* Run both models *)
      let shape = [| Int64.of_int actual_b; 3L;
                     Int64.of_int ps; Int64.of_int ps |] in
      let out_size = actual_b * 4 * ps * ps in
      let out1 = Onnxruntime.Session.run_ba sess1
        [| ("input", patch_ba, shape) |]
        [| "output" |]
        ~output_sizes:[| out_size |] in
      let out2 = Onnxruntime.Session.run_ba sess2
        [| ("input", patch_ba, shape) |]
        [| "output" |]
        ~output_sizes:[| out_size |] in
      let o1 = out1.(0) and o2 = out2.(0) in

      (* Average and accumulate with gradient blending *)
      for b = 0 to actual_b - 1 do
        let (top, _bottom, left, _right) = patch_arr.(!batch_idx + b) in
        for cls = 0 to 3 do
          for pi = 0 to ps - 1 do
            for pj = 0 to ps - 1 do
              let oidx = ((b * 4 + cls) * ps + pi) * ps + pj in
              let avg = (Array1.get o1 oidx +. Array1.get o2 oidx) *. 0.5 in
              let gw = Array1.get gradient (pi * ps + pj) in
              let si = top + pi and sj = left + pj in
              let pidx = cls * h * w + si * w + sj in
              Array1.set pred pidx (Array1.get pred pidx +. avg *. gw)
            done
          done
        done;
        (* Accumulate gradient weights *)
        for pi = 0 to ps - 1 do
          for pj = 0 to ps - 1 do
            let gw = Array1.get gradient (pi * ps + pj) in
            let si = top + pi and sj = left + pj in
            let widx = si * w + sj in
            Array1.set wsum widx (Array1.get wsum widx +. gw)
          done
        done
      done
    end;

    batch_idx := !batch_idx + actual_b
  done;

  (* Argmax over 4 classes → (H, W) int array *)
  let result = Array.make (h * w) 0 in
  for i = 0 to h - 1 do
    for j = 0 to w - 1 do
      let idx = i * w + j in
      let ws = Array1.get wsum idx in
      if ws > 0.0 then begin
        let best_cls = ref 0 in
        let best_val = ref neg_infinity in
        for cls = 0 to 3 do
          let v = Array1.get pred (cls * h * w + idx) /. ws in
          if v > !best_val then begin best_val := v; best_cls := cls end
        done;
        result.(idx) <- !best_cls
      end
    done
  done;
  result

(** Run OmniCloudMask on all timesteps. Returns flat (T*H*W) int array for mask_optimized. *)
let run_ocm ~model1_path ~model2_path ~n_threads ~cuda_device
    ~patch_size ~patch_overlap ~batch_size
    ~(s2_bands_data : flat_4d) ~(s2_masks_data : flat_3d_int) =
  let n_t = s2_bands_data.n in
  let h = s2_bands_data.h and w = s2_bands_data.w in
  Printf.printf "  OCM: %d timesteps, %dx%d\n%!" n_t h w;

  let env = Onnxruntime.Env.create ~log_level:3 "ocm" in
  let cuda_opt = if cuda_device >= 0 then Some cuda_device else None in
  let sess1 = Onnxruntime.Session.create env ~threads:n_threads ?cuda_device:cuda_opt model1_path in
  let sess2 = Onnxruntime.Session.create env ~threads:n_threads ?cuda_device:cuda_opt model2_path in

  (* Auto patch size *)
  let ps = if patch_size <= 0 then min 1000 (min h w) else patch_size in
  let ps = max ps 32 in
  let po = min patch_overlap (ps / 2) in
  Printf.printf "  OCM patch_size=%d, overlap=%d, batch_size=%d\n%!" ps po batch_size;

  let mask_opt = Array.make (n_t * h * w) 0 in

  for t = 0 to n_t - 1 do
    (* Extract R(B04=idx0), G(B03=idx2), NIR(B8A=idx4) into (3, H, W) float32 *)
    let rgn = Array1.create float32 c_layout (3 * h * w) in
    let band_indices = [| 0; 2; 4 |] in
    for c = 0 to 2 do
      let bi = band_indices.(c) in
      for i = 0 to h - 1 do
        for j = 0 to w - 1 do
          let src_idx = ((t * h + i) * w + j) * 10 + bi in
          let v = Array1.get s2_bands_data.data src_idx in
          Array1.set rgn (c * h * w + i * w + j) v
        done
      done
    done;

    (* Channel normalize *)
    let norm = channel_norm_3chw ~h ~w rgn in

    (* Run OCM *)
    let ocm_class = run_ocm_on_image ~sess1 ~sess2 ~h ~w
        ~patch_size:ps ~patch_overlap:po ~batch_size norm in

    (* Merge: mask_optimized[t] = (scl_mask==1) AND (ocm_class==0) AND (not all-zero) *)
    for i = 0 to h - 1 do
      for j = 0 to w - 1 do
        let px = i * w + j in
        let scl_valid = Array1.get s2_masks_data.idata (t * h * w + px) = 1 in
        let ocm_clear = ocm_class.(px) = 0 in
        (* Check if all 3 channels are zero (nodata) *)
        let all_zero =
          Array1.get rgn (0 * h * w + px) = 0.0 &&
          Array1.get rgn (1 * h * w + px) = 0.0 &&
          Array1.get rgn (2 * h * w + px) = 0.0 in
        if scl_valid && ocm_clear && not all_zero then
          mask_opt.(t * h * w + px) <- 1
      done
    done;

    if (t + 1) mod 10 = 0 || t = n_t - 1 then
      Printf.printf "  [%d/%d] timesteps processed\n%!" (t + 1) n_t
  done;
  { idata = Array1.of_array int c_layout mask_opt;
    n3 = n_t; h3 = h; w3 = w }

(* ======================== Main ======================== *)

let () =
  let input_tiff = ref "" in
  let output_dir = ref "" in
  let start_date = ref "2024-01-01" in
  let end_date = ref "2024-12-31" in
  let max_cloud = ref 100.0 in
  let data_source_str = ref "mpc" in
  let download_workers = ref 32 in
  let ocm_model_1 = ref "" in
  let ocm_model_2 = ref "" in
  let ocm_patch_size = ref 0 in
  let ocm_batch_size = ref 16 in
  let ocm_patch_overlap = ref 300 in
  let ocm_threads = ref 4 in
  let flat_output = ref false in
  let cuda_device = ref (-1) in

  let speclist = [
    ("--input_tiff", Arg.Set_string input_tiff, "Path to grid GeoTIFF");
    ("--output", Arg.Set_string output_dir, "Output directory for dpixel .npy files");
    ("--start", Arg.Set_string start_date, "Start date (YYYY-MM-DD)");
    ("--end", Arg.Set_string end_date, "End date (YYYY-MM-DD)");
    ("--max_cloud", Arg.Set_float max_cloud, "Max cloud cover %");
    ("--data_source", Arg.Set_string data_source_str, "Data source: mpc or aws (default: mpc)");
    ("--download_workers", Arg.Set_int download_workers, "Parallel GDAL download workers (default: 32)");
    ("--ocm_model_1", Arg.Set_string ocm_model_1, "Path to first OCM ONNX model (OCM skipped if not set)");
    ("--ocm_model_2", Arg.Set_string ocm_model_2, "Path to second OCM ONNX model (OCM skipped if not set)");
    ("--ocm_patch_size", Arg.Set_int ocm_patch_size, "OCM patch size, 0=auto (default: 0)");
    ("--ocm_batch_size", Arg.Set_int ocm_batch_size, "OCM patches per batch (default: 16)");
    ("--ocm_patch_overlap", Arg.Set_int ocm_patch_overlap, "OCM overlap pixels (default: 300)");
    ("--ocm_threads", Arg.Set_int ocm_threads, "OCM ONNX Runtime threads (default: 4)");
    ("--flat_output", Arg.Set flat_output, "Write .npy files directly to --output (no subdirectory)");
    ("--cuda", Arg.Set_int cuda_device, "CUDA device ID (e.g. 0) for GPU OCM inference");
  ] in
  Arg.parse speclist (fun _ -> ()) "Tessera dpixel download tool";

  let data_source = match String.lowercase_ascii !data_source_str with
    | "mpc" -> MPC
    | "aws" -> AWS
    | s -> failwith (Printf.sprintf "Unknown data source: %s (expected mpc or aws)" s)
  in

  if !input_tiff = "" then failwith "--input_tiff is required";
  if !output_dir = "" then failwith "--output is required";

  let grid_id = Filename.remove_extension (Filename.basename !input_tiff) in
  let date_range = !start_date ^ "/" ^ !end_date in
  let out_dir = if !flat_output then !output_dir
                else Filename.concat !output_dir grid_id in

  (* Skip if already complete *)
  if Sys.file_exists (Filename.concat out_dir "bands.npy") then begin
    Printf.printf "Output already exists: %s\nSkipping.\n%!" out_dir;
    exit 0
  end;

  let t_start = Unix.gettimeofday () in
  Printf.printf "\nDpixel download started: %s\n%!" grid_id;

  let n_workers = !download_workers in
  let client = Stac_client.make () in
  let roi = load_roi !input_tiff in
  Printf.printf "  Tile: %dx%d, resolution=%.6f\n%!" roi.height roi.width roi.resolution;
  let (xmin, ymin, xmax, ymax) = roi.bbox in

  (* GDAL HTTP tuning for remote COG access *)
  Gdal.set_config_option "GDAL_DISABLE_READDIR_ON_OPEN" "EMPTY_DIR";
  Gdal.set_config_option "GDAL_HTTP_MULTIRANGE" "YES";
  Gdal.set_config_option "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES" "YES";
  Gdal.set_config_option "GDAL_HTTP_MULTIPLEX" "YES";
  Gdal.set_config_option "GDAL_HTTP_VERSION" "2";
  Gdal.set_config_option "CPL_VSIL_CURL_CACHE_SIZE" "134217728";
  Gdal.set_config_option "GDAL_HTTP_MAX_RETRY" "3";
  Gdal.set_config_option "GDAL_HTTP_RETRY_DELAY" "1";

  (* Sentinel-2 *)
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
  let s2_bands_data = flatten_4d s2b in
  let s2_masks_data = flatten_3d_int s2m in
  let s2_doys_data = s2d in
  let t_s2_end = Unix.gettimeofday () in
  Printf.printf "  Result: %d valid days (%.1fs)\n%!" (Array.length s2b) (t_s2_end -. t_s2_start);

  (* Sentinel-1 *)
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
  let (s1a, s1ad, s1de, s1dd) = process_s1_groups ~roi ~n_workers ~data_source s1_groups in
  (if data_source = AWS then
    Gdal.set_config_option "GDAL_HTTP_HEADER_FILE" "");
  let s1_asc_data = flatten_4d s1a in
  let s1_asc_doy_data = s1ad in
  let s1_desc_data = flatten_4d s1de in
  let s1_desc_doy_data = s1dd in
  let t_s1_end = Unix.gettimeofday () in
  Printf.printf "  Ascending: %d passes, Descending: %d passes (%.1fs)\n%!"
    (Array.length s1a) (Array.length s1de) (t_s1_end -. t_s1_start);

  (* Save dpixel .npy files *)
  mkdir_p out_dir;
  Printf.printf "\nSaving dpixel data to %s\n%!" out_dir;
  save_bands (Filename.concat out_dir "bands.npy") s2_bands_data;
  save_masks (Filename.concat out_dir "masks.npy") s2_masks_data;
  save_doys_uint16 (Filename.concat out_dir "doys.npy") s2_doys_data;
  save_sar (Filename.concat out_dir "sar_ascending.npy") s1_asc_data;
  save_doys_int16 (Filename.concat out_dir "sar_ascending_doy.npy") s1_asc_doy_data;
  save_sar (Filename.concat out_dir "sar_descending.npy") s1_desc_data;
  save_doys_int16 (Filename.concat out_dir "sar_descending_doy.npy") s1_desc_doy_data;

  Printf.printf "  Saved 7 .npy files\n%!";

  (* OmniCloudMask — only run if both model paths were supplied *)
  if !ocm_model_1 <> "" && !ocm_model_2 <> "" then begin
    if not (Sys.file_exists !ocm_model_1) then
      failwith (Printf.sprintf "OCM model not found: %s" !ocm_model_1);
    if not (Sys.file_exists !ocm_model_2) then
      failwith (Printf.sprintf "OCM model not found: %s" !ocm_model_2);
    if s2_bands_data.n > 0 then begin
      Printf.printf "\nRunning OmniCloudMask...\n%!";
      let t_ocm_start = Unix.gettimeofday () in
      let mask_opt = run_ocm
        ~model1_path:!ocm_model_1
        ~model2_path:!ocm_model_2
        ~n_threads:!ocm_threads
        ~cuda_device:!cuda_device
        ~patch_size:!ocm_patch_size
        ~patch_overlap:!ocm_patch_overlap
        ~batch_size:!ocm_batch_size
        ~s2_bands_data ~s2_masks_data in
      save_masks (Filename.concat out_dir "mask_optimized.npy") mask_opt;
      let t_ocm_end = Unix.gettimeofday () in
      Printf.printf "  Saved mask_optimized.npy (%.1fs)\n%!" (t_ocm_end -. t_ocm_start)
    end else
      Printf.printf "\nSkipping OCM: no S2 timesteps\n%!"
  end else
    Printf.printf "\nSkipping OCM: no --ocm_model_1/--ocm_model_2 given\n%!";

  let t_end = Unix.gettimeofday () in
  Printf.printf "\nDone: %s (%.1fs)\n%!" grid_id (t_end -. t_start)
