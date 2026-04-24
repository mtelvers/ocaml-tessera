(lang dune 3.0)
(name tessera)

(generate_opam_files true)

(source (github mtelvers/ocaml-tessera))
(license MIT)
(authors "Mark Elvers <mark.elvers@tunbury.org>")
(maintainers "Mark Elvers <mark.elvers@tunbury.org>")

(package
 (name pipeline)
 (synopsis "Tessera Pipeline in OCaml")
 (description "Tessera Pipeline in OCaml")
 (depends
  (ocaml (>= 4.14))
  (dune (>= 3.0))
  stac_client
  gdal
  onnxruntime
  yojson))

(package
 (name dpixel)
 (synopsis "Tessera dpixel tool in OCaml")
 (description "Tessera dpixel tool in OCaml")
 (depends
  (ocaml (>= 4.14))
  (dune (>= 3.0))
  stac_client
  gdal
  npy
  onnxruntime
  yojson
  (cmdliner (>= 2.0))))
