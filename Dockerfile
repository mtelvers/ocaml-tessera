# syntax=docker/dockerfile:1
FROM ocaml/opam:debian-13-ocaml-5.3 AS build
RUN sudo ln -sf /usr/bin/opam-2.5 /usr/bin/opam && opam init --reinit -ni
RUN sudo rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' | sudo tee /etc/apt/apt.conf.d/keep-cache
RUN sudo apt update && sudo apt-get --no-install-recommends install -y \
    libcurl4-openssl-dev \
    libffi-dev \
    libgmp-dev \
    libgdal-dev \
    pkg-config \
    m4 \
    curl
# Install ONNX Runtime (used by dpixel for OmniCloudMask inference)
RUN curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz | \
    sudo tar xz -C /usr/local --strip-components=1
RUN sudo ldconfig
# Pin external dependencies
RUN opam pin add stac_client https://github.com/mtelvers/stac-client.git -n && \
    opam pin add gdal https://github.com/mtelvers/ocaml-gdal.git -n && \
    opam pin add onnxruntime https://github.com/mtelvers/ocaml-onnxruntime.git -n && \
    opam pin add npy https://github.com/mtelvers/ocaml-npy.git -n && \
    opam install -y stac_client gdal onnxruntime npy yojson
WORKDIR /src
COPY --chown=opam --link dune-project dune-workspace ./
COPY --chown=opam --link bin/dpixel.ml bin/
RUN echo '(executable (name dpixel) (libraries stac_client gdal npy onnxruntime yojson unix bigarray))' > bin/dune
RUN opam exec -- dune build bin/dpixel.exe

FROM debian:13
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN apt update && apt-get --no-install-recommends install -y \
    ca-certificates \
    curl \
    libgdal36 \
    libcurl4 \
    libffi8 && \
    ln -s /usr/lib/x86_64-linux-gnu/libgdal.so.36 /usr/lib/x86_64-linux-gnu/libgdal.so
# Install ONNX Runtime in runtime image
RUN curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz | \
    tar xz -C /usr/local --strip-components=1 && ldconfig
COPY --from=build --link /src/_build/default/bin/dpixel.exe /usr/local/bin/dpixel
ENTRYPOINT ["/usr/local/bin/dpixel"]
