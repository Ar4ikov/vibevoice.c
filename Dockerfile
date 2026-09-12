FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends cmake build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY include include
COPY src src
COPY cli cli
COPY third_party third_party
COPY tests tests
ARG CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90"
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVV_ENABLE_TRT=OFF \
    -DVV_BUILD_BENCH=OFF -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    && cmake --build build --parallel 2 \
    && ctest --test-dir build --output-on-failure \
    && strip build/vv_cli

FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/vv_cli /usr/local/bin/vv_cli
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/vv_cli"]
CMD ["serve", "--help"]
