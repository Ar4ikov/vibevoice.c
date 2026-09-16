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

# Which build this is, declared here and not in the stage above on purpose:
# an argument that reached the `cmake` layer would invalidate the cache on
# every commit and recompile five CUDA architectures for a string. The
# binary reads VV_BUILD_REF at startup, so `--version`, /health and the
# vibevoice_build_info metric all name the tag the image was published as.
ARG VV_VERSION=dev
ARG VV_REVISION=unknown
ENV VV_BUILD_REF=${VV_VERSION}
LABEL org.opencontainers.image.title="vibevoice.c" \
      org.opencontainers.image.description="VibeVoice-ASR in C and CUDA: transcription with diarization and timestamps, OpenAI-compatible server" \
      org.opencontainers.image.source="https://github.com/Ar4ikov/vibevoice.c" \
      org.opencontainers.image.documentation="https://github.com/Ar4ikov/vibevoice.c/blob/master/deploy/gpustack/README.md" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VV_VERSION}" \
      org.opencontainers.image.revision="${VV_REVISION}"

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/vv_cli"]
CMD ["serve", "--help"]
