FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends cmake build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY cmake cmake
COPY include include
COPY src src
COPY cli cli
COPY third_party third_party
COPY tests tests
ARG CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90"
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVV_ENABLE_TRT=OFF \
    -DVV_BUILD_BENCH=OFF -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    && cmake --build build --parallel 2 \
    && ctest --test-dir build --output-on-failure

# Which version this is. Declared after the build above on purpose: an
# argument that reached that layer would invalidate it on every commit and
# recompile five CUDA architectures for a string. Here it re-stamps one
# generated header, recompiles version.c and relinks -- seconds. The context
# has no .git, so CI passes what cmake/Version.cmake derived from the tag;
# a bare `docker build .` calls itself <header>-local.
ARG VV_VERSION=
ARG VV_REVISION=unknown
RUN v="${VV_VERSION:-$(sed -n 's/^#define VV_VERSION_STRING "\(.*\)"$/\1/p' include/vibevoice/vibevoice.h)-local}" \
    && cmake -B build -DVV_VERSION_FULL="$v" -DVV_REVISION="${VV_REVISION}" \
    && cmake --build build --target vv_cli --parallel 2 \
    && strip build/vv_cli \
    && build/vv_cli --version | tee /dev/stderr | grep -qF "vibevoice.c $v "

FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/vv_cli /usr/local/bin/vv_cli

ARG VV_VERSION=
ARG VV_REVISION=unknown
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
