# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# PATH에 Conda 경로 추가
ENV PATH="/opt/conda/bin:$PATH"

# 필요한 개발 도구들 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    wget \
    cmake \
    meson \
    ninja-build \
    pkg-config \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Python3 심볼릭 링크 생성
RUN ln -sf /opt/conda/bin/python /usr/local/bin/python3

# Python 패키지 설치
RUN pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3

# libspng 설치
RUN git clone https://github.com/randy408/libspng.git && \
    cd libspng && \
    git checkout v0.6.1 && \
    meson setup build && \
    cd build && \
    ninja && \
    ninja install && \
    ldconfig && \
    cd ../.. && \
    rm -rf libspng

WORKDIR /workspace

# bash 셸을 명시적으로 설정
SHELL ["/bin/bash", "-c"]

RUN echo '#!/bin/bash' > /entry.sh && \
    echo 'unset TORCH_CUDA_ARCH_LIST' >> /entry.sh && \
    echo 'exec "$@"' >> /entry.sh && \
    chmod +x /entry.sh

ENTRYPOINT ["/bin/bash"]