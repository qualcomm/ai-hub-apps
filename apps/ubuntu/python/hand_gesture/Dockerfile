FROM ubuntu:24.04

RUN apt update && apt install -y software-properties-common
RUN apt-add-repository -y ppa:ubuntu-qcom-iot/qcom-ppa
RUN apt update && apt install -y \
    wget \
    libcairo2-dev \
    pkg-config \
    libgirepository1.0-dev \
    gir1.2-gstreamer-1.0 \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    unzip \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-qcom-qmmfsrc \
    gstreamer1.0-plugins-qcom-vtransform \
    v4l2loopback-dkms \
    v4l2loopback-utils \
    linux-headers-$(uname -r)

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ENV LD_LIBRARY_PATH="/usr/lib/qcm6490:/usr/lib/aarch64-linux-gnu:/opt/host/lib"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV MESA_LOADER_DRIVER_OVERRIDE=msm
ENV LIBGL_DRIVERS_PATH=/lib/aarch64-linux-gnu/dri
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_adreno.json

RUN usermod -aG video root

WORKDIR /root

ENTRYPOINT ["python", "main.py"]
