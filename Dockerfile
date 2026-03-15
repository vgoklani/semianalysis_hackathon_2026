ARG PLATFORM=amd64
ARG NGC_PYTORCH_VERSION=26.02

FROM --platform=${PLATFORM} nvcr.io/nvidia/pytorch:${NGC_PYTORCH_VERSION}-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

ARG MAX_JOBS=32
ENV MAX_JOBS=${MAX_JOBS}

# NVTE => NVIDIA TRANSFORMER ENGINE
ENV NVTE_FRAMEWORK=pytorch,

ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /workspace

RUN ln -sf /usr/share/zoneinfo/US/Eastern /etc/localtime

RUN pip uninstall -y ninja && pip install --upgrade --quiet ninja packaging pip

COPY ./requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt

COPY ./manager.jupyterlab-settings /root/.jupyter/lab/user-settings/@jupyterlab/completer-extension

RUN jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

RUN pip uninstall -y pynvml || true
RUN RUN pip show pynvml && pip uninstall -y pynvml || echo "pynvml not installed, skipping"

# RUN pip3 install --force-reinstall --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130

# RUN pip install --upgrade --force-reinstall triton
# RUN pip3 install triton_kernels

# COPY ./build_flash_attention__components.sh /root/build_flash_attention__components.sh
# RUN /root/build_flash_attention__components.sh

EXPOSE 8888

VOLUME ["/data"]
VOLUME ["/log"]
VOLUME ["/src"]

WORKDIR /root/src

CMD ["sh", "-c", "jupyter lab --port=8888 --no-browser --ip=* --allow-root"]

# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
