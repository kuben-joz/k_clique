FROM nvidia/cuda:11.8.0-devel-ubuntu22.04


RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    ## Install packages
    && apt-get -y install --no-install-recommends \
    git unzip wget curl \
    ca-certificates build-essential libtool libssl-dev autoconf \
    cppcheck valgrind clang lldb llvm gdb \
    ## Install cmake 3.23.1
    && mkdir -p /tmp/cmake \
    && curl https://cmake.org/files/v3.23/cmake-3.23.1.tar.gz -o /tmp/cmake-3.23.1.tar.gz \
    && tar -xzf /tmp/cmake-3.23.1.tar.gz -C /tmp \
    && /bin/bash /tmp/cmake-3.23.1/bootstrap \
    && make -j$(nproc) && make install \
    && rm -r /tmp/cmake-3.23.1.tar.gz /tmp/cmake-3.23.1 \
    ## Post installation clean-up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ARG INSTALL_ZSH="false"
ARG UPGRADE_PACKAGES="false"
ARG COMMON_SCRIPT_SOURCE="https://raw.githubusercontent.com/microsoft/vscode-dev-containers/main/script-library/common-debian.sh"
ARG COMMON_SCRIPT_SHA="dev-mode"

RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    curl ca-certificates \
    build-essential cmake cppcheck valgrind clang lldb llvm gdb \
    2>&1 \
    ## Download & execute vscode common setup
    && curl -sSL  ${COMMON_SCRIPT_SOURCE} -o /tmp/common-setup.sh \
    && ([ "${COMMON_SCRIPT_SHA}" = "dev-mode" ] || (echo "${COMMON_SCRIPT_SHA} */tmp/common-setup.sh" | sha256sum -c -)) \
    && /bin/bash /tmp/common-setup.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" \
    ## Post installation clean up
    && rm /tmp/common-setup.sh \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*