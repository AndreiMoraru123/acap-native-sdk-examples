ARG ARCH=aarch64
ARG VERSION=12.6.0
ARG UBUNTU_VERSION=24.04
ARG REPO=axisecp
ARG SDK=acap-native-sdk

FROM ${REPO}/${SDK}:${VERSION}-${ARCH}-ubuntu${UBUNTU_VERSION}

#-------------------------------------------------------------------------------
# Install TensorFlow (only used to inspect the model)
#-------------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment for installations using pip
RUN python3 -m venv /opt/venv

# hadolint ignore=SC1091,DL3013
RUN . /opt/venv/bin/activate && pip install --no-cache-dir tensorflow

#-------------------------------------------------------------------------------
# Get YOLOv5 model and labels
#-------------------------------------------------------------------------------

# Use YOLOv8 license plate model
WORKDIR /opt/app/model
COPY ./Reader.tflite model.tflite
ARG CHIP=artpec8

# Use created labels for model
WORKDIR /opt/app/label
COPY ./labels_license_plate.txt labels.txt

#-------------------------------------------------------------------------------
# Build ACAP application
#-------------------------------------------------------------------------------

WORKDIR /opt/app
COPY ./app .

# Extract model parameters using the virtual environment
# hadolint ignore=SC1091
RUN . /opt/venv/bin/activate && python parameter_finder.py 'model/model.tflite'

RUN cp /opt/app/manifest.json.${CHIP} /opt/app/manifest.json && \
    . /opt/axis/acapsdk/environment-setup* && acap-build . \
    -a 'label/labels.txt' \
    -a 'model/model.tflite'
