FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
WORKDIR /glip_server
RUN apt-get update && \
    apt-get install -y python3-pip python-is-python3 wget && \
    rm -rf /var/lib/apt/lists/*
COPY . /glip_server/
RUN chmod +x install.sh
RUN chmod +x glip_server.py
RUN bash install.sh --skip-clone
ENTRYPOINT ["python3", "glip_server.py"]