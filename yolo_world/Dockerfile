FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
WORKDIR /yolo_world_server
RUN apt-get update && \
    apt-get install -y python3-pip python-is-python3 wget libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
COPY . /yolo_world_server/
RUN chmod +x install.sh
RUN chmod +x yolo_world_server_with_seg.py
RUN bash install.sh
RUN python yolo_inference_docker_init.py
ENTRYPOINT ["python3", "yolo_world_server_with_seg.py"]