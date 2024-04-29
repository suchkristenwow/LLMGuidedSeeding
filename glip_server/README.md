## GLIP Server

### Installation
To install locally create a conda environment and run:
```
bash install.sh
```

### Running
To run the server run:
```
python glip_server.py
```

### Docker Image
To build a docker image run
``` 
bash build_docker.sh VERSION_TAG
```
This will create a docker image with the name `glip_server:VERSION_TAG`

Exmaple:
```
bash build_docker.sh 1.0
```
will give the following image: `glip_server:1.0`

### Important:
Your docker runtime must have nvidia gpu support. To enable this install the Nvidia Container Runtime
(`setup_scripts/setup_nvidia_docker.sh`) and add the following to your docker daemon.json file ('/etc/docker/daemon.json')

``` 
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         } 
    },
    "default-runtime": "nvidia" 
}
```

For more details see [this](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime)


