# Use the official ROS Melodic image
FROM ros:melodic-ros-core

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    sudo \
    tmux \
    bash-completion \
    vim \
    wget \
    build-essential \
    pkg-config \
    libssl-dev \
    libjpeg-dev \
    rustc \
    curl \
    python2.7 \
    ros-melodic-image-geometry \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.9 manually
RUN wget https://www.python.org/ftp/python/3.9.7/Python-3.9.7.tgz && \
    tar xzf Python-3.9.7.tgz && \
    cd Python-3.9.7 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.9.7 Python-3.9.7.tgz

# Install pip for Python 3.9
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# Setup sources for ROS packages
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# Install Node.js and npm for React application
RUN curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash - && \
    apt-get install -y nodejs

# Set up bash history
RUN touch /root/.bash_history && \
    echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc 

# Create non-root user with a specific UID and GID
ARG USERNAME=kristen
ARG USER_UID=1001
ARG USER_GID=1001

RUN groupadd -g $USER_GID $USERNAME && \
    useradd -m -s /bin/bash -u $USER_UID -g $USER_GID $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to non-root user
USER $USERNAME
WORKDIR /home/$USERNAME

# Install scikit-build, setuptools-rust, and cargo
RUN python3.9 -m pip install --no-cache-dir scikit-build setuptools-rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/home/$USERNAME/.cargo/bin:${PATH}"

# Make port 80 available to the world outside this container
EXPOSE 80

COPY requirements.txt /home/$USERNAME/
RUN python3.9 -m pip install --no-cache-dir -r requirements.txt

# Copy ros_entrypoint.sh
COPY --chown=$USERNAME:$USERNAME ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

# Set up the entrypoint
ENTRYPOINT ["/ros_entrypoint.sh"]
