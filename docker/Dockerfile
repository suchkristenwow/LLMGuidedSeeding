# Use the official ROS Melodic image
FROM ros:melodic-ros-core

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    sudo \
    tmux \
    bash-completion \
    python3-pip \
    python3-dev \
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
    && rm -rf /var/lib/apt/lists/*

# Setup sources for ROS packages
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# Install Node.js and npm for React application
RUN curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash - && \
    apt-get install -y nodejs

# Set up bash history
RUN touch /root/.bash_history && \
    echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc 

# Switch to non-root user (replace 'myuser' with the desired username)
ARG USERNAME=kristen 
RUN useradd -ms /bin/bash $USERNAME
USER $USERNAME
WORKDIR /home/$USERNAME

# Install scikit-build, setuptools-rust, and cargo
RUN python3 -m pip install --no-cache-dir scikit-build setuptools-rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/home/$USERNAME/.cargo/bin:${PATH}"

# Make port 80 available to the world outside this container
EXPOSE 80

COPY requirements.txt /home/$USERNAME/ 
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy ros_entrypoint.sh
COPY --chown=$USERNAME:$USERNAME ros_entrypoint.sh /ros_entrypoint.sh
RUN sudo chmod +x /ros_entrypoint.sh

# Set up the entrypoint
ENTRYPOINT ["/ros_entrypoint.sh"]