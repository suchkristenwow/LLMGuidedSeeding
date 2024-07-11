#!/bin/bash
set -e

# Source the ROS setup script
source /opt/ros/melodic/setup.bash

# Execute the command passed to the entrypoint
exec "$@"
