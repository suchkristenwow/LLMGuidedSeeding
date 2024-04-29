#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    ros::init(argc, argv, "endpoint_marker_publisher");
    ros::NodeHandle nh;

    // Retrieve param string for endpoint x, y, z, radius
    std::string scene_endpoint;
    if (!nh.getParam("endpoint_marker_publisher/scene_endpoint", scene_endpoint)) {
        ROS_WARN("Failed to get parameter 'endpoint_marker_publisher/scene_endpoint'");
        return 1;
    }
    
    // Set up the endpoint marker
    std::istringstream iss(scene_endpoint);
    double x, y, z, radius;
    if (!(iss >> x >> y >> z >> radius)) {
        ROS_WARN("Failed to parse parameter 'endpoint_marker_publisher/scene_endpoint'");
        return 1;
    }

    // Convert from cm to m and from Unreal to ROS coordinate system
    x = x / 100.0;
    y = -1.0 * (y / 100.0);
    z = z / 100.0;
    radius = radius / 100.0;
    
    // Create a publisher for the marker
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("endpoint_marker", 100);

    // Endpoint marker message
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.ns = "basic_shapes";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = z;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = radius * 2;  
    marker.scale.y = radius * 2;  
    marker.scale.z = radius * 2;
    marker.color.r = 1.0f;
    marker.color.g = 0.0f;
    marker.color.b = 0.0f;
    marker.color.a = 0.4;

    // Publish the marker
    ros::Rate loop_rate(10.0); 
    while (ros::ok()) {
        marker_pub.publish(marker);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
