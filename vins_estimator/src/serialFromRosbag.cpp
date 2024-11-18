/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <stdio.h>

#include <chrono>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>

#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"

// To read rosbag
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

Estimator estimator;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
std::mutex m_buf;

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img0_buf.push(img_msg);
    m_buf.unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img1_buf.push(img_msg);
    m_buf.unlock();
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1") {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

// extract images with same timestamp from two topics
void sync_process()
{
    if (STEREO) {
        cv::Mat image0, image1;
        std_msgs::Header header;
        double time = 0;
        m_buf.lock();
        if (!img0_buf.empty() && !img1_buf.empty()) {
            double time0 = img0_buf.front()->header.stamp.toSec();
            double time1 = img1_buf.front()->header.stamp.toSec();
            // 0.003s sync tolerance
            if (time0 < time1 - 0.003) {
                img0_buf.pop();
                printf("throw img0\n");
            }
            else if (time0 > time1 + 0.003) {
                img1_buf.pop();
                printf("throw img1\n");
            }
            else {
                time = img0_buf.front()->header.stamp.toSec();
                header = img0_buf.front()->header;
                image0 = getImageFromMsg(img0_buf.front());
                img0_buf.pop();
                image1 = getImageFromMsg(img1_buf.front());
                img1_buf.pop();
                // printf("find img0 and img1\n");
            }
        }
        m_buf.unlock();
        if (!image0.empty()) estimator.inputImage(time, image0, image1);
    }
    else {
        cv::Mat image;
        std_msgs::Header header;
        double time = 0;
        m_buf.lock();
        if (!img0_buf.empty()) {
            time = img0_buf.front()->header.stamp.toSec();
            header = img0_buf.front()->header;
            image = getImageFromMsg(img0_buf.front());
            img0_buf.pop();
        }
        m_buf.unlock();
        if (!image.empty()) estimator.inputImage(time, image);
    }

    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
    return;
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++) {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if (feature_msg->channels.size() > 5) {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            // printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }
    double t = feature_msg->header.stamp.toSec();
    estimator.inputFeature(t, featureFrame);
    return;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true) {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}

void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true) {
        // ROS_WARN("use IMU!");
        estimator.changeSensorType(1, STEREO);
    }
    else {
        // ROS_WARN("disable IMU!");
        estimator.changeSensorType(0, STEREO);
    }
    return;
}

void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true) {
        // ROS_WARN("use stereo!");
        estimator.changeSensorType(USE_IMU, 1);
    }
    else {
        // ROS_WARN("use mono camera (left)!");
        estimator.changeSensorType(USE_IMU, 0);
    }
    return;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Usage: rosrun vins vins_from_rosbag [full path to config file] [rosbag_path]\n");
        return 1;
    }

    ros::init(argc, argv, "vins_from_rosbag");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    string config_file = argv[1];
    ROS_INFO("config_file: %s\n", argv[1]);

    readParameters(config_file);
    estimator.setParameter();

    const std::string rosbag_path = argv[2];
    ROS_INFO("Opening rosbag: %s", rosbag_path.c_str());
    // Open rosbag
    rosbag::Bag bag;
    bag.open(rosbag_path, rosbag::bagmode::Read);

    // Setup ROS
    registerPub(n);

    // Read topics from rosbag
    std::vector<std::string> topics_to_read;
    topics_to_read.push_back(std::string(IMAGE0_TOPIC));
    topics_to_read.push_back(std::string(IMU_TOPIC));
    if (STEREO) {
        topics_to_read.push_back(std::string(IMAGE1_TOPIC));
    }

    rosbag::View view(bag, rosbag::TopicQuery(topics_to_read));

    // Check if we are faster than just rosbag play :)
    auto loop_start = std::chrono::high_resolution_clock::now();

    for (rosbag::MessageInstance const m : view) {
        ROS_DEBUG("Read topic [%s]", m.getTopic().c_str());
        if (!ros::ok()) {
            break;
        }

        if (m.getTopic() == std::string(IMU_TOPIC)) {
            sensor_msgs::ImuConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
            imu_callback(imu_msg);
        }
        else if (m.getTopic() == std::string(IMAGE0_TOPIC)) {
            sensor_msgs::ImageConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
            img0_callback(img_msg);
        }
        else if (m.getTopic() == std::string(IMAGE1_TOPIC)) {
            sensor_msgs::ImageConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
            img1_callback(img_msg);
        }
        else {
            ROS_WARN("Ignoring topic %s", m.getTopic().c_str());
            continue;
        }

        // Sync process
        if (m.getTopic() != std::string(IMU_TOPIC)) {
            sync_process();

            // Measure time since last loop
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - loop_start);
            loop_start = now;
            ROS_DEBUG_THROTTLE(1, "Loop duration: %0.2fd ms | Effective freq: %0.2f Hz",
                               duration.count(), 1000.0 / duration.count());
        }
    }

    bag.close();
    return 0;
}
