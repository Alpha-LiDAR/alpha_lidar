#!/usr/bin/env python3
# coding=utf-8
from __future__ import print_function, division, absolute_import

import copy
import os
import struct

import cv2
import message_filters
import numpy as np
import numpy.linalg as LA
import ros_numpy
import rospy
import transforms3d
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image

frame_id = 0

T__world__o__odom = np.eye(4)


def compose(transform_matrix, point_cloud, keep_data=False):
    if type(transform_matrix) is list:
        return np.row_stack([compose(transform, points) for transform, points in zip(transform_matrix, point_cloud)])
    else:
        if len(point_cloud.shape) == 2:
            homogeneous_points = np.column_stack([point_cloud[:, :3], np.ones(len(point_cloud))])
            if not keep_data:
                return np.matmul(transform_matrix, homogeneous_points.T).T
            else:
                return np.column_stack([np.matmul(transform_matrix, homogeneous_points.T).T[:, :3], point_cloud[:, 3:]])
        else:
            # single point
            assert len(point_cloud.shape) == 1
            return np.squeeze(np.matmul(transform_matrix[:3, :3], point_cloud.reshape(3, 1)).T + transform_matrix[:3, 3])


def save_KITTI_bin(points, path):
    """
    save pointcloud file (KITTI format)
    :param points: pointcloud, numpy array of shape[N, 4], (x,y,z,intensity)
    :param path: save path
    :return:
    """
    points = points.reshape(-1)
    with open(path, 'wb') as bin:
        for data in points:
            bin.write(
                struct.pack('f', float(data))
            )


def get_RT_matrix(odom_msg):
    quaternion = np.asarray(
        [odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
         odom_msg.pose.pose.orientation.z])
    translation = np.asarray(
        [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z])

    rotation = transforms3d.quaternions.quat2mat(quaternion)

    # print(T_qua2rota)
    RT_matrix = np.eye(4)
    RT_matrix[:3, :3] = rotation
    RT_matrix[:3, 3] = translation.T

    # RT_matrix = np.matmul(R_axes, RT_matrix)

    return RT_matrix


def calc_pitch_distribution(point_cloud_origin):
    # calculate field of view
    x_coords = point_cloud_origin[:, 0]
    y_coords = point_cloud_origin[:, 1]
    z_coords = point_cloud_origin[:, 2]
    distance_xy = LA.norm([x_coords, y_coords], axis=0)

    pitch_angles = np.rad2deg(np.arctan(z_coords / distance_xy))
    return pitch_angles


def calc_fov(eval_type, transform_odom_to_imu, point_cloud_imu):
    if eval_type == 'raw':
        # raw point cloud
        point_cloud_origin = point_cloud_imu
        pitch_angles = calc_pitch_distribution(point_cloud_origin)
        pitch_fov = pitch_angles.max() - pitch_angles.min()
    elif eval_type == 'alpha':
        # transform to local map coordinate
        point_cloud_odom = compose(transform_odom_to_imu, point_cloud_imu)
        # de-center, keep only FoV angle info
        point_cloud_origin = point_cloud_odom[:, :3] - transform_odom_to_imu[:3, 3]
        # merge front and back
        pitch_angles_front = calc_pitch_distribution(point_cloud_origin[np.where(point_cloud_origin[:, 0] >= 0)])
        pitch_angles_back = calc_pitch_distribution(point_cloud_origin[np.where(point_cloud_origin[:, 0] < 0)])

        pitch_fov = (pitch_angles_front.max() - pitch_angles_front.min()) + (pitch_angles_back.max() - pitch_angles_back.min())
    else:
        raise NotImplementedError
    return pitch_fov


def callback(odom_msg, point_cloud_msg, stat_msg):
    global frame_id, odom_list

    # save trajectory
    if odom_msg.pose.pose.position.x == 0 and odom_msg.pose.pose.position.y == 0:
        print('wrong pose')
        return

    transform_odom_to_imu_estimated = get_RT_matrix(odom_msg)
    odom_file = os.path.join(ROOT, '{}.odom'.format(odom_msg.header.stamp.to_sec()))
    np.savetxt(odom_file, np.matmul(T__world__o__odom, transform_odom_to_imu_estimated))

    if frame_id % 5 == 0:
        image_pub_raw.publish(title_image_raw_msg)
        image_pub_alpha.publish(title_image_alpha_msg)
        try:
            point_cloud_msg.fields = [point_cloud_msg.fields[0], point_cloud_msg.fields[1], point_cloud_msg.fields[2],
                             point_cloud_msg.fields[4], point_cloud_msg.fields[5], point_cloud_msg.fields[6],
                             point_cloud_msg.fields[3], point_cloud_msg.fields[7]]
            point_cloud_array = ros_numpy.numpify(point_cloud_msg)
            if len(point_cloud_array.shape) == 2:
                point_cloud = np.zeros((point_cloud_array.shape[0] * point_cloud_array.shape[1], 4))
            else:
                point_cloud = np.zeros((point_cloud_array.shape[0], 4))
            # parse lidar point array
            point_cloud[:, 0] = point_cloud_array['x'].reshape(-1)
            point_cloud[:, 1] = point_cloud_array['y'].reshape(-1)
            point_cloud[:, 2] = point_cloud_array['z'].reshape(-1)
            point_cloud = point_cloud[::2, :]

            # calculate FoV gain
            fov_raw = calc_fov('raw', transform_odom_to_imu_estimated, point_cloud)
            fov_alpha = calc_fov('alpha', transform_odom_to_imu_estimated, point_cloud)
            stat_file = os.path.join(ROOT, '{}.stats'.format(odom_msg.header.stamp.to_sec()))
            np.savetxt(stat_file, np.array([
                fov_raw, fov_alpha,
                stat_msg.pose.pose.position.x,
                stat_msg.pose.pose.position.y,
            ]), )
        except Exception as e:
            print(e)

    print("saving {:03}....".format(frame_id))
    frame_id += 1


def create_title_image():
    img_raw = np.zeros((70, 240, 3), dtype=np.uint8)
    img_alpha = np.zeros((40, 240, 3), dtype=np.uint8)

    font_weight = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_raw, 'Raw LiDAR sensor output ->', (5, 20),
                font, 0.5, (255, 0, 0), font_weight, cv2.LINE_AA)
    cv2.putText(img_raw, '(w/o alphaLiDAR)', (55, 45),
                font, 0.4, (255, 0, 0), font_weight, cv2.LINE_AA)
    cv2.putText(img_alpha, 'alphaLiDAR result ->', (5, 25),
                font, 0.68, (0, 255, 100), font_weight, cv2.LINE_AA)

    return img_raw, img_alpha


def numpy_to_ros_image(numpy_image, encoding):
    # cvBridge may not work with py3
    height, width, channels = numpy_image.shape

    ros_image = Image()
    ros_image.header.stamp = rospy.Time.now()
    ros_image.height = height
    ros_image.width = width
    ros_image.encoding = encoding
    ros_image.is_bigendian = 0
    ros_image.step = width * channels
    ros_image.data = numpy_image.tobytes()

    return ros_image


if __name__ == '__main__':
    rospy.init_node("alpha_lidar_statistics_saver", anonymous=True)
    rospy.loginfo("Start....")

    # if len(sys.argv) <= 1:
    #     rospy.logerr(
    #         'Useage: rosrun [pakage] save_pcd_odom_KITTI.py PATH TOPIC_POINTCLOUD TOPIC_ODOMETRY ENABLE_WORLD_TF')
    #     exit(1)
    # else:
    ROOT = '/tmp/alpha_lidar'
    if not os.path.exists(ROOT):
        os.mkdir(ROOT)
    else:
        os.system('rm -rf {}'.format(ROOT))
        os.mkdir(ROOT)

    title_image_raw, title_image_alpha = create_title_image()
    # title_image_raw_msg = bridge.cv2_to_imgmsg(title_image_raw, encoding="rgb8")
    # title_image_alpha_msg = bridge.cv2_to_imgmsg(title_image_alpha, encoding="rgb8")
    title_image_raw_msg = numpy_to_ros_image(title_image_raw, encoding="rgb8")
    title_image_alpha_msg = numpy_to_ros_image(title_image_alpha, encoding="rgb8")
    image_pub_raw = rospy.Publisher('/title_raw', Image, queue_size=10)
    image_pub_alpha = rospy.Publisher('/title_alpha', Image, queue_size=10)


    # subscribe odometry and pointclouds
    odom_sub = message_filters.Subscriber('/Odometry', Odometry)
    points_sub = message_filters.Subscriber('/cloud_registered_body', PointCloud2)
    stats_sub = message_filters.Subscriber('/stats', Odometry)
    ts = message_filters.ApproximateTimeSynchronizer([odom_sub, points_sub, stats_sub], 100, slop=0.0001,
                                                     allow_headerless=False)
    ts.registerCallback(callback)

    rospy.spin()
