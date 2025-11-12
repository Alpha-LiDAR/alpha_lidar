// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/random_sample.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include "voxel_map_util.hpp"

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   time_sync_en = false, extrinsic_est_en = true, path_en = true, encoder_fusion_en=true;
double lidar_time_offset = 0.0;
double encoder_zeropoint_offset_deg = 0.0;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double intensity_th = 1.0f;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string  lid_topic, imu_topic, encoder_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0;
double total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0, lidar_end_time_prev=0;
int    effective_feature_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iteration_count = 0, features_downsampled_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_index = 0;
int scan_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
double publish_limit_z = 1000000.0;
int publish_dense_skip = 1;
int publish_downsample_points = 1000000;
int publish_path_skip = 1;
bool adaptive_voxelization = false;
std::vector<double> adaptive_threshold;
std::vector<double> adaptive_multiple_factor;
bool init_gravity_with_pose;

vector<vector<int>>  pointSearchInd_surf;
vector<PointVector>  Nearest_Points;
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
vector<double>       extrinT_encoder(3, 0.0);
vector<double>       extrinR_encoder(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
deque<nav_msgs::Odometry::ConstPtr> encoder_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr features_undistorted(new PointCloudXYZI());
PointCloudXYZI::Ptr features_downsampled_body(new PointCloudXYZI());
PointCloudXYZI::Ptr features_downsampled_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normal_vectors(new PointCloudXYZI(100000, 1));
//PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
//PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;
std::vector<M3D> var_down_body;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterAdaptive;
pcl::RandomSample<PointType> downSizeFilterVis;

std::vector<float> nn_dist_in_feats;
std::vector<float> nn_plane_std;
PointCloudXYZI::Ptr feats_with_correspondence(new PointCloudXYZI());

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

SO3 R_lidar_o_encoder(Eye3d);
V3D t_lidar_o_encoder(Zero3d);
SO3 R_encoder_o_zero_point(Eye3d);

// params for voxel mapping algorithm
double min_eigen_value = 0.003;
int max_layer = 0;

int max_cov_points_size = 50;
int max_points_size = 50;
double sigma_num = 2.0;
double max_voxel_size = 1.0;
std::vector<int> layer_size;

double ranging_cov = 0.0;
double angle_cov = 0.0;
std::vector<double> layer_point_size;

bool publish_voxel_map = false;
int publish_max_voxel_layer = 0;

std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point, state_point_prev;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

const bool var_contrast(pointWithCov &x, pointWithCov &y) {
    return (x.cov.diagonal().norm() < y.cov.diagonal().norm());
};

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
//    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;

    po->curvature = pi->curvature;
    po->normal_x = pi->normal_x;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    auto time_offset = lidar_time_offset;
//    std::printf("lidar offset:%f\n", lidar_time_offset);
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() + time_offset < last_timestamp_lidar)
    {
//        ROS_ERROR("lidar loop back, clear buffer");
        ROS_ERROR("lidar loop back, skip this scan!!!");
//        lidar_buffer.clear();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    // 删除过少的点
    // std::printf("points: %ld\n", ptr->size());
    if (ptr->size() < 120)
    {
//        ROS_ERROR("lidar loop back, clear buffer");
        ROS_ERROR("Too few points, skip this scan!!!");
//        lidar_buffer.clear();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }

    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec() + time_offset);
    last_timestamp_lidar = msg->header.stamp.toSec() + time_offset;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
//        ROS_ERROR("lidar loop back, clear buffer");
        ROS_ERROR("lidar loop back, skip this scan!!!");
//        lidar_buffer.clear();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    if (timestamp < last_timestamp_imu)
    {
//        ROS_WARN("imu loop back, clear buffer");
//        imu_buffer.clear();
        ROS_WARN("imu loop back, ignoring!!!");
        ROS_WARN("current T: %f, last T: %f", timestamp, last_timestamp_imu);
        return;
    }
    // 剔除异常数据
    if (std::abs(msg->angular_velocity.x) > 10
        || std::abs(msg->angular_velocity.y) > 10
        || std::abs(msg->angular_velocity.z) > 10) {
        ROS_WARN("Large IMU measurement!!! Drop Data!!! %.3f  %.3f  %.3f",
                 msg->angular_velocity.x,
                 msg->angular_velocity.y,
                 msg->angular_velocity.z
        );
        return;
    }

//    // 如果是第一帧 拿过来做重力对齐
//    // TODO 用多帧平均的重力
//    if (is_first_imu) {
//        double acc_vec[3] = {msg_in->linear_acceleration.x, msg_in->linear_acceleration.y, msg_in->linear_acceleration.z};
//
//        R__world__o__initial = SO3(g2R(Eigen::Vector3d(acc_vec)));
//
//        is_first_imu = false;
//    }

    last_timestamp_imu = timestamp;

    mtx_buffer.lock();

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void encoder_cbk(const nav_msgs::Odometry::ConstPtr &msg_in)
{
    encoder_buffer.push_back(msg_in);
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {

//            std::printf("\nFirst 100 points: \n");
//            for(int i=0; i < 100; ++i){
//                std::printf("%f ", meas.lidar->points[i].curvature  / double(1000));
//            }
//
//            std::printf("\n Last 100 points: \n");
//            for(int i=100; i >0; --i){
//                std::printf("%f ", meas.lidar->points[meas.lidar->size() - i - 1].curvature / double(1000));
//            }
//            std::printf("last point offset time: %f\n", meas.lidar->points.back().curvature / double(1000));
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
//            lidar_end_time = meas.lidar_beg_time + (meas.lidar->points[meas.lidar->points.size() - 20]).curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
//            std::printf("pcl_bag_time: %f\n", meas.lidar_beg_time);
//            std::printf("lidar_end_time: %f\n", lidar_end_time);
        }

        meas.lidar_end_time = lidar_end_time;
//        std::printf("Scan start timestamp: %f, Scan end time: %f\n", meas.lidar_beg_time, meas.lidar_end_time);

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());


void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        if (scan_index <= 1){
            return;
        }
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? features_undistorted : feats_down_body);
        // 随机下采样 只用于可视化
        if(!dense_pub_en){
            downSizeFilterVis.setSample(publish_downsample_points);
            downSizeFilterVis.setSeed (std::rand());
            downSizeFilterVis.setInputCloud(feats_down_body);
            downSizeFilterVis.filter(*laserCloudFullRes);
        }
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI laserCloudWorld;
        for (int i = 0; i < size; i++)
        {
            if (i % publish_dense_skip !=0) {
                continue;
            }
            PointType const * const p = &laserCloudFullRes->points[i];
            if(p->intensity < intensity_th){
                continue;
            }
            // 转换到gravity aligned系 删除过高的点, 保证平面的可视化效果
            V3D p_body;
            p_body << p->x, p->y, p->z;
            V3D p_gravaity = p_imu->Initial_R_wrt_G * state_point.rot * state_point.offset_R_L_I * p_body;
            if (p_gravaity.z() > publish_limit_z) {
                continue;
            }

//            if (p->x < 0 and p->x > -4
//                    and p->y < 1.5 and p->y > -1.5
//                            and p->z < 2 and p->z > -1) {
//                continue;
//            }
            PointType p_world;

            RGBpointBodyToWorld(p, &p_world);
//            if (p_world.z > 1) {
//                continue;
//            }
            laserCloudWorld.push_back(p_world);
//            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
//                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
//    int size = features_undistorted->points.size();
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? features_undistorted : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&laserCloudFullRes->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;

}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );

    static tf::TransformBroadcaster br_world;
    transform.setOrigin(tf::Vector3(0, 0, 0));
    q.setValue(p_imu->Initial_R_wrt_G.x(), p_imu->Initial_R_wrt_G.y(), p_imu->Initial_R_wrt_G.z(), p_imu->Initial_R_wrt_G.w());
    transform.setRotation(q);
    br_world.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "world", "camera_init"));
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % publish_path_skip == 0)
    {
        path.header.stamp = msg_body_pose.header.stamp;
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void transformLidar(const state_ikfom &state_point, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
    trans_cloud->clear();
    for (size_t i = 0; i < input_cloud->size(); i++) {
        pcl::PointXYZINormal p_c = input_cloud->points[i];
        Eigen::Vector3d p_lidar(p_c.x, p_c.y, p_c.z);
        // HACK we need to specify p_body as a V3D type!!!
        V3D p_body = state_point.rot * (state_point.offset_R_L_I * p_lidar + state_point.offset_T_L_I) + state_point.pos;
        PointType pi;
        pi.x = p_body(0);
        pi.y = p_body(1);
        pi.z = p_body(2);
        pi.intensity = p_c.intensity;
        trans_cloud->points.push_back(pi);
    }
}

//M3D transformLiDARCovToWorld(Eigen::Vector3d &p_lidar, const esekfom::esekf<state_ikfom, 12, input_ikfom>& kf, const Eigen::Matrix3d& COV_lidar)
//{
//    double match_start = omp_get_wtime();
//    // FIXME 这里首先假定LiDAR系和body是重叠的 没有外参
//    M3D point_crossmat;
//    point_crossmat << SKEW_SYM_MATRX(p_lidar);
//    // 注意这里Rt的cov顺序
//    M3D rot_var = kf.get_P().block<3, 3>(3, 3);
//    M3D t_var = kf.get_P().block<3, 3>(0, 0);
//    auto state = kf.get_x();
//
//    // Eq. (3)
//    M3D COV_world =
//            state.rot * COV_lidar * state.rot.conjugate()
//            + state.rot * (-point_crossmat) * rot_var * (-point_crossmat).transpose()  * state.rot.conjugate()
//            + t_var;
//    return COV_world;
//    // Voxel map 真实实现
////    M3D cov_world = R_body * COV_lidar * R_body.conjugate() +
////          (-point_crossmat) * rot_var * (-point_crossmat).transpose() + t_var;
//
//}

M3D transformLiDARCovToWorld(Eigen::Vector3d &p_lidar, const esekfom::esekf<state_ikfom, 12, input_ikfom>& kf, const Eigen::Matrix3d& COV_lidar)
{
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(p_lidar);
    auto state = kf.get_x();

    // lidar到body的方差传播
    // 注意外参的var是先rot 后pos
    M3D il_rot_var = kf.get_P().block<3, 3>(6, 6);
    M3D il_t_var = kf.get_P().block<3, 3>(9, 9);

    M3D COV_body =
            state.offset_R_L_I * COV_lidar * state.offset_R_L_I.conjugate()
            + state.offset_R_L_I * (-point_crossmat) * il_rot_var * (-point_crossmat).transpose() * state.offset_R_L_I.conjugate()
            + il_t_var;

    // body的坐标
    V3D p_body = state.offset_R_L_I * p_lidar + state.offset_T_L_I;

    // body到world的方差传播
    // 注意pose的var是先pos 后rot
    point_crossmat << SKEW_SYM_MATRX(p_body);
    M3D rot_var = kf.get_P().block<3, 3>(3, 3);
    M3D t_var = kf.get_P().block<3, 3>(0, 0);

    // Eq. (3)
    M3D COV_world =
        state.rot * COV_body * state.rot.conjugate()
        + state.rot * (-point_crossmat) * rot_var * (-point_crossmat).transpose()  * state.rot.conjugate()
        + t_var;

    return COV_world;
    // Voxel map 真实实现
//    M3D cov_world = R_body * COV_lidar * R_body.conjugate() +
//          (-point_crossmat) * rot_var * (-point_crossmat).transpose() + t_var;

}

void observation_model_share(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
//    laserCloudOri->clear();
//    corr_normvect->clear();
    feats_with_correspondence->clear();
    total_residual = 0.0;

    // =================================================================================================================
    // 用当前迭代轮最新的位姿估计值 将点云转换到world地图系
    vector<pointWithCov> pv_list;
    PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
    // FIXME stupid mistake 这里应该用迭代的最新线性化点
    // FIXME stupid mistake 这里应该用迭代的最新线性化点
//    transformLidar(state_point, feats_down_body, world_lidar);
    transformLidar(s, feats_down_body, world_lidar);
    pv_list.resize(feats_down_body->size());
    for (size_t i = 0; i < feats_down_body->size(); i++) {
        // 保存body系和world系坐标
        pointWithCov pv;
        pv.point << feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z;
        pv.point_world << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
        // 计算lidar点的cov
        // 注意这个在每次迭代时是存在重复计算的 因为lidar系的点云covariance是不变的
        // M3D cov_lidar = calcBodyCov(pv.point, ranging_cov, angle_cov);
        M3D cov_lidar = var_down_body[i];
        // 将body系的var转换到world系
        M3D cov_world = transformLiDARCovToWorld(pv.point, kf, cov_lidar);
        pv.cov = cov_world;
        pv.cov_lidar = cov_lidar;
        pv_list[i] = pv;
    }

    // ===============================================================================================================
    // 查找最近点 并构建residual
    double match_start = omp_get_wtime();
    std::vector<ptpl, Eigen::aligned_allocator<ptpl>> ptpl_list;
    std::vector<V3D, Eigen::aligned_allocator<V3D>> non_match_list;

    current_state_point = s;
    BuildResidualListOMP(voxel_map, max_voxel_size, 3.0, max_layer, pv_list,
                         ptpl_list, non_match_list);
    double match_end = omp_get_wtime();
    // std::printf("Match Time: %f\n", match_end - match_start);

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    // 根据匹配结果 设置H和R的维度
    // h_x是观测值对状态量的导数 TODO 为什么不加上状态量对状态量误差的导数？？？？像quaternion那本书？
    effct_feat_num = ptpl_list.size();
    if (effct_feat_num < 1){
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23 因为点面距离只和位姿 外参有关 对其他状态量的导数都是0
    ekfom_data.h.resize(effct_feat_num);
    ekfom_data.R.resize(effct_feat_num, 1); // 把R作为向量 用的时候转换成diag
//    ekfom_data.R.setZero();
//    printf("isDiag: %d  R norm: %f\n", ekfom_data.R.isDiagonal(1e-10), ekfom_data.R.norm());

//    // 求每个匹配点到平面的距离
//    for (int i = 0; i < ptpl_list.size(); i++) {
//        // 取出匹配到的world系norm
//        PointType pl;
//        pl.x = ptpl_list[i].normal(0);
//        pl.y = ptpl_list[i].normal(1);
//        pl.z = ptpl_list[i].normal(2);
//
//        // 将原始点云转换到world系
//        V3D pi_world(s.rot * (s.offset_R_L_I * ptpl_list[i].point + s.offset_T_L_I) + s.pos);
//
//        // 求点面距离
//        float dis = pi_world.x() * pl.x + pi_world.y() * pl.y + pi_world.z() * pl.z + ptpl_list[i].d;
//        pl.intensity = dis;
////        std::printf("%.5f   %.5f\n", dis, ptpl_list[i].pd2);
////        std::printf("%.5f  %.5f\n", pi_world.x(), ptpl_list[i].point_world.x());
////        std::printf("%.5f  %.5f\n", pi_world.y(), ptpl_list[i].point_world.y());
//
//        PointType pi_body;
//        pi_body.x = ptpl_list[i].point(0);
//        pi_body.y = ptpl_list[i].point(1);
//        pi_body.z = ptpl_list[i].point(2);
//        laserCloudOri->push_back(pi_body);
//        corr_normvect->push_back(pl);
//        // for visualization
//        feats_with_correspondence->push_back(pi_body);
//
//        total_residual += fabs(dis);
//    }
//    assert(laserCloudOri->size() == effct_feat_num && corr_normvect->size() == effct_feat_num);
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < effct_feat_num; i++)
    {

//        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(ptpl_list[i].point);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
//        const PointType &norm_p = corr_normvect->points[i];
//        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);
        V3D norm_vec(ptpl_list[i].normal);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            // ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            // ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            ekfom_data.h_x.block<1, 12>(i,0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
//        ekfom_data.h(i) = -norm_p.intensity;
        float pd2 = norm_vec.x() * ptpl_list[i].point_world.x()
                + norm_vec.y() * ptpl_list[i].point_world.y()
                + norm_vec.z() * ptpl_list[i].point_world.z()
                + ptpl_list[i].d;
        ekfom_data.h(i) = -pd2;

        /*** Covariance ***/
//        // norm_p中存了匹配的平面法向 还有点面距离
//        V3D point_world = s.rot * (s.offset_R_L_I * ptpl_list[i].point + s.offset_T_L_I) + s.pos;
//        // /*** get the normal vector of closest surface/corner ***/
//        Eigen::Matrix<double, 1, 6> J_nq;
//        J_nq.block<1, 3>(0, 0) = point_world - ptpl_list[i].center;
//        J_nq.block<1, 3>(0, 3) = -ptpl_list[i].normal;
//        double sigma_l = J_nq * ptpl_list[i].plane_cov * J_nq.transpose();
//
//        M3D cov_lidar = calcBodyCov(ptpl_list[i].point, ranging_cov, angle_cov);
//        M3D R_cov_Rt = s.rot * cov_lidar * s.rot.conjugate();
//        // HACK 1. 因为是标量 所以求逆直接用1除
//        // HACK 2. 不同分量的方差用加法来合成 因为公式(12)中的Sigma是对角阵，逐元素运算之后就是对角线上的项目相加
//        double R_inv = 1.0 / (sigma_l + norm_vec.transpose() * R_cov_Rt * norm_vec);

        // norm_p中存了匹配的平面法向 还有点面距离
        // V3D point_world = s.rot * (s.offset_R_L_I * ptpl_list[i].point + s.offset_T_L_I) + s.pos;
        V3D point_world = ptpl_list[i].point_world;
        // /*** get the normal vector of closest surface/corner ***/
        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = point_world - ptpl_list[i].center;
        J_nq.block<1, 3>(0, 3) = -ptpl_list[i].normal;
        double sigma_l = J_nq * ptpl_list[i].plane_cov * J_nq.transpose();

        // M3D cov_lidar = calcBodyCov(ptpl_list[i].point, ranging_cov, angle_cov);
        M3D cov_lidar = ptpl_list[i].cov_lidar;
        M3D R_cov_Rt = s.rot * s.offset_R_L_I * cov_lidar * s.offset_R_L_I.conjugate() * s.rot.conjugate();
        // HACK 1. 因为是标量 所以求逆直接用1除
        // HACK 2. 不同分量的方差用加法来合成 因为公式(12)中的Sigma是对角阵，逐元素运算之后就是对角线上的项目相加
        double R_inv = 1.0 / (sigma_l + norm_vec.transpose() * R_cov_Rt * norm_vec);

        // 计算测量方差R并赋值 目前暂时使用固定值
        // ekfom_data.R(i) = 1.0 / LASER_POINT_COV;
        ekfom_data.R(i) = R_inv;
    }

    // std::printf("Effective Points: %d\n", effct_feat_num);
    res_mean_last = total_residual / effct_feat_num;
    // std::printf("res_mean: %f\n", res_mean_last);
    // std::printf("ef_num: %d\n", effct_feat_num);
}

nav_msgs::Odometry uncoupled_encoder_fusion(double time_odom, const state_ikfom &state)
{
    nav_msgs::Odometry result_odom;
    result_odom.header.frame_id = "camera_init";
    result_odom.child_frame_id = "encoder_body";
    result_odom.header.stamp = ros::Time().fromSec(time_odom);

    if (time_odom <= 0) {
        ROS_WARN("First frame, return.");
        result_odom.child_frame_id = "-1";
        return result_odom;
    }
    // 如果没有encoder数据或者odom数据太早 直接返回
    if (encoder_buffer.empty() || time_odom < encoder_buffer.front()->header.stamp.toSec()) {
        // ROS_WARN("No encoder data or odometry data too early!");
        result_odom.child_frame_id = "-1";
        return result_odom;
    }
    // 如果encoder数据太晚，需要等待或者直接drop
    if (encoder_buffer.back()->header.stamp.toSec() < time_odom) {
        ROS_WARN("Odometry data too late, need wait for encoder! %f, %f",
                 encoder_buffer.back()->header.stamp.toSec(), time_odom);
        result_odom.child_frame_id = "-2";
        return result_odom;
    }
    bool found = false;
    // 找到距离time最近的前后两帧encoder数据
    for (auto it = encoder_buffer.begin(); it != encoder_buffer.end() - 1; it++) {
        double time_encoder_head = (*it)->header.stamp.toSec();
        double time_encoder_tail = (*(it + 1))->header.stamp.toSec();
        if (time_odom <= time_encoder_tail) {
            // find encoder poses previous and next to current odometry
            found = true;
            SO3 encoder_head = SO3(
                    (*it)->pose.pose.orientation.w,
                    (*it)->pose.pose.orientation.x,
                    (*it)->pose.pose.orientation.y,
                    (*it)->pose.pose.orientation.z);
            SO3 encoder_tail = SO3(
                    (*(it + 1))->pose.pose.orientation.w,
                    (*(it + 1))->pose.pose.orientation.x,
                    (*(it + 1))->pose.pose.orientation.y,
                    (*(it + 1))->pose.pose.orientation.z);

            // slerp to align the encoder measurements with odometry
            SO3 R_base_o_encoder = encoder_head.slerp(
                    (time_odom - time_encoder_head) / (time_encoder_tail - time_encoder_head), encoder_tail);
            SO3 R_encoder_o_base = (R_base_o_encoder * R_encoder_o_zero_point).inverse();

            // lidar w.r.t odom(camera_init) pose
            SO3 R_odom_o_lidar = state.rot * state.offset_R_L_I;
            V3D t_odom_o_lidar = state.rot * state.offset_T_L_I + state.pos;

            // R_odom_o_lidar * (R_lidar_o_encoder * R_encoder_o_base + t_lidar_o_encoder) + t_odom_o_lidar
            SO3 R_odom_o_base = R_odom_o_lidar * R_lidar_o_encoder * R_encoder_o_base;
            V3D t_odom_o_base = R_odom_o_lidar * t_lidar_o_encoder + t_odom_o_lidar;

            /*
            // debug information
            auto euler_head = encoder_head.toRotationMatrix().eulerAngles(0, 1, 2);
            auto euler_tail = encoder_tail.toRotationMatrix().eulerAngles(0, 1, 2);
            auto euler_slerp = R_base_o_encoder.toRotationMatrix().eulerAngles(0, 1, 2);

            std::printf("\n\n\n");
            std::printf("head: %.7f -> %.3f %.3f %.3f\n",
                        time_encoder_head, euler_head[0], euler_head[1], euler_head[2]);
            std::printf("tail: %.7f -> %.3f %.3f %.3f\n",
                        time_encoder_tail, euler_tail[0], euler_tail[1], euler_tail[2]);
            std::printf("odom: %.7f -> %.3f %.3f %.3f\n",
                        time_odom, euler_slerp[0], euler_slerp[1], euler_slerp[2]);
            std::printf("Q front: %.7f, Q back: %.3f, Q size: %ld\n",
                        encoder_buffer.front()->header.stamp.toSec(),
                        encoder_buffer.back()->header.stamp.toSec(),
                        encoder_buffer.size());
            */

            result_odom.pose.pose.orientation.w = R_odom_o_base.w();
            result_odom.pose.pose.orientation.x = R_odom_o_base.x();
            result_odom.pose.pose.orientation.y = R_odom_o_base.y();
            result_odom.pose.pose.orientation.z = R_odom_o_base.z();
            result_odom.pose.pose.position.x = t_odom_o_base.x();
            result_odom.pose.pose.position.y = t_odom_o_base.y();
            result_odom.pose.pose.position.z = t_odom_o_base.z();
            break;
        }
    }
    // clear up encoder queue
    // XXX may need lock？
    double time_encoder = encoder_buffer.front()->header.stamp.toSec();
    while (time_encoder < time_odom - 0.1 && !encoder_buffer.empty()) {
        encoder_buffer.pop_front();
        time_encoder = encoder_buffer.front()->header.stamp.toSec();
    }
    // std::printf("After Q clean up, size: %ld\n", encoder_buffer.size());

    return result_odom;
}


/*** ROS subscribe initialization ***/
ros::Subscriber sub_pcl;
ros::Subscriber sub_imu;
ros::Subscriber sub_encoder;
ros::Publisher pubLaserCloudFull;
ros::Publisher pubLaserCloudFull_body;
ros::Publisher pubLaserCloudEffect;
ros::Publisher pubLaserCloudMap;
ros::Publisher pubOdomAftMapped;
ros::Publisher pubOdomEncoderBody;
ros::Publisher pubExtrinsic;
ros::Publisher pubPath;
ros::Publisher voxel_map_pub;
ros::Publisher stats_pub;

// for Plane Map
bool init_map = false;


// statistic
double sum_optimize_time = 0, sum_update_time = 0;
std::fstream stat_latency("/tmp/latency.csv", std::ios::out);

void execute(){
    // execute one step of state estimation and mapping
    if (flg_first_scan)
    {
        first_lidar_time = Measures.lidar_beg_time;
        p_imu->first_lidar_time = first_lidar_time;
        flg_first_scan = false;
        // continue;
        return;
    }

    double t_optimize_start = omp_get_wtime();
    p_imu->Process(Measures, kf, features_undistorted);
    state_point = kf.get_x();
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

  if (features_undistorted->empty() || (features_undistorted == NULL)) {
    ROS_WARN("No point, skip this scan!\n");
    // continue;
    return;
  }

    flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
    // ===============================================================================================================
    // 第一帧 如果ekf初始化了 就初始化voxel地图
    if (flg_EKF_inited && !init_map) {
        PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
        transformLidar(state_point, features_undistorted, world_lidar);
        std::vector<pointWithCov> pv_list;

        // std::cout << kf.get_P() << std::endl;
        // 计算第一帧所有点的covariance 并用于构建初始地图
        for (size_t i = 0; i < world_lidar->size(); i++) {
            pointWithCov pv;
            pv.point << world_lidar->points[i].x, world_lidar->points[i].y,
                    world_lidar->points[i].z;
            V3D point_this(features_undistorted->points[i].x,
                           features_undistorted->points[i].y,
                           features_undistorted->points[i].z);
            // if z=0, error will occur in calcBodyCov. To be solved
            if (point_this[2] == 0) {
                point_this[2] = 0.001;
            }
            M3D cov_lidar = calcBodyCov(point_this, ranging_cov, angle_cov);
            // 转换到world系
            M3D cov_world = transformLiDARCovToWorld(point_this, kf, cov_lidar);

            pv.cov = cov_world;
            pv_list.push_back(pv);
            Eigen::Vector3d sigma_pv = pv.cov.diagonal();
            sigma_pv[0] = sqrt(sigma_pv[0]);
            sigma_pv[1] = sqrt(sigma_pv[1]);
            sigma_pv[2] = sqrt(sigma_pv[2]);
        }

        // 当前state point 赋值
        current_state_point = kf.get_x();
        buildVoxelMap(pv_list, max_voxel_size, max_layer, layer_size,
                      max_points_size, max_points_size, min_eigen_value,
                      voxel_map);
        std::cout << "build voxel map" << std::endl;

        if (publish_voxel_map) {
            current_state_point = kf.get_x();
            pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
            publish_frame_world(pubLaserCloudFull);
            publish_frame_body(pubLaserCloudFull_body);
        }
        init_map = true;
        // continue;
        return;
    }

    /*** downsample the feature points in a scan ***/
    downSizeFilterSurf.setInputCloud(features_undistorted);
    downSizeFilterSurf.filter(*feats_down_body);

    // 如果首次下采样点数量还是太多(一般是大场景,不需要这么多点) 那么就adaptive 再次下采样
    if (adaptive_voxelization) {
        size_t feats_down_size_first = feats_down_body->points.size();
        // 倒序查找LUT确定下采样的粒度
        // XXX 需要保证LUT是升序的!
        int search_idx;
        bool is_found = false;
        if (feats_down_size_first < adaptive_threshold[0]) {
            // LUT的第一个值是特殊保护值,用于退化环境升采样
            // 如果点数太少了,少于保护值, 那么反倒进行升采样, 防止退化
            search_idx = 0;
            is_found = true;
        } else{
            for (search_idx = adaptive_threshold.size() - 1; search_idx > 0; search_idx--) {
                // std::printf("\n %ld %ld %f %f\n",
                //            search_idx, adaptive_threshold.size(), adaptive_threshold[search_idx], adaptive_multiple_factor[search_idx]);
                // 首次大于阈值, 就用对应的下采样粒度
                if (feats_down_size_first > adaptive_threshold[search_idx]) {
                    is_found = true;
                    break;
                }
            }
        }
        // 如果查找到LUT中的某个阈值, 就用对应的粒度下采样
        if (is_found) {
            float leaf_size_scaled = filter_size_surf_min * adaptive_multiple_factor[search_idx];
            downSizeFilterAdaptive.setLeafSize(leaf_size_scaled, leaf_size_scaled, leaf_size_scaled);
            if (leaf_size_scaled < filter_size_surf_min) {
                // 升采样 用原始点云
                downSizeFilterAdaptive.setInputCloud(features_undistorted);
            } else {
                downSizeFilterAdaptive.setInputCloud(feats_down_body);
            }
            downSizeFilterAdaptive.filter(*feats_down_body);
            std::printf("ADV: RAW: %10ld | First:  %10ld | Adap: %10ld, %5fpts, %5fm\n",
                        features_undistorted->size(),
                        feats_down_size_first,
                        feats_down_body->size(),
                        adaptive_threshold[search_idx],
                        leaf_size_scaled);
        }
    }

    sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);

    features_downsampled_size = feats_down_body->points.size();
    // 由于点云的body var是一直不变的 因此提前计算 在迭代时可以复用
    var_down_body.clear();
    for (auto & pt:feats_down_body->points) {
        V3D point_this(pt.x, pt.y, pt.z);
        var_down_body.push_back(calcBodyCov(point_this, ranging_cov, angle_cov));
    }

    /*** ICP and iterated Kalman filter update ***/
    if (features_downsampled_size < 5)
    {
        ROS_WARN("Too few points (<5 points), skip this scan!\n");
        // continue;
        return;
    }
    // ===============================================================================================================
    // 开始迭代滤波
    /*** iterated state estimation ***/
    double solve_H_time = 0;
    kf.update_iterated_dyn_share_diagonal();
//            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
    double t_optimize_end = omp_get_wtime();
    sum_optimize_time += t_optimize_end - t_optimize_start;

    state_point = kf.get_x();

//    // HACK 强行重置ba bg
//    state_point.ba.setZero();
//    state_point.bg.setZero();
//    kf.change_x(state_point);

    euler_cur = SO3ToEuler(state_point.rot);
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
    geoQuat.x = state_point.rot.coeffs()[0];
    geoQuat.y = state_point.rot.coeffs()[1];
    geoQuat.z = state_point.rot.coeffs()[2];
    geoQuat.w = state_point.rot.coeffs()[3];

    // ===============================================================================================================
    // 更新地图
    /*** add the points to the voxel map ***/
    double t_update_start = omp_get_wtime();
    // 用最新的状态估计将点及点的covariance转换到world系
    std::vector<pointWithCov> pv_list;
    PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
    transformLidar(state_point, feats_down_body, world_lidar);
    for (size_t i = 0; i < feats_down_body->size(); i++) {
        // 保存body系和world系坐标
        pointWithCov pv;
        pv.point << feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z;
        // 计算lidar点的cov
        // FIXME 这里错误的使用世界系的点来calcBodyCov时 反倒在某些seq（比如hilti2022的03 15）上效果更好 需要考虑是不是init_plane时使用更大的cov更好
        // 注意这个在每次迭代时是存在重复计算的 因为lidar系的点云covariance是不变的
        // M3D cov_lidar = calcBodyCov(pv.point, ranging_cov, angle_cov);
        M3D cov_lidar = var_down_body[i];
        // 将body系的var转换到world系
        M3D cov_world = transformLiDARCovToWorld(pv.point, kf, cov_lidar);

        // 最终updateVoxelMap需要用的是world系的point
        pv.cov = cov_world;
        pv.point << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
        pv_list.push_back(pv);
    }

    // 当前state point 赋值
    current_state_point = kf.get_x();
    std::sort(pv_list.begin(), pv_list.end(), var_contrast);
    updateVoxelMapOMP(pv_list, max_voxel_size, max_layer, layer_size,
                      max_points_size, max_points_size, min_eigen_value,
                      voxel_map);
    double t_update_end = omp_get_wtime();
    sum_update_time += t_update_end - t_update_start;
    // ===============================================================================================================
    // fuse encoder angles with odometry, transform odomtry to a fixed frame (w.r.t carrier)
    if (encoder_fusion_en){
        nav_msgs::Odometry odometry_encoder_body =
                uncoupled_encoder_fusion(lidar_end_time_prev, state_point_prev);
        // XXX: use previous frame to avoid the high probability of odometry fall outside of encoder queue
        // TODO: perform encoder fusion in seperated thread with fixed frequency
        state_point_prev = kf.get_x();
        lidar_end_time_prev = lidar_end_time;
        if (!(odometry_encoder_body.child_frame_id == "-1" || odometry_encoder_body.child_frame_id == "-2"))
        {
            pubOdomEncoderBody.publish(odometry_encoder_body);
        }
    }
    scan_index++;
    // ===============================================================================================================
    // 可视化相关
    /******* Publish odometry *******/
    double t_vis_start = omp_get_wtime();
    publish_odometry(pubOdomAftMapped);
//
//            /*** add the feature points to map kdtree ***/
//            map_incremental();
//
    // TODO skip first few frames
    // TODO downsample dense point clouds
    /******* Publish points *******/
    if (path_en)                         publish_path(pubPath);
    if (scan_pub_en)      publish_frame_world(pubLaserCloudFull);
    if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
    if (publish_voxel_map && pubLaserCloudMap.getNumSubscribers() > 0) {
//        current_state_point = kf.get_x();
        pubColoredVoxels(voxel_map, publish_max_voxel_layer, pubLaserCloudMap, lidar_end_time);
    }
    if (publish_voxel_map && voxel_map_pub.getNumSubscribers() > 0) {
//        current_state_point = kf.get_x();
        pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
    }
    double t_vis_end = omp_get_wtime();
    nav_msgs::Odometry stat_msg;
    stat_msg.header = odomAftMapped.header;
    stat_msg.pose.pose.position.x = t_optimize_end - t_optimize_start;
    stat_msg.pose.pose.position.y = t_update_end - t_update_start;
    stats_pub.publish(stat_msg);
    // publish_effect_world(pubLaserCloudEffect);
    // publish_map(pubLaserCloudMap);
//
    std::printf("v: %.2f %.2f %.2f BA: %.4f %.4f %.4f   BG: %.4f %.4f %.4f   g: %.4f %.4f %.4f\n",
                kf.get_x().vel.x(),kf.get_x().vel.y(),kf.get_x().vel.z(),
                kf.get_x().ba.x(),kf.get_x().ba.y(),kf.get_x().ba.z(),
                kf.get_x().bg.x(),kf.get_x().bg.y(),kf.get_x().bg.z(),
                kf.get_x().grav.get_vect().x(), kf.get_x().grav.get_vect().y(), kf.get_x().grav.get_vect().z()
    );

    std::printf("Mean Latency: %.3fs |  Mean Topt: %.5fs   Tu: %.5fs   | Cur Topt: %.5fs   Tu: %.5fs   Tvis: %.5fs\n",
                (sum_optimize_time + sum_update_time) / scan_index + (t_vis_end - t_vis_start),
                sum_optimize_time / scan_index, sum_update_time / scan_index,
                t_optimize_end - t_optimize_start,
                t_update_end - t_update_start,
                t_vis_end - t_vis_start);
    stat_latency
            << lidar_end_time << ", "
            << t_optimize_end - t_optimize_start << ", "
            << t_update_end - t_update_start << ", "
            << t_vis_end - t_vis_start << ", "
            << std::endl;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<double>("time_offset", lidar_time_offset, 0.0);

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<double>("publish/intensity_th", intensity_th, 1.0);

    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<string>("common/encoder_topic", encoder_topic,"/encoder");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);

    // mapping algorithm params
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<int>("mapping/max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<int>("mapping/max_points_size", max_points_size, 100);
    nh.param<int>("mapping/max_cov_points_size", max_cov_points_size, 100);
    nh.param<vector<double>>("mapping/layer_point_size", layer_point_size,vector<double>());
    nh.param<int>("mapping/max_layer", max_layer, 2);
    nh.param<double>("mapping/voxel_size", max_voxel_size, 1.0);
    nh.param<double>("mapping/down_sample_size", filter_size_surf_min, 0.5);
    std::cout << "filter_size_surf_min:" << filter_size_surf_min << std::endl;
    nh.param<double>("mapping/plannar_threshold", min_eigen_value, 0.01);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<bool>("mapping/encoder_fusion_en", encoder_fusion_en, false);
    nh.param<vector<double>>("mapping/extrinsic_T_encoder_lidar", extrinT_encoder, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R_encoder_lidar", extrinR_encoder, vector<double>());
    nh.param<double>("mapping/encoder_offset_deg", encoder_zeropoint_offset_deg, 0);
    nh.param<bool>("mapping/adaptive_voxelization", adaptive_voxelization, false);
    nh.param<vector<double>>("mapping/adaptive_threshold", adaptive_threshold, vector<double>({1000}));
    nh.param<vector<double>>("mapping/adaptive_multiple_factor", adaptive_multiple_factor, vector<double>({2.0}));
    nh.param<bool>("mapping/init_gravity_with_pose", init_gravity_with_pose, false);

    // noise model params
    nh.param<double>("noise_model/ranging_cov", ranging_cov, 0.02);
    nh.param<double>("noise_model/angle_cov", angle_cov, 0.05);
    nh.param<double>("noise_model/gyr_cov",gyr_cov,0.1);
    nh.param<double>("noise_model/acc_cov",acc_cov,0.1);
    nh.param<double>("noise_model/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("noise_model/b_acc_cov",b_acc_cov,0.0001);

    // visualization params
    nh.param<bool>("publish/pub_voxel_map", publish_voxel_map, false);
    nh.param<int>("publish/publish_max_voxel_layer", publish_max_voxel_layer, 0);
    nh.param<int>("publish/publish_downsample_points", publish_downsample_points, 1000000);
    nh.param<int>("publish/publish_dense_skip", publish_dense_skip, 1);
    nh.param<int>("publish/publish_path_skip", publish_path_skip, 1);
    nh.param<double>("publish/publish_limit_z", publish_limit_z, 1000000.0);

    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 1);
    nh.param<bool>("preprocess/feature_extract_enable", p_pre->feature_enabled, false);
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    for (int i = 0; i < layer_point_size.size(); i++) {
        layer_size.push_back(layer_point_size[i]);
    }

    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    scan_index = 0;

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    p_imu->set_init_gravity_with_pose(init_gravity_with_pose);

    // calc encoder w.r.t lidar extrinsic
    M3D MAT_R_encoder_o_lidar;
    V3D t_encoder_o_lidar;
    t_encoder_o_lidar<<VEC_FROM_ARRAY(extrinT_encoder);
    MAT_R_encoder_o_lidar<<MAT_FROM_ARRAY(extrinR_encoder);
    SO3 R_encoder_o_lidar = MAT_R_encoder_o_lidar;
    // inverse for convenient
    R_lidar_o_encoder = R_encoder_o_lidar.inverse();
    t_lidar_o_encoder = -(R_lidar_o_encoder * t_encoder_o_lidar);
    // encoder zero-point offset
    R_encoder_o_zero_point = Eigen::AngleAxisd(
            encoder_zeropoint_offset_deg / 180.0 * M_PI, V3D::UnitZ());

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, observation_model_share, NUM_MAX_ITERATIONS, epsi);

    /*** ROS subscribe initialization ***/
    sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    sub_encoder = nh.subscribe(encoder_topic, 200000, encoder_cbk);
    pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pubOdomEncoderBody = nh.advertise<nav_msgs::Odometry>("/Encoder", 100000);
    pubExtrinsic = nh.advertise<nav_msgs::Odometry>("/Extrinsic", 100000);
    pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);
    voxel_map_pub = nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
    stats_pub = nh.advertise<nav_msgs::Odometry>("/stats", 10000);

//------------------------------------------------------------------------------------------------------
    // statistic
    stat_latency.setf(ios::fixed);
    stat_latency.precision(10);//精度为输出小数点后5位

////------------------------------------------------------------------------------------------------------
//    // 用rosbag读取
//    signal(SIGINT, SigHandle);
//
//    std::string bag_file = "/tmp/2024-03-05-17-21-56.bag";
//    rosbag::Bag bag;
//    try {
//        bag.open(bag_file, rosbag::bagmode::Read);
//    } catch (const rosbag::BagException& e) {
//        ROS_ERROR("Could not open bag file: %s", e.what());
//        return -1;
//    }
//
//    std::vector<std::string> topics;
//    topics.push_back(std::string("/velodyne_points"));
//    topics.push_back(std::string("/imu/data"));
//
//    rosbag::View view(bag, rosbag::TopicQuery(topics));
//        foreach(rosbag::MessageInstance const m, view) {
//            // 中途退出
//            if (flg_exit) break;
//
//            sensor_msgs::PointCloud2ConstPtr lidar_msg = m.instantiate<sensor_msgs::PointCloud2>();
//            if (lidar_msg != NULL) {
//                standard_pcl_cbk(lidar_msg);
//            }
//
//            sensor_msgs::ImuConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
//            if (imu_msg != NULL) {
//                imu_cbk(imu_msg);
//            }
//
//            if (sync_packages(Measures)) {
//                // execute one step
//                execute();
//            }
//        }
//    bag.close();

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();
        if(sync_packages(Measures))
        {
            // execute one step
            execute();
        }

        status = ros::ok();
        rate.sleep();
    }
    stat_latency.close();
    return 0;
}
