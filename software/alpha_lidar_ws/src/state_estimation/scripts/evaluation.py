#!/usr/bin/env python
# coding=utf-8
import glob
import sys
from collections import defaultdict

import numpy as np
import os
import matplotlib.pyplot as plt
import transforms3d
from evo import entry_points
import argparse

# setup fonts
plt.rcParams["figure.autolayout"] = True
plt.rcParams['pdf.fonttype'] = 42

AX_LABEL_SIZE = 14
LEGEND_SIZE = 12
CBAR_SIZE = 12


def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re


def pose_list_to_TUM_ndarray(pose_list):
    tum_array = []
    # for ts, pose in pose_list[:len(pose_list) // 2]:
    for timestamp, pose in pose_list:
        pose_quat = transforms3d.quaternions.mat2quat(pose[:3, :3])
        tum_array.append([
            timestamp,
            pose[0, 3], pose[1, 3], pose[2, 3],
            pose_quat[1], pose_quat[2], pose_quat[3], pose_quat[0]
        ])

    return np.row_stack(tum_array)


def load_state_estimation_result(root):
    """
    load state estimation trajectory (odometry)
    @param root: path to the folder containing odometry files
    """
    trajectory = []
    path_list = glob.glob(os.path.join(root, '*.odom'))
    for file_name in sorted(path_list):
        trajectory.append([
            float(file_name.split('/')[-1][:-5]),
            np.loadtxt(file_name),
        ])
    trajectory = pose_list_to_TUM_ndarray(trajectory)
    return trajectory


def process_traj(ground_truth, trajectory):
    # Store trajectory in format required by evo
    ground_truth_path, trajectory_path = None, None
    if type(ground_truth) is str:
        ground_truth_path = ground_truth
    elif type(ground_truth) is np.ndarray:
        ground_truth_path = '/tmp/__gt.traj'
        np.savetxt(ground_truth_path, ground_truth, fmt='%.9f')
    else:
        raise NotImplementedError('Traj type not support')

    if type(trajectory) is str:
        trajectory_path = trajectory
    elif type(trajectory) is np.ndarray:
        trajectory_path = '/tmp/__est.traj'
        np.savetxt(trajectory_path, trajectory, fmt='%.9f')

    else:
        raise NotImplementedError('Traj type not support')

    return ground_truth_path, trajectory_path


def evo_ate(ground_truth, trajectory,
            pose_relation='trans_part', silent=False, plot=False, align=True, t_max_diff=None,
            n_to_align=None, additional_params=[]):
    ground_truth_path, trajectory_path = process_traj(ground_truth, trajectory)
    # clear parameters
    additional_params.clear()
    if silent:
        additional_params += ['--silent']
    if plot:
        additional_params += ['-p']
    if align:
        additional_params += ['--align']
    if n_to_align:
        additional_params += ['--n_to_align', str(n_to_align)]
    if t_max_diff:
        additional_params += ['--t_max_diff', str(t_max_diff)]
    sys.argv = [sys.argv[0], 'tum',
                ground_truth_path, trajectory_path,
                '-r', pose_relation,
                '--plot_mode', 'xy',
                ] + additional_params
    # print(sys.argv, additional_params, plot)
    return entry_points.ape()


def load_stat_sequence(root):
    """
    load fov sequence
    """
    timestamp = []
    fov_raw = []
    fov_alpha = []
    latency_opt = []
    latency_update = []
    path_list = glob.glob(os.path.join(root, '*.stats'))
    for file_name in sorted(path_list):
        timestamp.append(float(file_name.split('/')[-1][:-6]))
        stat = np.loadtxt(file_name)
        fov_raw.append(stat[0])
        fov_alpha.append(stat[1])
        latency_opt.append(stat[2])
        latency_update.append(stat[3])
    return timestamp, fov_raw, fov_alpha, np.array(latency_opt), np.array(latency_update)


def evaluate_segmented_local_ate(trajectory_estimated, trajectory_ground_truth, windows_length, window_step, ):
    """
    calculate mean ATE of local map segments, which is used to evaluate the quality of integrated local map
    """
    ATE_stats = defaultdict(list)
    print('-' * 40)
    print(
        f'ATE (Absolute Trajectory Error) of each map segment:')
    for window_idx, window_frontier in enumerate(range(0, len(trajectory_estimated), window_step)):
        if window_frontier + windows_length >= len(trajectory_estimated):
            break
        # crop segments
        trajectory_eval = trajectory_estimated[window_frontier:min(window_frontier + windows_length, len(trajectory_estimated) - 1)]
        try:
            ape = evo_ate(trajectory_ground_truth, trajectory_eval, align=True, silent=True, t_max_diff=0.1)
        except Exception as e:
            print('No associated timestamp')
            continue
        print(f'| Map segment: {window_idx + 1:4d}', end='')
        for key, value in ape.stats.items():
            ATE_stats[key].append(value)
            print(f'| {key}: {value:.5f} m', end='')
        print()
    print('\n' + '-' * 40)
    print(
        f'Average ATE (Absolute Trajectory Error) of all map segments: {np.mean(ATE_stats["mean"]):.5f} m \n'
        f'(Under the settings of: map segment length = {windows_length} frames | map segment window step = {window_step} frames)')


def evaluate_stat(root):
    timestamp, fov_raw_deg, fov_alpha_deg, latency_opt, latency_update \
        = load_stat_sequence(root)
    timestamp = (np.array(timestamp) - timestamp[0])

    plt.gcf().set_size_inches(6, 6)
    plt.subplot(2, 1, 1)
    plt.title('Latency of state estimation', fontsize=20)
    latency_raw = latency_opt + latency_update
    # smooth latency for visualization
    latency = moving_average(latency_raw, 10)

    plt.plot(timestamp, latency, label='Latency')
    plt.hlines(latency.mean(), xmin=0, xmax=timestamp[-1], label='Average latency',
               color='tab:red', linestyles='--', linewidth=2, zorder=5, alpha=0.8)

    plt.xlabel('Running time (s)', fontsize=AX_LABEL_SIZE, labelpad=0)
    plt.ylabel('Latency (s)', fontsize=AX_LABEL_SIZE)
    plt.tick_params(labelsize=CBAR_SIZE)
    plt.grid(True, axis='y', linestyle='--', which='major', color='grey', alpha=.25)
    plt.legend(fontsize=LEGEND_SIZE, ncol=3, loc='upper center')

    plt.subplot(2, 1, 2)
    plt.title('FoV (Field of View) comparison', fontsize=20)
    # smooth for visualization
    fov_alpha = moving_average(fov_alpha_deg, 5)
    fov_raw = moving_average(fov_raw_deg, 5)

    plt.plot(timestamp, fov_alpha, label='$\\alpha$LiDAR', color='tab:red')
    plt.plot(timestamp, fov_raw, label='Raw LiDAR', color='tab:blue')

    plt.hlines(fov_alpha.mean(), xmin=0, xmax=timestamp[-1], color='tab:red', linestyles='--',
               linewidth=2, zorder=5, alpha=0.8)
    plt.hlines(fov_raw.mean(), xmin=0, xmax=timestamp[-1], color='tab:blue', linestyles='--',
               linewidth=2, zorder=5, alpha=0.8)

    plt.xlabel('Running time (s)', fontsize=AX_LABEL_SIZE, labelpad=0)
    plt.ylabel('Vertical FoV ($\circ$)', fontsize=AX_LABEL_SIZE)
    plt.tick_params(labelsize=CBAR_SIZE)
    plt.yticks([0, 90, 180, 270, 360])
    plt.ylim(-10, 490)
    plt.grid(True, axis='y', linestyle='--', which='major', color='grey', alpha=.25)
    plt.legend(fontsize=LEGEND_SIZE, ncol=3, loc='upper center')

    plt.subplots_adjust(wspace=0, hspace=0, bottom=0.6)

    plt.tight_layout()

    print('\n' + '-' * 40)
    print(
        f'Optimization latency:\t | mean: {np.mean(latency_opt):.4f} s | min: {np.min(latency_opt):.4f} s | max: {np.max(latency_opt):.4f} s')
    print(
        f'Map update latency:\t | mean: {np.mean(latency_update):.4f} s | min: {np.min(latency_update):.4f} s | max: {np.max(latency_update):.4f} s')
    print(
        f'Total latency:\t\t | mean: {np.mean(latency_raw):.4f} s | min: {np.min(latency_raw):.4f} s | max: {np.max(latency_raw):.4f} s')

    print('\n' + '-' * 40)
    print(
        f'Average FoV (Field of View):  | alphaLiDAR: {np.mean(fov_alpha_deg):.2f} degree | Raw LiDAR: {np.mean(fov_raw_deg):.2f} degree')

    print('\n' + '-' * 40)

    plt.savefig('/tmp/alpha_lidar_stats.png', bbox_inches='tight', dpi=300)
    print(f'Result saved to /tmp/alpha_lidar_stats.png.')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation code')
    parser.add_argument('--gt_path', type=str, help='Path to the ground truth trajectory file (TUM format).')
    args = parser.parse_args()

    evaluation_tmp_root = '/tmp/alpha_lidar'
    if not os.path.exists(evaluation_tmp_root):
        print('State estimation result files do not exist, please run the state estimation code first.')
        exit(1)

    if args.gt_path is not None:
        trajectory_estimated = load_state_estimation_result(root=evaluation_tmp_root)
        trajectory_ground_truth = np.loadtxt(args.gt_path)
        evaluate_segmented_local_ate(trajectory_estimated, trajectory_ground_truth, windows_length=200, window_step=100)

    evaluate_stat(evaluation_tmp_root)
