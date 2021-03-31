import matplotlib.pyplot as plt
plt.style.use(['science', 'grid', 'bright'])  # bright, high-contrast, retro, high-vis, muted, vibrant
from matplotlib.figure import figaspect
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import math
import os
from trilateration import trilateration
from trilateration_types import Point, Measurement3d
from datetime import date

pd.set_option('mode.chained_assignment', 'warn')
# if you set a value on a copy, warning will show

# set metadata
plot_format = 'svg'
show_plots = False  # only set to true when simulating one single scenario!
save_plots = True
save_errors_csv = False
save_input_errors_csv = False  # if true, set sample sizes to 1, and both ground truths to true
# sample_sizes = [1, 2, 3, 4, 6, 8, 10]
# sample_sizes = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# sample_sizes = [1, 5, 10, 50, 100, 500]
# sample_sizes = [50, 100, 500]
# sample_sizes = [1, 5, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
sample_sizes = [100]
# sample_sizes = np.arange(1, 101)
# sample_sizes = np.arange(1,21)
use_position_ground_truth = False
use_distance_ground_truth = False
# experiments = [0, 1, 2, 4, 5] # 5.2.2021
# experiments = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16] # 10.2.2021
experiments = [0]  # Scenarios 1-6
# scenarios = [1, 2, 3, 4, 5, 6]
scenarios = [1, 6]

# get today's date for naming directories
date = str(date.today())

# set ground truth mode for naming files
if use_position_ground_truth and not use_distance_ground_truth:
    gt_mode = 'gt_pos'
elif use_distance_ground_truth and not use_position_ground_truth:
    gt_mode = 'gt_dis'
elif use_distance_ground_truth and use_position_ground_truth:
    gt_mode = 'gt_all'
else:
    gt_mode = ''

# path to mean error plots / files
scenario_string = ''.join([str(i) for i in scenarios])
fn_mean_error = 'mean_error_scenarios_' + scenario_string + '_' + gt_mode + '_' + str(sample_sizes[-1]) + '.' + plot_format
dir_mean_error = os.path.join('simulation_results', date)
path_mean_error = os.path.join(dir_mean_error, fn_mean_error)
# create mean plot directory
if (save_plots or save_errors_csv) and not os.path.isdir(dir_mean_error):
    os.mkdir(dir_mean_error)
    print('Directory ' + dir_mean_error + ' created.')

# create dataframe for storing the errors of each scenario
errors_per_scenario_max_idx = 500 // sample_sizes[0]
errors_per_scenario_indices = np.arange(errors_per_scenario_max_idx)
errors_all_max_idx = scenarios[-1] * errors_per_scenario_max_idx
errors_all_indices = np.arange(errors_all_max_idx)
errors_df = pd.DataFrame(columns=sample_sizes)
input_cols_number = 3 * len(sample_sizes)
input_cols = np.arange(input_cols_number)
input_errors_ranging_df = pd.DataFrame(columns=input_cols)
input_errors_position_df = pd.DataFrame(columns=input_cols)

if save_errors_csv or save_input_errors_csv:  # unfortunately there are no multidimensional pandas dataframes
    fn_errors_mean_csv = 'errors_all_' + gt_mode + '.csv'
    fn_input_errors_ranging_csv = 'input_errors_ranging.csv'
    fn_input_errors_position_csv = 'input_errors_position.csv'
    path_errors_all_csv = os.path.join(dir_mean_error, fn_errors_mean_csv)
    path_input_errors_ranging_csv = os.path.join(dir_mean_error, fn_input_errors_ranging_csv)
    path_input_errors_position_csv = os.path.join(dir_mean_error, fn_input_errors_position_csv)

# iterate over all scenarios
for scenario in scenarios:
    log_dir = "measurements/scenario_" + str(scenario)
    plot_dir = os.path.join(log_dir, 'plots', date)
    print('Simulating scenario ' + str(scenario))
    errors_per_scenario = pd.DataFrame(index=errors_per_scenario_indices, columns=sample_sizes)
    input_errors_ranging_per_scenario = pd.DataFrame(index=errors_per_scenario_indices, columns=input_cols)
    input_errors_position_per_scenario = pd.DataFrame(index=errors_per_scenario_indices, columns=input_cols)
    # iterate over all experiments
    for i in experiments:
        # path to log files
        exp_no = i
        fn_vicon = 'EXPERIMENT_' + str(exp_no) + '_VICON.log'
        fn_log = 'EXPERIMENT_' + str(exp_no) + '_LOGGING.log'
        fn_range = 'EXPERIMENT_' + str(exp_no) + '_RANGING.log'
        path_vicon = os.path.join(log_dir, fn_vicon)
        path_log = os.path.join(log_dir, fn_log)
        path_range = os.path.join(log_dir, fn_range)

        # path to 2d/3d plot files
        fn_2d = 'scenario_' + str(log_dir[-1]) + '_' + gt_mode + '_' + str(sample_sizes[-1]) + '_2d.' + plot_format
        fn_3d = 'scenario_' + str(log_dir[-1]) + '_' + gt_mode + '_' + str(sample_sizes[-1]) + '_3d.' + plot_format
        path_2d = os.path.join(plot_dir, fn_2d)
        path_3d = os.path.join(plot_dir, fn_3d)
        # path to error plot files
        fn_error = 'scenario_' + str(log_dir[-1]) + '_' + gt_mode + '_' + str(sample_sizes[-1]) + '_error.' + plot_format
        path_error = os.path.join(plot_dir, fn_error)

        # create plot directory
        if save_plots and not os.path.isdir(plot_dir):
            os.mkdir(os.path.dirname(plot_dir))
            os.mkdir(plot_dir)
            print('Directory ' + plot_dir + ' created.')

        # read data files
        header_log = ["ms", "id", "no", "x", "y", "z", "t/r"]
        header_vicon = ["ms", "id", "x", "y", "z"]
        header_range = ["ms", "ctr", "ctr/msmt", "r", "sys_time", "Null1", "Null2"]
        df_drone = pd.read_csv(path_log, header=None, names=header_log)
        df_vicon = pd.read_csv(path_vicon, header=None, names=header_vicon)
        try:  # this file does not always exist
            df_range = pd.read_csv(path_range, header=None, names=header_range)
        except:
            df_range = None
            pass

        # separate different measurement types
        drone_node, drone_meas, drone_drone = [pd.DataFrame(y) for x, y in df_drone.groupby('id', as_index=False)]
        vicon_drone, vicon_node = [pd.DataFrame(y) for x, y in df_vicon.groupby('id', as_index=False)]
        # invert drone's y axis
        drone_meas.loc[:, 'y'] = -drone_meas.loc[:, 'y']
        drone_drone.loc[:, 'y'] = -drone_drone.loc[:, 'y']
        drone_node.loc[:, 'y'] = -drone_node.loc[:, 'y']
        # Get estimated drone position just after ranging
        est_pos_after_ranging_idx = drone_meas.index + 2  # for 1 not all entries exist
        est_pos_after_ranging = [drone_drone[drone_drone.index == idx] for idx in est_pos_after_ranging_idx]  # This is an array of DataFrames
        # check that those entries really exist
        check = [est_pos_after_ranging_idx[i] in drone_drone.index for i in np.arange(3)]
        if not np.array(check).all():
            print('Error in Scenario ' + str(scenario))
        # reset all indices to start at zero / fill gaps
        drone_node = drone_node.reset_index()
        drone_meas = drone_meas.reset_index()
        drone_drone = drone_drone.reset_index()
        vicon_drone = vicon_drone.reset_index()
        vicon_node = vicon_node.reset_index()
        # rename measurement's header
        drone_meas = drone_meas.rename(columns={"t/r": "r"})

        # extract drone's coordinates (in cm)
        x_drone_drone = drone_drone.loc[:, "x"].to_numpy()
        y_drone_drone = drone_drone.loc[:, "y"].to_numpy()
        z_drone_drone = drone_drone.loc[:, "z"].to_numpy()
        x_vicon_drone = vicon_drone.loc[:, "x"].to_numpy() / 10
        y_vicon_drone = vicon_drone.loc[:, "y"].to_numpy() / 10
        z_vicon_drone = vicon_drone.loc[:, "z"].to_numpy() / 10
        # get drone's origin in vicon's coordinate system
        drone_origin_vicon = [x_vicon_drone[0], y_vicon_drone[0], z_vicon_drone[0]]
        # set vicon's origin to drone's origin
        x_vicon_drone -= drone_origin_vicon[0]
        y_vicon_drone -= drone_origin_vicon[1]
        z_vicon_drone -= drone_origin_vicon[2]
        # convert back to data frame
        x_vicon_drone = pd.DataFrame(x_vicon_drone)
        y_vicon_drone = pd.DataFrame(y_vicon_drone)
        z_vicon_drone = pd.DataFrame(z_vicon_drone)
        # make sure not to have two verisons of data
        drone_drone = None
        #vicon_drone = None

        # extract node's coordinates (in cm)
        x_vicon_node = vicon_node.iloc[0].at["x"] / 10
        y_vicon_node = vicon_node.iloc[0].at["y"] / 10
        z_vicon_node = vicon_node.iloc[0].at["z"] / 10
        x_drone_node = drone_node.iloc[-1].at["x"] * 100
        y_drone_node = drone_node.iloc[-1].at["y"] * 100
        z_drone_node = drone_node.iloc[-1].at["z"] * 100
        # set vicon's origin to drone's origin
        x_vicon_node -= drone_origin_vicon[0]
        y_vicon_node -= drone_origin_vicon[1]
        z_vicon_node -= drone_origin_vicon[2]
        # make sure not to have two versions of data
        vicon_node = None
        drone_node = None

        # use the mean of the last estimated position before and the first estimated position after the ranging
        x_mean_est_pos = [(est_pos_after_ranging[i].loc[:, 'x'].to_numpy()[0] + drone_meas.loc[i, 'x']) / 2 for i in np.arange(3)]
        y_mean_est_pos = [(est_pos_after_ranging[i].loc[:, 'y'].to_numpy()[0] + drone_meas.loc[i, 'y']) / 2 for i in np.arange(3)]
        z_mean_est_pos = [(est_pos_after_ranging[i].loc[:, 'z'].to_numpy()[0] + drone_meas.loc[i, 'z']) / 2 for i in np.arange(3)]
        # extract drone's ranging measurements (in cm)
        x_drone_meas = x_mean_est_pos
        y_drone_meas = y_mean_est_pos
        z_drone_meas = z_mean_est_pos
        r_drone_meas = drone_meas.loc[:, "r"].to_numpy()
        # get measurements used for trilateration
        measurement1 = drone_meas.iloc[0, :]
        measurement1.at['x'] = x_mean_est_pos[0]
        measurement1.at['y'] = y_mean_est_pos[0]
        measurement1.at['z'] = z_mean_est_pos[0]
        measurement2 = drone_meas.iloc[1, :]
        measurement2.at['x'] = x_mean_est_pos[1]
        measurement2.at['y'] = y_mean_est_pos[1]
        measurement2.at['z'] = z_mean_est_pos[1]
        measurement3 = drone_meas.iloc[2, :]
        measurement3.at['x'] = x_mean_est_pos[2]
        measurement3.at['y'] = y_mean_est_pos[2]
        measurement3.at['z'] = z_mean_est_pos[2]
        # make sure not to have two versions of data
        drone_meas = None

        # extract responder's ranging measurements (in cm)
        if df_range is not None:
            # convert ranging measurements from m to cm
            df_range['r'] = df_range['r'] * 100
            # separate the three waypoints
            waypoint_idx = df_range.index[df_range['ctr/msmt'] == 0].tolist()
            wp1_idx_offset = waypoint_idx[0]  # zero
            wp2_idx_offset = waypoint_idx[1]
            wp3_idx_offset = waypoint_idx[2]
            arr1 = np.arange(wp1_idx_offset, wp2_idx_offset)
            arr2 = np.arange(wp2_idx_offset, wp3_idx_offset)
            arr3 = np.arange(wp3_idx_offset, df_range.index[-1] + 1)  # until end of file
            ranges1 = df_range[df_range.index.isin(arr1)]
            ranges2 = df_range[df_range.index.isin(arr2)]
            ranges3 = df_range[df_range.index.isin(arr3)]
            # reset indices of waypoint dataframes
            ranges1 = ranges1.reset_index()
            ranges2 = ranges2.reset_index()
            ranges3 = ranges3.reset_index()
        # iterate through sample sizes
        for sample_ctr, sample_size in enumerate(sample_sizes):
            # set up simulation
            meas_per_wp = min(ranges1.shape[0], ranges2.shape[0], ranges3.shape[0])
            # sample_size = 5
            sample_range = np.arange(sample_size)
            num_samples = meas_per_wp // sample_size  # integer division
            range_means1 = np.empty(num_samples)
            range_means2 = np.empty(num_samples)
            range_means3 = np.empty(num_samples)
            errors_per_sample_size = np.empty(num_samples)
            input_cols_per_sample_size = [3*sample_ctr, 3*sample_ctr+1, 3*sample_ctr+2]
            input_errors_ranging_per_sample_size = pd.DataFrame(index=np.arange(num_samples), columns=input_cols_per_sample_size)
            # iterate through the samples
            for j in np.arange(num_samples):
                jth_sample_range = sample_range + j * sample_size
                # calculate ranging means of the sample sets for each waypoint
                mean1 = ranges1[ranges1.index.isin(jth_sample_range)]['r'].mean()
                range_means1[j] = mean1
                mean2 = ranges2[ranges2.index.isin(jth_sample_range)]['r'].mean()
                range_means2[j] = mean2
                mean3 = ranges3[ranges3.index.isin(jth_sample_range)]['r'].mean()
                range_means3[j] = mean3
                # use calculated ranging distances for plotting (the one of the last sample set)
                r_drone_meas[0] = mean1
                r_drone_meas[1] = mean2
                r_drone_meas[2] = mean3
                # use calculated ranging distances for trilateration
                measurement1.at['r'] = mean1
                measurement2.at['r'] = mean2
                measurement3.at['r'] = mean3
                if use_position_ground_truth | use_distance_ground_truth:
                    # calculate mean of ground truth position for each waypoint
                    # waypoint 1
                    wp1_sample_range = ranges1[ranges1.index.isin(jth_sample_range)]
                    wp1_sample_range_times = wp1_sample_range.loc[:, 'ms']
                    vicon_drone_index_wp1 = [vicon_drone['ms'].sub(time).abs().idxmin() for time in wp1_sample_range_times]
                    x_mean1 = x_vicon_drone[x_vicon_drone.index.isin(vicon_drone_index_wp1)].mean()[0]
                    y_mean1 = y_vicon_drone[y_vicon_drone.index.isin(vicon_drone_index_wp1)].mean()[0]
                    z_mean1 = z_vicon_drone[z_vicon_drone.index.isin(vicon_drone_index_wp1)].mean()[0]
                    # waypoint 2
                    wp2_sample_range = ranges2[ranges2.index.isin(jth_sample_range)]
                    wp2_sample_range_times = wp2_sample_range.loc[:, 'ms']
                    vicon_drone_index_wp2 = [vicon_drone['ms'].sub(time).abs().idxmin() for time in wp2_sample_range_times]
                    x_mean2 = x_vicon_drone[x_vicon_drone.index.isin(vicon_drone_index_wp2)].mean()[0]
                    y_mean2 = y_vicon_drone[y_vicon_drone.index.isin(vicon_drone_index_wp2)].mean()[0]
                    z_mean2 = z_vicon_drone[z_vicon_drone.index.isin(vicon_drone_index_wp2)].mean()[0]
                    # waypoint 3
                    wp3_sample_range = ranges3[ranges3.index.isin(jth_sample_range)]
                    wp3_sample_range_times = wp3_sample_range.loc[:, 'ms']
                    vicon_drone_index_wp3 = [vicon_drone['ms'].sub(time).abs().idxmin() for time in wp3_sample_range_times]
                    x_mean3 = x_vicon_drone[x_vicon_drone.index.isin(vicon_drone_index_wp3)].mean()[0]
                    y_mean3 = y_vicon_drone[y_vicon_drone.index.isin(vicon_drone_index_wp3)].mean()[0]
                    z_mean3 = z_vicon_drone[z_vicon_drone.index.isin(vicon_drone_index_wp3)].mean()[0]
                    if use_position_ground_truth:
                        # use calculated position for plotting (the one of the last sample set)
                        x_drone_meas[0] = x_mean1
                        x_drone_meas[1] = x_mean2
                        x_drone_meas[2] = x_mean3
                        y_drone_meas[0] = y_mean1
                        y_drone_meas[1] = y_mean2
                        y_drone_meas[2] = y_mean3
                        # z_drone_meas[0] = z_mean1
                        # z_drone_meas[1] = z_mean2
                        # z_drone_meas[2] = z_mean3
                        # use calculated position for trilateration
                        measurement1.at['x'] = x_mean1
                        measurement1.at['y'] = y_mean1
                        # measurement1.at['z'] = z_mean1
                        measurement2.at['x'] = x_mean2
                        measurement2.at['y'] = y_mean2
                        # measurement2.at['z'] = z_mean2
                        measurement3.at['x'] = x_mean3
                        measurement3.at['y'] = y_mean3
                        # measurement3.at['z'] = z_mean3
                    if use_distance_ground_truth:
                        # calculate ground truth distance to node for each waypoint
                        pos_delta_x_wp1 = x_mean1 - x_vicon_node
                        pos_delta_y_wp1 = y_mean1 - y_vicon_node
                        pos_delta_z_wp1 = z_mean1 - z_vicon_node
                        r_wp1 = np.linalg.norm([pos_delta_x_wp1, pos_delta_y_wp1, pos_delta_z_wp1], 2)
                        pos_delta_x_wp2 = x_mean2 - x_vicon_node
                        pos_delta_y_wp2 = y_mean2 - y_vicon_node
                        pos_delta_z_wp2 = z_mean2 - z_vicon_node
                        r_wp2 = np.linalg.norm([pos_delta_x_wp2, pos_delta_y_wp2, pos_delta_z_wp2], 2)
                        pos_delta_x_wp3 = x_mean3 - x_vicon_node
                        pos_delta_y_wp3 = y_mean3 - y_vicon_node
                        pos_delta_z_wp3 = z_mean3 - z_vicon_node
                        r_wp3 = np.linalg.norm([pos_delta_x_wp3, pos_delta_y_wp3, pos_delta_z_wp3], 2)
                        # calculate ranging error
                        r_error1 = r_wp1 - wp1_sample_range.loc[:, 'r'].mean()
                        r_error2 = r_wp2 - wp2_sample_range.loc[:, 'r'].mean()
                        r_error3 = r_wp3 - wp3_sample_range.loc[:, 'r'].mean()
                        input_errors_ranging_per_sample_size.iloc[j, :] = [r_error1, r_error2, r_error3]
                        # print(r_error1)
                        # print(r_error2)
                        # print(r_error3)
                        # use calculated distance for plotting (the one of the last sample set)
                        r_drone_meas[0] = r_wp1
                        r_drone_meas[1] = r_wp2
                        r_drone_meas[2] = r_wp3
                        # use calculated distance for trilateration
                        measurement1.at['r'] = r_wp1
                        measurement2.at['r'] = r_wp2
                        measurement3.at['r'] = r_wp3
                        # use ground truth altitude for projection to ground
                        measurement1.at['z'] = pos_delta_z_wp1
                        measurement2.at['z'] = pos_delta_z_wp2
                        measurement3.at['z'] = pos_delta_z_wp3
                # create Measurement3d objects
                m1 = Measurement3d(measurement1.at['x'], measurement1.at['y'], measurement1.at['z'], measurement1.at['r'])
                m2 = Measurement3d(measurement2.at['x'], measurement2.at['y'], measurement2.at['z'], measurement2.at['r'])
                m3 = Measurement3d(measurement3.at['x'], measurement3.at['y'], measurement3.at['z'], measurement3.at['r'])
                # calculate trilateration
                estimated_node_pos = trilateration(m1, m2, m3)
                # use calculated node pos for plotting (the one of the last sample set)
                x_drone_node = estimated_node_pos.x
                y_drone_node = estimated_node_pos.y
                # calculate error
                delta_x = estimated_node_pos.x - x_vicon_node
                delta_y = estimated_node_pos.y - y_vicon_node
                l2_error = np.linalg.norm([delta_x, delta_y], 2)
                errors_per_sample_size[j] = l2_error
                if save_plots or show_plots:
                    break
            # for j in samples end
            # Store errors_per_sample_size in errors_per_scenario
            NaNs = pd.DataFrame(index=errors_per_scenario_indices[len(errors_per_sample_size):], columns=[sample_size])
            errors_per_sample_size = pd.concat([pd.DataFrame(errors_per_sample_size, columns=[sample_size]), NaNs])
            errors_per_scenario.loc[:, sample_size] = errors_per_sample_size
            # Store input_errors_ranging_per_sample_size in input_errors_ranging_per_scenario
            NaNs_input = pd.DataFrame(index=errors_per_scenario_indices[len(errors_per_sample_size):], columns=[0])
            temp = pd.concat([input_errors_ranging_per_sample_size.iloc[:, 0], NaNs_input])
            input_errors_ranging_per_scenario.loc[:, input_cols_per_sample_size[0]] = temp.loc[:, 0]
            temp = pd.concat([input_errors_ranging_per_sample_size.iloc[:, 1], NaNs_input])
            input_errors_ranging_per_scenario.loc[:, input_cols_per_sample_size[1]] = temp.loc[:, 0]
            temp = pd.concat([input_errors_ranging_per_sample_size.iloc[:, 2], NaNs_input])
            input_errors_ranging_per_scenario.loc[:, input_cols_per_sample_size[2]] = temp.loc[:, 0]
            print('Sample size ' + str(sample_size) + ' completed')
        # for sample_size in sample_sizes end

        # ------------------------------------------ Plotting ---------------------------------------------------------
        # construct ranging circles (projected to ground)
        angle = np.linspace(0, 2 * np.pi, 600)
        r_circ = [[math.sqrt(pow(r_drone_meas[ctr], 2) - pow(height, 2))] for ctr, height in enumerate(z_drone_meas)]
        x_circ = [[r * np.cos(angle)] for r in r_circ]
        y_circ = [[r * np.sin(angle)] for r in r_circ]
        z_circ = [0, 0]
        # shift circles to measuring point
        x_circ = [[x_circ[ctr] + val] for ctr, val in enumerate(x_drone_meas)]
        y_circ = [[y_circ[ctr] + val] for ctr, val in enumerate(y_drone_meas)]

        # plot 2d
        fig1 = plt.figure(figsize=(8, 8))
        ax = plt.axes()
        ax.scatter(x_drone_drone, y_drone_drone, s=2, color='blue', label="Drone's pos. estimates")
        ax.scatter(x_drone_node, y_drone_node, s=100, color='blue')
        ax.scatter(x_vicon_drone, y_vicon_drone, s=2, color='red', label="Ground truth pos.")
        ax.scatter(x_vicon_node, y_vicon_node, s=100, color='red')
        # plot ranging measurements of drone
        colors = ['green', 'orange', 'purple']
        labels = ['Measurement 1', 'Measurement 2', 'Measurement 3']
        for ctr, val in enumerate(x_circ):
            ax.scatter(x_circ[ctr], y_circ[ctr], s=2, color=colors[ctr], label=labels[ctr])
            ax.scatter(x_drone_meas[ctr], y_drone_meas[ctr], s=100, color=colors[ctr])
        ax.set_title('Scenario ' + str(log_dir[-1]), fontsize=20)
        plt.tick_params(labelsize=15)
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)
        ax.legend(scatterpoints=1, markerscale=7, fontsize=14)
        plt.grid('show')
        # ax.grid(True)
        if show_plots:
            plt.show()
        else:
            plt.close(fig1)
        if save_plots:
            fig1.savefig(path_2d, format=plot_format, transparent=False, facecolor='white')
            plt.close(fig1)


        # plot 3d
        # fig2 = plt.figure(figsize=(8, 8))
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(x_drone_drone, y_drone_drone, z_drone_drone, s=2, color='blue', label="drone's pos. estimates")
        # ax.scatter3D(x_drone_node, y_drone_node, 0, s=50, color='blue')
        # ax.scatter3D(x_vicon_drone, y_vicon_drone, z_vicon_drone, s=2, color='red', label="ground truth")
        # ax.scatter3D(x_vicon_node, y_vicon_node, 0, s=50, color='red')
        # colors = ['green', 'orange', 'purple']
        # labels = ['Measurement 1', 'Measurement 2', 'Measurement 3']
        # for ctr, val in enumerate(x_circ):
        #     ax.scatter(x_circ[ctr], y_circ[ctr], 0, s=2, color=colors[ctr], label=labels[ctr])
        #     ax.scatter(x_drone_meas[ctr], y_drone_meas[ctr], 0, s=30, color=colors[ctr])
        # ax.set_title('Scenario ' + str(log_dir[-1]) + ' 3D')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # ax.legend(scatterpoints=1, markerscale=5)
        # plt.grid('show')
        # if show_plots:
        #     plt.show()
        # else:
        #     plt.close(fig2)
        # if save_plots:
        #     fig2.savefig(path_3d, format=plot_format, transparent=False, facecolor='white')
        #     plt.close(fig2)

    # for i in experiments end
    if save_errors_csv:
        errors_df = pd.concat([errors_df, errors_per_scenario])
    if save_input_errors_csv:
        input_errors_ranging_df = pd.concat([input_errors_ranging_df, input_errors_ranging_per_scenario])
# for scenario in scenarios end
if save_errors_csv:
    errors_df.to_csv(path_errors_all_csv, mode='a')
if save_input_errors_csv:
    input_errors_ranging_df.to_csv(path_input_errors_ranging_csv, mode='a')
    # input_errors_position_df.to_csv(path_input_errors_position_csv, mode='a')


