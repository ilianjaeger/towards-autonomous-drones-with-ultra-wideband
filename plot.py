import matplotlib.pyplot as plt
plt.style.use(['science', 'grid', 'bright'])  # bright, high-contrast, retro, high-vis, muted, vibrant
import pandas as pd
import numpy as np
import os

# set metadata
save = True
abs_errors_vs_samplesizes = False
rel_errors_vs_samplesizes = False
errors_vs_samplesize = False
scenarios_mean = False
scenarios_box = False
input_error_ranging = True
date = '2021-03-31'
plot_format = 'svg'
scenarios = [1, 2, 3, 4, 5, 6]
max_samplesize = 20
specific_samplesize = 1
fontsize = 25
# sample_sizes = np.arange(10,101,10).tolist()
sample_sizes = [1, 10, 20, 30, 40, 50]
# sample_sizes = [1, 2, 4, 6, 8, 10]

# start plotting
gt_modes = ['', 'gt_pos', 'gt_dis']
for i, gt_mode in enumerate(gt_modes):
    # read csv data
    csv_dir = os.path.join('simulation_results', date)
    plot_dir = os.path.join(csv_dir, 'plots')
    if save and not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
        print('Directory ' + plot_dir + ' created.')
    fn_errors_csv = 'errors_all_' + gt_mode + '.csv'
    # fn_errors_plot = 'errors_all_' + gt_mode + '.' + plot_format
    path_errors_csv = os.path.join(csv_dir, fn_errors_csv)
    # path_errors_plot = os.path.join(plot_dir, fn_errors_plot)
    # read files
    errors_df = pd.read_csv(path_errors_csv)
    gt_labels = ['Measured Data', 'Ground Truth Position']
    gt_labels_all = ['Measured Data', 'Ground Truth Position', 'Ground Truth Distance']

    # ABSOLUTE ERRORS vs. sample size boxplot for mean over all scenarios
    if abs_errors_vs_samplesizes and (gt_mode != 'gt_dis'):
        # data = errors_df.iloc[:, 1:(max_samplesize+1)]
        data = errors_df.iloc[:, sample_sizes]
        boxprops = dict(color='black')
        medianprops = dict(color='red')
        fig = data.plot(kind='box', whis=(0, 100), fontsize=fontsize, figsize=(8, 8), color=dict(boxes='black', whiskers='black', medians='r', caps='black')).get_figure()
        # fig = plt.figure(figsize=(8,8))
        # plt.boxplot(data, whis=(0, 100))
        ax = plt.axes()
        ax.set_ylim(-5, 75)
        # plt.title('Localization Error for ' + str(gt_labels[i]), fontsize=fontsize)
        plt.xlabel('Measurements per Waypoint', fontsize=fontsize)
        plt.ylabel('Absolute Localization Error [cm]', fontsize=fontsize)
        if not save:
            plt.show()
        if save:
            fn_errors_vs_samplesizes = 'abs_errors_samplesizes_' + gt_mode + '_' + str(sample_sizes[-1]) + '.' + plot_format
            path_errors_vs_samplesizes = os.path.join(plot_dir, fn_errors_vs_samplesizes)
            fig.savefig(path_errors_vs_samplesizes, format=plot_format, transparent=False, facecolor='white')
            plt.close(fig)

    # plot data per scenario
    # split up scenarios
    scenario_boundary_idcs = errors_df.index[errors_df.iloc[:, 0] == 0].tolist()
    boundaries = scenario_boundary_idcs
    boundaries.append(errors_df.index[-1]+1)
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'black']
    # scenario_labels = ['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4', 'Scenario 5', 'Scenario 6']
    scenario_labels = ['1', '2', '3', '4', '5', '6']

    # RELATIVE ERRORS vs. sample size boxplot for mean over all scenarios
    if rel_errors_vs_samplesizes and (gt_mode != 'gt_dis'):
        rel_error_array = []
        # print(errors_df)
        for scenario in scenarios:
            scenario_df = errors_df[boundaries[scenario-1]: boundaries[scenario]]
            scenario_df = scenario_df.iloc[:, sample_sizes]
            median = scenario_df.median()[0]
            scenario_df = scenario_df - median
            rel_error_array.append(scenario_df)
        rel_error_df = pd.concat(rel_error_array)
        # fig = rel_error_df.boxplot(figsize=(8, 8), wisk=(5,95)).get_figure()
        fig = rel_error_df.plot(kind='box', figsize=(8, 8), whis=(0, 100), fontsize=fontsize, color=dict(boxes='black', whiskers='black', medians='r', caps='black')).get_figure()
        ax = plt.axes()
        ax.set_ylim(-30, 45)
        plt.xlabel('Measurements per Waypoint', fontsize=fontsize)
        plt.ylabel('Relative Localization Error [cm]', fontsize=fontsize)
        # plt.title('Relative Error for ' + str(gt_labels[i]), fontsize=fontsize)
        if not save:
            plt.show()
        if save:
            fn_rel_errors = 'rel_errors_samplesizes_' + gt_mode + '_' + str(sample_sizes[-1]) + '.' + plot_format
            path_rel_errors = os.path.join(plot_dir, fn_rel_errors)
            fig.savefig(path_rel_errors, format=plot_format, transparent=False, facecolor='white')
            plt.close(fig)

    # plot mean of all scenarios in one plot
    if scenarios_mean and (gt_mode != 'gt_dis'):
        fig1 = plt.figure(figsize=(8, 8))
        ax = plt.axes()
        ax.set_ylim(0,60)
        plt.tick_params(labelsize=fontsize / 1.5)
        for scenario in scenarios:
            scenario_df = errors_df[boundaries[scenario-1]: boundaries[scenario]]
            # scenario_df = scenario_df.iloc[:, 1:(max_samplesize+1)]
            scenario_df = scenario_df.iloc[:, sample_sizes]
            # print batch means of scenario
            means = scenario_df.mean()
            ax.scatter(means.index, means, label='Scenario ' + str(scenario), color=colors[scenario-1], marker='x', s=200)
            plt.plot(means.index, means, color=colors[scenario-1], linestyle='dashed', linewidth=2)
        plt.tick_params(labelsize=fontsize)
        # plt.title('Comparison of Mean Error for ' + str(gt_labels[i]), fontsize=fontsize)
        ax.set_xlabel('Measurements per Waypoint', fontsize=fontsize)
        ax.set_ylabel('Mean Localization Error [cm]', fontsize=fontsize)
        ax.legend(scatterpoints=1, markerscale=1, fontsize=fontsize/1.1)
        if not save:
            plt.show()
        if save:
            fn_errors_mean = 'errors_samplesizes_mean_' + gt_mode + '_' + str(sample_sizes[-1]) + '.' + plot_format
            path_errors_mean = os.path.join(plot_dir, fn_errors_mean)
            fig1.savefig(path_errors_mean, format=plot_format, transparent=False, facecolor='white')
            plt.close(fig1)

    # plot boxplot of all scenarios separately
    if scenarios_box:
        fn_errors_means = 'errors_all_means_' + gt_mode + '.csv'
        path_errors_means = os.path.join(plot_dir, fn_errors_means)
        data = np.empty([500, 6])
        for scenario in scenarios:
            scenario_df = errors_df[boundaries[scenario-1]: boundaries[scenario]]

            scenario_df.mean().to_csv(path_errors_means)
            # print(scenario_df.iloc[:, specific_samplesize].mean())
            data[:, scenario-1] = scenario_df.iloc[:, specific_samplesize]
        data1 = pd.DataFrame(data)
        print(gt_mode)
        print(data1.max())
        print(data1.quantile(0.75))
        print(data1.median())
        print(data1.quantile(0.25))
        print(data1.min())
        print(data1.mean())
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes()
        if gt_mode == '':
            plt.boxplot(data, whis=(0, 100))
        else:
            if gt_mode == 'gt_pos':
                boxcolors = ['tomato', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen']
            else:
                boxcolors = ['tomato', 'lightgreen', 'tomato', 'lightgreen', 'lightgreen', 'tomato']

            box = plt.boxplot(data, whis=(0, 100), patch_artist=True, medianprops=dict(color='black'))
            for patch, color in zip(box['boxes'], boxcolors):
                patch.set_facecolor(color)
            ax.legend([box["boxes"][0], box["boxes"][1]], ['Median error increased', 'Median error decreased'], fontsize=fontsize)
        ax.set(xticklabels=scenario_labels)
        plt.tick_params(labelsize=fontsize)
        #plt.xticks(rotation=30)
        ax.set_ylim(-5, 75)
        # plt.title('Comparison of Errors for ' + str(gt_labels_all[i]), fontsize=fontsize)
        plt.xlabel('Scenario', fontsize=fontsize)
        plt.ylabel('Localization Error [cm]', fontsize=fontsize)
        if not save:
            plt.show()
        if save:
            fn_errors_box = 'errors_all_' + gt_mode + '_' + str(specific_samplesize) + '_colored.' + plot_format
            path_errors_box = os.path.join(plot_dir, fn_errors_box)
            fig.savefig(path_errors_box, format=plot_format, transparent=False, facecolor='white')
            plt.close(fig)

# error for specific sample size boxplot for mean over all scenarios
if errors_vs_samplesize:
    gt_labels = ['Measured Data', 'GT Pos', 'GT Dist']
    # gt_modes = ['', 'gt_pos', 'gt_dis']
    gt_modes = ['']
    errors = pd.DataFrame(index=np.arange(3000), columns=gt_modes)
    for gt_mode in gt_modes:
        # read csv data
        fn_errors_csv = 'errors_all_' + gt_mode + '.csv'
        # fn_errors_plot = 'errors_all_' + gt_mode + '.' + plot_format
        path_errors_csv = os.path.join(csv_dir, fn_errors_csv)
        # path_errors_plot = os.path.join(plot_dir, fn_errors_plot)
        # read files
        errors_df = pd.read_csv(path_errors_csv)
        specific_col = errors_df.iloc[:, specific_samplesize]
        errors.loc[:, gt_mode] = specific_col
        print(gt_mode)
        print(specific_col.max())
        print(specific_col.quantile(0.75))
        print(specific_col.median())
        print(specific_col.quantile(0.25))
        print(specific_col.min())
        print(specific_col.mean())

    # fig = errors.plot(kind='box', whis=(5, 95), fontsize=fontsize, figsize=(8, 8)).get_figure()
    # fig = plt.figure(figsize=(8, 8))
    fig = plt.figure(figsize=(4, 8))
    ax = plt.axes()
    plt.boxplot(errors, whis=(0, 100))
    # ax.set(xticklabels=gt_labels)
    ax.set(xticklabels=['1-6'])
    plt.tick_params(labelsize=fontsize)
    # plt.xticks(rotation=30)
    # plt.title('Mean Localization Error', fontsize=fontsize)
    plt.xlabel('Scenario', fontsize=fontsize)
    plt.ylabel('Localization Error [cm]', fontsize=fontsize)
    if not save:
        plt.show()
    if save:
        # fn_errors_vs_samplesize = 'errors_gt_comparison_all_' + str(specific_samplesize) + '.' + plot_format
        fn_errors_vs_samplesize = 'errors_gt_comparison_' + str(specific_samplesize) + '.' + plot_format
        path_errors_vs_samplesize = os.path.join(plot_dir, fn_errors_vs_samplesize)
        fig.savefig(path_errors_vs_samplesize, format=plot_format, transparent=False, facecolor='white')
        plt.close(fig)


if input_error_ranging:
    fn_errors_csv = 'input_errors_ranging.csv'
    path_errors_csv = os.path.join(csv_dir, fn_errors_csv)
    # read files
    errors_df = pd.read_csv(path_errors_csv)
    end_col = len(errors_df.columns)
    plot_cols = np.arange(end_col//3)+1
    wp1 = errors_df.iloc[:, 1:end_col:3]
    wp1.columns = plot_cols
    wp2 = errors_df.iloc[:, 2:end_col:3]
    wp2.columns = plot_cols
    wp3 = errors_df.iloc[:, 3:end_col:3]
    wp3.columns = plot_cols
    # iterate through scenarios
    scenarios = [4]
    boundaries = [0,500,1000,1500,2000,2500,3000]
    for scenario in scenarios:
        scenario_df = errors_df[boundaries[scenario - 1]: boundaries[scenario]]
        # scenario_df = errors_df[500: 1000]
        fig = wp1.plot(kind='box', whis=(0,100), fontsize=fontsize, figsize=(8, 8)).get_figure()
        ax = plt.axes()
        plt.title('Ranging Error for Scenario ' + str(scenario), fontsize=fontsize)
        ax.set_xlabel('Measurements per Waypoint', fontsize=fontsize)
        ax.set_ylabel('Ranging Error [cm]', fontsize=fontsize)
        # ax.legend(scatterpoints=1, markerscale=1, fontsize=fontsize / 1.1)
        plt.tick_params(labelsize=fontsize / 1.5)
        if not save:
            plt.show()
        if save:
            fn_input_errors_ranging = 'input_errors_ranging_scenario_' + str(scenario) + '.' + plot_format
            path_input_errors_ranging = os.path.join(plot_dir, fn_input_errors_ranging)
            fig.savefig(path_input_errors_ranging, format=plot_format, transparent=False, facecolor='white')
            plt.close(fig)
