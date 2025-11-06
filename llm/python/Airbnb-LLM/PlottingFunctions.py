import numpy as np

def plot_flexible_radar_GI(ax, indicators, all_vars, color='blue', title='Radar Chart',
                        show_std=False, group_labels=True, label_size=10, fixed_max=None, show_radial_labels=True):

    groups = {
    'RMD':  ['rmd_z'],
    'House Price': ['house_price_z'],
    'Demographic': ['BME_z'],
    'Mobility': ['churn_z'],
    'Wages': ['wage_change_z']}


    # Calculate stats
    means = indicators[all_vars].mean().values.tolist()
    means += means[:1]
    angles = np.linspace(0, 2 * np.pi, len(all_vars), endpoint=False).tolist()
    angles += angles[:1]

    if show_std:
        stds = indicators[all_vars].std().values.tolist()
        stds += stds[:1]
        std_upper = np.array(means) + np.array(stds)
        std_lower = np.array(means) - np.array(stds)
        max_val = max(std_upper) * 1.1
    else:
        std_upper = std_lower = None
        max_val = max(means) * 1.1

    if fixed_max is not None:
        max_val = fixed_max

    # Set scale and draw background
    if fixed_max is not None:
        ax.set_ylim(-fixed_max, fixed_max)
    else:
        data_min = min(means if not show_std else std_lower)
        data_max = max(means if not show_std else std_upper)
        buffer = (data_max - data_min) * 0.1
        ax.set_ylim(data_min - buffer, data_max + buffer)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    # Optionally show radial circle labels
    if show_radial_labels:
        n_circles = 4
        min_ylim, max_ylim = ax.get_ylim()
        ticks = np.linspace(min_ylim, max_ylim, n_circles + 1)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.1f}" for tick in ticks], fontsize=9)
    else:
        ax.set_yticklabels([])

    # Plot main line
    ax.plot(angles, means, color=color, linewidth=4, label='Mean')

    # Optional std shading
    if show_std and std_upper is not None:
        n_steps = 10
        for i in range(n_steps):
            alpha = 0.1 # * (1 - i / n_steps)
            lower = std_lower + (np.array(means) - std_lower) * (i / n_steps)
            upper = std_upper - (std_upper - np.array(means)) * (i / n_steps)
            ax.fill_between(angles, lower, upper, color=color, alpha=alpha)

    # Title
    ax.set_title(title, size=13, pad=30)

    # Group labels and separators
    if group_labels:
        group_labels_dict = {}
        separators = []
        current_idx = 0
        for group_name, group_vars in groups.items():
            count = len(group_vars)
            group_indices = list(range(current_idx, current_idx + count))
            group_angles = [angles[i] for i in group_indices]
            middle_angle = np.mean(group_angles)
            group_labels_dict[group_name] = middle_angle
            end_idx = current_idx + count
            if end_idx < len(all_vars):
                separators.append(angles[end_idx])
            current_idx = end_idx

        # Draw separators
        for sep in separators:
            ax.plot([sep, sep], [0, max_val], color='black', linewidth=2.5, linestyle='--')

        # Draw group headings
        for group_name, angle in group_labels_dict.items():
            ax.text(angle, max_val * 1.15, group_name,
                    ha='center', va='center', fontsize=label_size,
                    fontweight='bold', rotation=np.rad2deg(angle), rotation_mode='anchor')


def plot_mean_rose_plot_with_gradient(ax, indicators, variables, color):

    # Variables and stats
    means = [indicators[f'{var}'].mean() for var in variables]
    std_devs = [indicators[f'{var}'].std() for var in variables]

    # Close the loop
    num_vars = len(variables)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    means += means[:1]
    std_devs += std_devs[:1]

    std_upper = np.array(means) + np.array(std_devs)
    std_lower = np.array(means) - np.array(std_devs)

    # Plot mean line
    ax.plot(angles, means, color='black', linewidth=2, label='Mean')

    # Simulate gradient by plotting multiple thin bands
    n_steps = 10  # Number of gradient layers
    for i in range(n_steps):
        alpha = 0.3 * (1 - i / n_steps)  # Fading alpha for gradient
        lower = std_lower + (means - std_lower) * (i / n_steps)
        upper = std_upper - (std_upper - means) * (i / n_steps)

        # Fill between lower and upper bands
        ax.fill_between(angles, lower, upper, color=color, alpha=alpha)

    # Set variable labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(variables)
    ax.set_yticklabels([])

    # Title
    ax.set_title('Radar Chart with Gradient SD Range', size=14)

# def plot_mean_rose_plot_with_gradient(ax, indicators, color):
#     # Variables and stats
#     variables = ['NINO', 'Churn', 'JSA', 'HP', 'PROF2POP']
#     means = [indicators[f'{var}_mean'].mean() for var in variables]
#     std_devs = [indicators[f'{var}_mean'].std() for var in variables]

#     # Close the loop
#     num_vars = len(variables)
#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#     angles += angles[:1]
#     means += means[:1]
#     std_devs += std_devs[:1]

#     std_upper = np.array(means) + np.array(std_devs)
#     std_lower = np.array(means) - np.array(std_devs)

#     # Plot mean line
#     ax.plot(angles, means, color='black', linewidth=2, label='Mean')

#     # Simulate gradient by plotting multiple thin bands
#     n_steps = 10  # Number of gradient layers
#     for i in range(n_steps):
#         alpha = 0.3 * (1 - i / n_steps)  # Fading alpha for gradient
#         lower = std_lower + (means - std_lower) * (i / n_steps)
#         upper = std_upper - (std_upper - means) * (i / n_steps)

#         # Fill between lower and upper bands
#         ax.fill_between(angles, lower, upper, color=color, alpha=alpha)

#     # Set variable labels
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(variables)
#     ax.set_yticklabels([])

#     # Title
#     ax.set_title('Radar Chart with Gradient SD Range', size=14)

def plot_mean_rose_plot(ax, indicators, color, just_means=False):
    variables = ['NINO', 'Churn', 'JSA', 'HP', 'PROF2POP']
    # calculate means
    means = [float(indicators['NINO_mean'].mean()), float(indicators['Churn_mean'].mean()), float(indicators['JSA_mean'].mean()), 
             float(indicators['HP_mean'].mean()), float(indicators['PROF2POP_mean'].mean())]
    # calculate standard deviations
    std_devs = [float(indicators['NINO_mean'].std()), float(indicators['Churn_mean'].std()), float(indicators['JSA_mean'].std()), 
                float(indicators['HP_mean'].std()), float(indicators['PROF2POP_mean'].std())]
    # calculate angles for plotting
    num_vars = len(variables)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Add the first value to the end (to close the plotting loop)
    angles = angles + angles[:1]
    means = means + means[:1]
    std_devs = std_devs + std_devs[:1]
    
    # Find std above, and below
    std_upper = [mean + std for mean, std in zip(means, std_devs)]
    std_lower = [mean - std for mean, std in zip(means, std_devs)]

    # Plot the mean values
    ax.plot(angles, means, label='Mean', color=color, linewidth=2)
    ax.scatter(angles, means, color=color, s=20)

    # Plot standard deviation below the mean (Mean - Std)
    if just_means==False:
        # # Plot standard deviation above the mean (Mean + Std)
        ax.plot(angles, std_upper, label='Mean + Std', color=color, linestyle='--', linewidth=1)
        ax.scatter(angles, std_upper,  color=color, s=20)
        
        ax.plot(angles, std_lower, label='Mean - Std', color=color, linestyle='--', linewidth=1)
        ax.scatter(angles, std_lower,  color=color, s=20)
    
        # # Fill the area between the mean and the upper standard deviation
        ax.fill(angles, std_upper, color=color, alpha=0.5)
    
        # # Fill the area between the mean and the lower standard deviation
        ax.fill(angles, std_lower, color=color, alpha=0.5)

    # # Add labels for each variable
    # ax.set_yticklabels([])  # Remove radial ticks
    ax.set_xticks(angles[:-1])  # Set the angular ticks
    ax.set_xticklabels(variables, fontsize=12)  # Set the variable labels

def plot_mean_rose_with_group_labels(ax, indicators, color, all_vars):

    groups = {'Churn': [col for col in all_vars if 'Churn' in col], 'NINO': [col for col in all_vars if 'NINO' in col],
             'JSA': [col for col in all_vars if 'JSA' in col], 'HP': [col for col in all_vars if 'HP' in col],
              'PROF2POP': [col for col in all_vars if 'PROF2POP' in col]}
        
    # Example: Calculate means and std_devs (replace with your method)
    means = indicators[all_vars].mean().values.tolist()
    std_devs = indicators[all_vars].std().values.tolist()
    
    # Close loop for radar chart
    means += means[:1]
    std_devs += std_devs[:1]
    
    # Angles
    num_vars = len(all_vars)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Plotting means + gradient fills (your existing method)
    std_upper = np.array(means) + np.array(std_devs)
    std_lower = np.array(means) - np.array(std_devs)

    # Plot mean line
    ax.plot(angles, means, color='black', linewidth=2, label='Mean')

    # Simulate gradient by plotting multiple thin bands
    n_steps = 10  # Number of gradient layers
    for i in range(n_steps):
        alpha = 0.3 * (1 - i / n_steps)  # Fading alpha for gradient
        lower = std_lower + (means - std_lower) * (i / n_steps)
        upper = std_upper - (std_upper - means) * (i / n_steps)

        # Fill between lower and upper bands
        ax.fill_between(angles, lower, upper, color=color, alpha=alpha)

    # Set variable labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Title
    ax.set_title('Radar Chart with Gradient SD Range', size=14)

    # Find group labels and separators
    group_labels = {}
    separators = []
    current_idx = 0
    for group_name, group_vars in groups.items():
        count = len(group_vars)
        group_indices = list(range(current_idx, current_idx + count))
        group_angles = [angles[i] for i in group_indices]
        middle_angle = np.mean(group_angles)
        group_labels[group_name] = middle_angle
        end_idx = current_idx + count
        if end_idx < num_vars:
            separators.append(angles[end_idx])
        current_idx = end_idx

    # Add separators
    max_value = max(np.array(means) + np.array(std_devs)) * 1.1  # or a fixed value
    for sep in separators:
        ax.plot([sep, sep], [0, max_value], color='black', linewidth=2.5, linestyle='--')

    # Add group labels
    for group_name, angle in group_labels.items():
        ax.text(angle, max_value * 1.15, group_name, ha='center', va='center', fontsize=12, fontweight='bold', rotation=np.rad2deg(angle), rotation_mode='anchor')

def plot_rose(ax, indicators, lsoa_code, color):
    this_lsoa = indicators[indicators['LSOA11CD'] == lsoa_code]
    this_lsoa.reset_index(inplace=True, drop=True)
    del this_lsoa['LSOA11CD']
    del this_lsoa['LA_NAME']
    del this_lsoa['gentrification_prediction_code']
    this_lsoa_T = this_lsoa.T
    this_lsoa_T.reset_index(inplace=True)
    this_lsoa_T = this_lsoa_T.rename(columns={0: 'scores'})
    # E01014485_T['max_vals'] = max_vals[0].values
    df = this_lsoa_T.copy()
    # df['pct'] = df['scores'] / df['max_vals']
    df = df.set_index('index')
    
    # Replace NaN values with 0 in the plot data (but keep NaNs in the DataFrame)
    df['scores_for_plot'] = df['scores'].fillna(0)
    
    # Recalculate N and angles after replacing NaNs
    N = df.shape[0]
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    
    # Assign new angles
    df['radar_angles'] = theta
    
    # Plot
    ax.bar(df['radar_angles'], df['scores_for_plot'], width=2 * np.pi / N, linewidth=2, edgecolor='k', alpha=0.6, color=color)
    
    # Adjust labels (keeping original categories with NaNs)
    ax.set_xticks(theta)
    ax.set_xticklabels(df.index)
    ax.set_yticklabels([])

# def plot_rose(ax, indicators, lsoa_code, color):
#     this_lsoa = indicators[indicators['LSOA11CD'] == lsoa_code]
#     this_lsoa.reset_index(inplace=True, drop=True)
#     del this_lsoa['LSOA11CD']
#     del this_lsoa['LA_NAME']
#     del this_lsoa['gentrification_prediction_code']
#     this_lsoa_T = this_lsoa.T
#     this_lsoa_T.reset_index(inplace=True)
#     this_lsoa_T = this_lsoa_T.rename(columns={0: 'scores'})
#     # E01014485_T['max_vals'] = max_vals[0].values
#     df = this_lsoa_T.copy()
#     # df['pct'] = df['scores'] / df['max_vals']
#     df = df.set_index('index')
    
#     # Replace NaN values with 0 in the plot data (but keep NaNs in the DataFrame)
#     df['scores_for_plot'] = df['scores'].fillna(0)
    
#     # Recalculate N and angles after replacing NaNs
#     N = df.shape[0]
#     theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    
#     # Assign new angles
#     df['radar_angles'] = theta
    
#     # Plot
#     ax.bar(df['radar_angles'], df['scores_for_plot'], width=2 * np.pi / N, linewidth=2, edgecolor='k', alpha=0.6, color=color)
    
#     # Adjust labels (keeping original categories with NaNs)
#     ax.set_xticks(theta)
#     ax.set_xticklabels(df.index)
#     ax.set_yticklabels([])



def plot_flexible_radar(ax, indicators, all_vars, color='blue', title='Radar Chart',
                        show_std=False, group_labels=True, label_size=10, fixed_max=None, show_radial_labels=True):

    # Define variable groups (adjust as needed)
    groups = { 'Churn': [col for col in all_vars if 'Churn' in col],
        'NINO': [col for col in all_vars if 'NINO' in col],
        'JSA': [col for col in all_vars if 'JSA' in col],
        'HP': [col for col in all_vars if 'HP' in col],
        'PROF2POP': [col for col in all_vars if 'PROF2POP' in col]}

    # Calculate stats
    means = indicators[all_vars].mean().values.tolist()
    means += means[:1]
    angles = np.linspace(0, 2 * np.pi, len(all_vars), endpoint=False).tolist()
    angles += angles[:1]

    if show_std:
        stds = indicators[all_vars].std().values.tolist()
        stds += stds[:1]
        std_upper = np.array(means) + np.array(stds)
        std_lower = np.array(means) - np.array(stds)
        max_val = max(std_upper) * 1.1
    else:
        std_upper = std_lower = None
        max_val = max(means) * 1.1

    if fixed_max is not None:
        max_val = fixed_max

    # Set scale and draw background
    if fixed_max is not None:
        ax.set_ylim(-fixed_max, fixed_max)
    else:
        data_min = min(means if not show_std else std_lower)
        data_max = max(means if not show_std else std_upper)
        buffer = (data_max - data_min) * 0.1
        ax.set_ylim(data_min - buffer, data_max + buffer)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    # Optionally show radial circle labels
    if show_radial_labels:
        n_circles = 4
        min_ylim, max_ylim = ax.get_ylim()
        ticks = np.linspace(min_ylim, max_ylim, n_circles + 1)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.1f}" for tick in ticks], fontsize=9)
    else:
        ax.set_yticklabels([])

    # Plot main line
    ax.plot(angles, means, color=color, linewidth=4, label='Mean')

    # Optional std shading
    if show_std and std_upper is not None:
        n_steps = 10
        for i in range(n_steps):
            alpha = 0.1 # * (1 - i / n_steps)
            lower = std_lower + (np.array(means) - std_lower) * (i / n_steps)
            upper = std_upper - (std_upper - np.array(means)) * (i / n_steps)
            ax.fill_between(angles, lower, upper, color=color, alpha=alpha)

    # Title
    ax.set_title(title, size=13, pad=30)

    # Group labels and separators
    if group_labels:
        group_labels_dict = {}
        separators = []
        current_idx = 0
        for group_name, group_vars in groups.items():
            count = len(group_vars)
            group_indices = list(range(current_idx, current_idx + count))
            group_angles = [angles[i] for i in group_indices]
            middle_angle = np.mean(group_angles)
            group_labels_dict[group_name] = middle_angle
            end_idx = current_idx + count
            if end_idx < len(all_vars):
                separators.append(angles[end_idx])
            current_idx = end_idx

        # Draw separators
        for sep in separators:
            ax.plot([sep, sep], [0, max_val], color='black', linewidth=2.5, linestyle='--')

        # Draw group headings
        for group_name, angle in group_labels_dict.items():
            ax.text(angle, max_val * 1.15, group_name,
                    ha='center', va='center', fontsize=label_size,
                    fontweight='bold', rotation=np.rad2deg(angle), rotation_mode='anchor')


        