from suite2p.extraction import dcnv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

from matplotlib import use
import core.util as cu

use('qt5agg')


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def get_psth(data, event_train, twin):
    psth_collection = []
    for i in np.where(np.diff(event_train) > 0)[0]:
        st = np.maximum(i - twin, 0)
        ed = np.minimum(i + twin, data.shape[0])
        d = np.pad(data[st:ed, :], ((st - i + twin, i + twin - ed), (0, 0)), mode='constant', constant_values=np.nan).T
        psth_collection.append(d)
    psth_collection = np.array(psth_collection)
    pre_FR = np.nanmean(psth_collection[:, :, :(twin + 1)], axis=2)
    post_FR = np.nanmean(psth_collection[:, :, (twin + 1):], axis=2)
    respScore = (post_FR - pre_FR) / (post_FR + pre_FR)
    return psth_collection, respScore


def on_press(event):
    if event.key == 'escape':
        plt.close()


def paired_folder_structurer(root_dir, genotype_pattern, pattern_pairs, fn):
    pattern_1 = pattern_pairs[0]
    pattern_2 = pattern_pairs[1]
    pattern_1_list = []
    pattern_2_list = []
    if genotype_pattern:
        root_dir_folders = [i for i in os.listdir(root_dir) if genotype_pattern.lower() in i.lower()]
    else:
        root_dir_folders = os.listdir(root_dir)

    for i_rf in root_dir_folders:
        i_rf_dir = os.path.join(root_dir, i_rf)
        if pattern_1 in os.listdir(i_rf_dir) and pattern_2 in os.listdir(i_rf_dir):
            pat_1_dir = os.path.join(i_rf_dir, pattern_1)
            pat_2_dir = os.path.join(i_rf_dir, pattern_2)
            i_pattern_1_list = []
            i_pattern_2_list = []
            for session_folder_name in os.listdir(pat_1_dir):
                session_folder_dir = os.path.join(pat_1_dir, session_folder_name)
                if os.path.isdir(session_folder_dir) and "session_" in session_folder_name:
                    if os.path.isfile(os.path.join(session_folder_dir, fn)):
                        i_pattern_1_list.append(session_folder_dir)

            for session_folder_name in os.listdir(pat_2_dir):
                session_folder_dir = os.path.join(pat_2_dir, session_folder_name)
                if os.path.isdir(session_folder_dir) and "session_" in session_folder_name:
                    if os.path.isfile(os.path.join(session_folder_dir, fn)):
                        i_pattern_2_list.append(session_folder_dir)

            if len(i_pattern_1_list)>0 and len(i_pattern_2_list)>0:
                for i in range(len(i_pattern_1_list)):
                    pattern_1_list.append(i_pattern_1_list[i])

                for i in range(len(i_pattern_2_list)):
                    pattern_2_list.append(i_pattern_2_list[i])

    return pattern_1_list, pattern_2_list

def group_by_animal(pattern_list):
    animal_list = {}
    for i in pattern_list:
        animal_name = i.split('\\')[-3]
        if animal_name not in animal_list.keys():
            animal_list[animal_name] = [i]
        else:
            animal_list[animal_name].append(i)
    return animal_list


# %% Define anaysis parameters: PLEASE CHANGE THESE PARAMETERS TO FIT YOUR DATA!
# File name
root_dir = 'J:\\Hansol_Yue\\experiment data\\'  # Replace with the root directory of your data
genotype_pattern = 'etv'  # Replace with the genotype of your data (etv/rspo/lypd), if left empty (''), all folders will be included
exp_pattern_pair = ('FC_1', 'FC_2')  # Do not change this, it allows the following code to sniff for the FC experiment folder
fn = 'preprocessed_20230313162341.h5'  # H5 file name

# Spike detection parameter
ca_tau = 0.5  # Replace with the real tau time of your calcium indicator
ca_fs = 5  # Replace with the real frame rate of the corresponding recording

# PSTH parameters
twin = 20  # Time window for PSTH, in frames, e.g. twin = 30, the PSTH will include the calcium/spike activities in 30 frames before and after the onset of shock/freezing event

# Threshold for determine whether a neuron is positively or negatively correlated with shock/freezing

# Pick an appropriate value based on the distribution of shock response score, the neurons with shock response score
# above this threshold will be considered as positively correlated with shock, and the neurons with shock response
# score below -shock_corr_thre will be considered as negatively correlated with shock
shock_corr_thre = 0.85

# Pick an appropriate value based on the distribution of freezing response score, the neurons with freezing response score
# above this threshold will be considered as positively correlated with freezing, and the neurons with freezing response
# score below -freezing_corr_thre will be considered as negatively correlated with freezing
freezing_corr_thre = 0.5

#%% Get the list of folders for the two patterns
pattern_1_list, pattern_2_list = paired_folder_structurer(root_dir, genotype_pattern, exp_pattern_pair, fn)
pattern_1_list = group_by_animal(pattern_1_list)
pattern_2_list = group_by_animal(pattern_2_list)

#### Extract genotype, shock, freezing, spike, and calcium data from each file in all the folders listed in fdir_list
FC_1_dset = {}
for i_animal_id, i_dir_list in pattern_1_list.items():
    FC_1_dset[i_animal_id] = []
    for i_dir in i_dir_list:
        i_genotype = i_dir.split('\\')[-3]
        behav_df, ca_df = cu.load_data_file(os.path.join(i_dir,fn))
        iter_shock_train = behav_df['Floor shock active'].to_numpy()
        iter_freezing_train = behav_df['Freezing'].to_numpy()
        iter_ca_train = ca_df.to_numpy()
        iter_ca_time = ca_df.index.to_numpy()
        iter_spike_train = dcnv.oasis(F=iter_ca_train.T, batch_size=1, tau=ca_tau, fs=ca_fs).T
        FC_1_dset[i_animal_id].append([i_genotype, iter_shock_train, iter_freezing_train, iter_spike_train, iter_ca_train, iter_ca_time])

FC_2_dset = {}
for i_animal_id, i_dir_list in pattern_2_list.items():
    FC_2_dset[i_animal_id] = []
    for i_dir in i_dir_list:
        i_genotype = i_dir.split('\\')[-3]
        behav_df, ca_df = cu.load_data_file(os.path.join(i_dir,fn))
        iter_shock_train = behav_df['Floor shock active'].to_numpy()
        iter_freezing_train = behav_df['Freezing'].to_numpy()
        iter_ca_train = ca_df.to_numpy()
        iter_ca_time = ca_df.index.to_numpy()
        iter_spike_train = dcnv.oasis(F=iter_ca_train.T, batch_size=1, tau=ca_tau, fs=ca_fs).T
        FC_2_dset[i_animal_id].append([i_genotype, iter_shock_train, iter_freezing_train, iter_spike_train, iter_ca_train,iter_ca_time])



#%% Compute freezing frequency for FC_2_dset

FC_2_freezing_freq = {}
for i_animal_id, i_dset in FC_2_dset.items():
    FC_2_freezing_freq[i_animal_id] = []
    for i in i_dset:
        # iter_freezing_freq = np.nansum(i[2])/len(i[2]) # YZ_120523
        i_stop = np.where(i[5]>180)[0][0]
        iter_freezing_freq = np.nansum(i[2][:i_stop])/i_stop # YZ_120523
        FC_2_freezing_freq[i_animal_id].append(iter_freezing_freq)
    FC_2_freezing_freq[i_animal_id] = np.nanmean(np.array(FC_2_freezing_freq[i_animal_id]))

#%% compute pro/anti-shock neuron percentage for FC_1_dset
FC_1_neuron_percentage = {}
for i_animal_id, i_dset in FC_1_dset.items():
    pro_shock_neuron_count = 0
    anti_shock_neuron_count = 0
    total_neuron_count = 0
    for i in i_dset:
        i_shock_PSTH, i_shock_respScore = get_psth(i[3],i[1],twin)
        pro_shock_neuron_count += np.sum(i_shock_respScore > shock_corr_thre)
        anti_shock_neuron_count += np.sum(i_shock_respScore < -shock_corr_thre)
        total_neuron_count += i_shock_respScore.shape[0]
    FC_1_neuron_percentage[i_animal_id] = [pro_shock_neuron_count/total_neuron_count, anti_shock_neuron_count/total_neuron_count]

#%% show the pro/anti-shock neuron percentage vs freezing freq for each animal
freezeFreq_ShockNeuronPercentage_pair = []
for k in FC_1_neuron_percentage.keys():
    freezeFreq_ShockNeuronPercentage_pair.append([FC_2_freezing_freq[k], *FC_1_neuron_percentage[k]])

freezeFreq_ShockNeuronPercentage_pair = np.array(freezeFreq_ShockNeuronPercentage_pair)
from scipy import stats

fig = plt.figure(1)
fig.clf()
ax1 = fig.add_subplot(121)
ax1.scatter(freezeFreq_ShockNeuronPercentage_pair[:,0], freezeFreq_ShockNeuronPercentage_pair[:,1], c='r', label='Pro-shock')
slope, intercept, r_value, p_value, std_err = stats.linregress(freezeFreq_ShockNeuronPercentage_pair[:,0], freezeFreq_ShockNeuronPercentage_pair[:,1])
linreg_model_func = lambda x: slope*x + intercept
linreg_model_1 = np.array(list(map(linreg_model_func, freezeFreq_ShockNeuronPercentage_pair[:,0])))
ax1.plot(freezeFreq_ShockNeuronPercentage_pair[:,0], linreg_model_1, c='r')
plt.xlabel('Freezing frequency')
plt.ylabel('Pro-shock neuron percentage')
plt.title('r^2 = {:.1f}, p = {:.1f}'.format(r_value**2, p_value))

ax2 = fig.add_subplot(122)
ax2.scatter(freezeFreq_ShockNeuronPercentage_pair[:,0], freezeFreq_ShockNeuronPercentage_pair[:,2], c='b', label='Anti-shock')
slope, intercept, r_value, p_value, std_err = stats.linregress(freezeFreq_ShockNeuronPercentage_pair[:,0], freezeFreq_ShockNeuronPercentage_pair[:,2])
#a=stats.pearsonr(freezeFreq_ShockNeuronPercentage_pair[:,0], freezeFreq_ShockNeuronPercentage_pair[:,2])[0]
linreg_model_func = lambda x: slope*x + intercept
linreg_model_2 = np.array(list(map(linreg_model_func, freezeFreq_ShockNeuronPercentage_pair[:,0])))
ax2.plot(freezeFreq_ShockNeuronPercentage_pair[:,0], linreg_model_2, c='b')
plt.xlabel('Freezing frequency')
plt.ylabel('Anti-shock neuron percentage')
plt.title('r^2 = {:.1f}, p = {:.1f}'.format(r_value**2, p_value))
#plt.suptitle(stats.pearsonr(freezeFreq_ShockNeuronPercentage_pair[:,0], freezeFreq_ShockNeuronPercentage_pair[:,2])[0])
plt.show()

#%% Show behavior trace for Etv1_1(low freezing freq) and Etv1_9(high freezing freq)

#low_freeze_freq = FC_2_dset['Etv1_1'][0][2]
#high_freeze_freq = FC_2_dset['Etv1_9'][0][2]

fig = plt.figure(123)
fig.clf()
ax = fig.add_subplot(111)

iterator = 0
plot_order = [0,1,2,3,4]  # for ETV1 [0,1,3,2,4] for lypd : [2,4,1,3,0]
for i,p in zip(FC_2_dset.values(),plot_order):
    itrace = i[0][2]
    itime = i[0][5]
    itrace[np.isnan(itrace)] = 0
    ax.plot(itime,itrace * 0 + 1.5*p, color=[0.6, 0.6, 0.6])
    ax.plot(itime,itrace+1.5*p,'k')
    print(len(itrace))

ax.set_yticks([0,1.5,3,4.5,6])
ax.set_yticklabels([list(FC_2_dset.keys())[i] for i in plot_order])

#%% compair FC_1 baseline (the first 3 minutes activities) with FC_2 baseline
FC_1_baseline = [v[0][3][:np.where(v[0][5]>180)[0][0],:] for k,v in FC_1_dset.items()]
FC_2_baseline = [v[0][3][:np.where(v[0][5]>180)[0][0],:] for k,v in FC_2_dset.items()]
FC_1_baseline_mean = np.hstack([np.nanmean(i,axis=0) for i in FC_1_baseline])
FC_2_baseline_mean = np.hstack([np.nanmean(i,axis=0) for i in FC_2_baseline])
len_difference = len(FC_1_baseline_mean)-len(FC_2_baseline_mean)

FC_1_binarized_baseline = [i > 1 for i in FC_1_baseline]
FC_2_binarized_baseline = [i > 1 for i in FC_2_baseline]
FC_1_binarized_baseline_mean = np.hstack([np.nanmean(i,axis=0) for i in FC_1_binarized_baseline])
FC_2_binarized_baseline_mean = np.hstack([np.nanmean(i,axis=0) for i in FC_2_binarized_baseline])

if len_difference < 0:
    FC_1_baseline_mean = np.hstack([FC_1_baseline_mean,np.nan*np.ones(-len_difference)])
    FC_1_binarized_baseline_mean = np.hstack([FC_1_binarized_baseline_mean,np.nan*np.ones(-len_difference)])
elif len_difference > 0:
    FC_2_baseline_mean = np.hstack([FC_2_baseline_mean, np.nan * np.ones(len_difference)])
    FC_2_binarized_baseline_mean = np.hstack([FC_2_binarized_baseline_mean, np.nan * np.ones(len_difference)])

import pandas as pd
import seaborn as sns

# Convert ndarray to DataFrame
df = pd.DataFrame({'FC_1':FC_1_baseline_mean,
                    'FC_2':FC_2_baseline_mean})
#
df2 = pd.DataFrame({'FC_1':FC_1_binarized_baseline_mean,
                    'FC_2':FC_2_binarized_baseline_mean})
# df3 = pd.DataFrame({'FC_1':FC_1_baseline_mean, 'FC_2':FC_2_baseline_mean})
# fig = plt.figure(2)
# fig.clf()
# Plot using swarmplot
sns.swarmplot(data=df2)

# Set labels and title
plt.xlabel('Data')
plt.ylabel('averaged firing rate')
plt.title('Swarmplot of Two Groups')
