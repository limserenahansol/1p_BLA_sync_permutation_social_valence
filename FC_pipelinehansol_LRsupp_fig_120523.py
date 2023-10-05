from suite2p.extraction import dcnv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

from matplotlib import use
import core.util as cu

np.random.seed(42)  # add this to all scripts to make results reproducible

use('qt5agg')
#%%
def flatten_list(l):
    # This function is used to flatten a list of lists
    return [item for sublist in l for item in sublist]

def get_psth(data,event_train,twin):
    # This function is used to generate PSTH for a given data and event train
    # data: 2D array, each row is a time point, each column is a neuron
    # event_train: 1D array, each element is the time point of an event
    # twin: int, the pre and post event time window (in frames) for PSTH
    psth_collection = []
    for i in np.where(np.diff(event_train)>0)[0]:
        st = np.maximum(i-twin,0)
        ed = np.minimum(i+twin,data.shape[0])
        d = np.pad(data[st:ed,:],((st-i+twin,i+twin-ed),(0,0)),mode='constant',constant_values=np.nan).T
        psth_collection.append(d)
    psth_collection = np.array(psth_collection)
    pre_FR = np.nanmean(psth_collection[:,:,:(twin+1)],axis=2)
    post_FR = np.nanmean(psth_collection[:,:,(twin+1):],axis=2)
    respScore = (post_FR-pre_FR)/(post_FR+pre_FR)  # Event (freezing/shock) response score for each neuron
    return psth_collection, respScore

def on_press(event):
    # Something for the interactive plot
    if event.key == 'escape':
        plt.close()

def folder_path_finder(root_dir, genotype_pattern, exp_pattern, fn):
    # This function is used to find the folder path of the h5 file
    fdir_list = []
    if genotype_pattern:
        root_dir_folders = [i for i in os.listdir(root_dir) if genotype_pattern.lower() in i.lower()]
    else:
        root_dir_folders = os.listdir(root_dir)

    for i_rf in root_dir_folders:
        i_rf_dir = os.path.join(root_dir, i_rf)
        exp_folder_name = [i for i in os.listdir(i_rf_dir) if exp_pattern in i]
        if len(exp_folder_name) > 0:
            i_exp_dir = os.path.join(i_rf_dir, exp_folder_name[0])
            for session_folder_name in os.listdir(i_exp_dir):
                session_folder_dir = os.path.join(i_exp_dir, session_folder_name)
                if os.path.isdir(session_folder_dir) and 'session_' in session_folder_name:
                    if os.path.isfile(os.path.join(session_folder_dir, fn)):
                        fdir_list.append(os.path.join(session_folder_dir, fn))

    return fdir_list

#%% Define anaysis parameters: PLEASE CHANGE THESE PARAMETERS TO FIT YOUR DATA!
# File name
root_dir = 'J:\\Hansol_Yue\\experiment data\\'  # Replace with the root directory of your data
# genotype_pattern = 'etv1_1'  # Replace with the genotype of your data (etv/rspo/lypd), if left empty (''), all folders will be included   # Commented by YZ 2023-08-17
# exp_pattern = 'FC_2'  # Do not change this, it allows the following code to sniff for the FC experiment folder   # Commented by YZ 2023-08-17
genotype_pattern = 'lypd'  # Replace with the genotype of your data (etv/rspo/lypd), if left empty (''), all folders will be included  # Added by YZ 2023-08-17
exp_pattern = 'FC_1'  # Do not change this, it allows the following code to sniff for the FC experiment folder   # Added by YZ 2023-08-17
fn = 'preprocessed_20230313162341.h5' # H5 file name
### rspo2 we dont have FC1
# Spike detection parameter
ca_tau = .75 # Replace with the real tau time of your calcium indicator
ca_fs = 10  # Replace with the real frame rate of the corresponding recording

# PSTH parameters
twin = 20 # Time window for PSTH, in frames, e.g. twin = 30, the PSTH will include the calcium/spike activities in 30 frames before and after the onset of shock/freezing event

# Threshold for determine whether a neuron is positively or negatively correlated with shock/freezing

# Pick an appropriate value based on the distribution of shock response score, the neurons with shock response score
# above this threshold will be considered as positively correlated with shock, and the neurons with shock response
# score below -shock_corr_thre will be considered as negatively correlated with shock
shock_corr_thre = 0.75

# Pick an appropriate value based on the distribution of freezing response score, the neurons with freezing response score
# above this threshold will be considered as positively correlated with freezing, and the neurons with freezing response
# score below -freezing_corr_thre will be considered as negatively correlated with freezing
freezing_corr_thre = 0.4
#%% Obtain all folder paths
fdir_list = folder_path_finder(root_dir, genotype_pattern, exp_pattern, fn)
# print(fdir_list)  # uncomment this line to see the list of all folder paths

#### Extract genotype, shock, freezing, spike, and calcium data from each file in all the folders listed in fdir_list
# folder_list = [
#     ('Etv1_1', 'social_1', '1'), ('Etv1_1', 'social_1', '2'),
#     ('Etv1_2', 'social_1', '1'), ('Etv1_3', 'social_1', '1'), ('Etv1_3', 'social_2', '1'), ('Etv1_3', 'social_2', '2'),
#     ('Etv1_4', 'social_1', '1'), ('Etv1_4', 'social_2', '1'), ('Etv1_5', 'social_1', '3'), ('Etv1_5', 'social_1', '4'),
#     ('Etv1_6', 'social_1', '1'), ('Etv1_6', 'social_1', '2'), ('Etv1_7', 'social_1', '3'), ('Etv1_7', 'social_1', '4')
#     # ('Rspo2_3','social_1','1'),
# ]
# for im, ie, isess in folder_list:
#     fn = 'J:\\Hansol_Yue\experiment data\{}\{}\session_{}\preprocessed_20230310215144.h5'.format(im, ie, isess)
#     behav_df, ca_df = cu.load_data_file(fn)
# folder_list = [
#    ('Lypd1_1','FC_1','1'),('Lypd1_7','FC_1','1'),
#    ('Lypd1_15','FC_1','2'),('Etv1_4', 'FC_1', '1'),('Etv1_5', 'FC_1', '1'),('Etv1_7', 'FC_1', '2')]
#
# FC_dset = []
# for im, ie, isess in folder_list:
#     fn = 'J:\\Hansol_Yue\experiment data\{}\{}\session_{}\preprocessed_20230313162341.h5'.format(im, ie, isess)
#     behav_df, ca_df = cu.load_data_file(fn)
#     iter_shock_train = behav_df['Floor shock active'].to_numpy()
#     iter_freezing_train = behav_df['Freezing'].to_numpy()
#     iter_ca_train = ca_df.to_numpy()
#     iter_spike_train = dcnv.oasis(F=iter_ca_train.T, batch_size=1, tau=ca_tau, fs=ca_fs).T
#     FC_dset.append([i_genotype, iter_shock_train, iter_freezing_train, iter_spike_train, iter_ca_train])

FC_dset = []
for i_dir in fdir_list:
    if "etv1_19" not in i_dir.lower():
        i_genotype = i_dir.split(exp_pattern)[0].split('\\')[-2]
        behav_df, ca_df = cu.load_data_file(i_dir)
        iter_shock_train = behav_df['Floor shock active'].to_numpy()
        iter_freezing_train = behav_df['Freezing'].to_numpy()
        iter_ca_train = ca_df.to_numpy()
        iter_spike_train = dcnv.oasis(F=iter_ca_train.T, batch_size=1, tau=ca_tau, fs=ca_fs).T
        FC_dset.append([i_genotype, iter_shock_train, iter_freezing_train, iter_spike_train, iter_ca_train])

#%% Plot the shock response score sorted spike traces and the histogram of the shock response score
fig = plt.figure(1)
# Press 'escape' to close the figure and exit the for loop
fig.canvas.mpl_connect('key_press_event', on_press)
for idx in range(len(FC_dset)):
    iter_spike_train = FC_dset[idx][3]
    iter_shock_train = FC_dset[idx][1]
    print(np.unique(iter_shock_train))
    shock_FR = np.nanmean(iter_spike_train[iter_shock_train>0,:],axis=0)
    noshock_FR = np.nanmean(iter_spike_train[iter_shock_train==0,:],axis=0)
    shockScr = (shock_FR-noshock_FR)/(shock_FR+noshock_FR)
    shockScr[np.isnan(shockScr)] = 0
    shockScrSortIdx = np.argsort(shockScr)[::-1]
    # create colormap for freezingScr
    cmap = plt.get_cmap('cool')
    cmap = cmap((shockScr+1)/2)
    if plt.fignum_exists(1):
        fig.clf()
        plt.subplot(121)
        for i in range(iter_spike_train.shape[1]):
            plt.plot(cu.rnorm(iter_spike_train[:,shockScrSortIdx[i]])+i, color=cmap[shockScrSortIdx[i],:3])
        plt.imshow(iter_shock_train[None,:], extent=[0, iter_spike_train.shape[0], 0, iter_spike_train.shape[1]], aspect='auto',cmap='binary',vmin=-.5, vmax=1.5)
        plt.title('Dark: shock, Light: no shock')

        plt.subplot(122)
        plt.hist(shockScr,twin, color='k')
        plt.title(idx)
        plt.draw()
        plt.show()
        plt.waitforbuttonpress()
    else:
        break

#%% Plots the freezing response score sorted spike traces and the histogram of the freezing response score
fig = plt.figure(1)
# Press 'escape' to close the figure and exit the for loop
fig.canvas.mpl_connect('key_press_event', on_press)
for idx in range(len(FC_dset)):
    iter_spike_train = FC_dset[idx][3]
    iter_freezing_train = FC_dset[idx][2]
    # print(np.unique(iter_freezing_train))
    freezing_FR = np.nanmean(iter_spike_train[iter_freezing_train>0,:],axis=0)
    nofreezing_FR = np.nanmean(iter_spike_train[iter_freezing_train==0,:],axis=0)
    freezingScr = (freezing_FR-nofreezing_FR)/(freezing_FR+nofreezing_FR)
    freezingScr[np.isnan(freezingScr)] = 0
    freezingScrSortIdx = np.argsort(freezingScr)[::-1]

    # create colormap for freezingScr
    cmap = plt.get_cmap('cool')
    cmap = cmap((freezingScr+1)/2)
    # Press 'escape' to close the figure and exit the for loop, press other keys to switch to the next experiment result
    if plt.fignum_exists(1):
        fig.clf()
        plt.subplot(121)
        for i in range(iter_spike_train.shape[1]):
            plt.plot(cu.rnorm(iter_spike_train[:,freezingScrSortIdx[i]])+i, color=cmap[freezingScrSortIdx[i],:3])
        plt.imshow(iter_freezing_train[None,:], extent=[0, iter_spike_train.shape[0], 0, iter_spike_train.shape[1]], aspect='auto',cmap='binary',vmin=-.5, vmax=1.5)
        plt.title('Dark: freezing, Light: no freezing')

        plt.subplot(122)
        plt.hist(freezingScr,twin, color='k')
        plt.title(idx)
        plt.draw()
        plt.show()
        plt.waitforbuttonpress()
    else:
        break

#%% (Uncomment this to view the plots) Compute the freezing score based on the firing rate during the first freezing period
fig = plt.figure(1)
# Press 'escape' to close the figure and exit the for loop
fig.canvas.mpl_connect('key_press_event', on_press)
for idx in range(len(FC_dset)):
    iter_spike_train = FC_dset[idx][3]
    iter_freezing_train = FC_dset[idx][2]
    if np.nansum(iter_freezing_train)>0:
        first_freezing_onset_idx = np.where(np.diff(iter_freezing_train)>0)[0][0]
        first_freezing_offset_idx = np.where(np.diff(iter_freezing_train)<0)[0][0]
        first_freezing_duration = first_freezing_offset_idx - first_freezing_onset_idx
        first_freezing_FR = np.nanmean(iter_spike_train[first_freezing_onset_idx:first_freezing_offset_idx,:],axis=0)
        nofreezing_FR = np.nanmean(iter_spike_train[iter_freezing_train==0,:],axis=0)
        first_freezing_score = (first_freezing_FR - nofreezing_FR) / (first_freezing_FR + nofreezing_FR)

        first_freezing_score[np.isnan(first_freezing_score)] = 0
        freezingScrSortIdx = np.argsort(first_freezing_score)[::-1]
        # create colormap for first_freezing_score
        cmap = plt.get_cmap('cool')
        cmap = cmap((first_freezing_score + 1) / 2)
        if plt.fignum_exists(1):
            fig.clf()
            plt.subplot(121)
            for i in range(iter_spike_train.shape[1]):
                plt.plot(cu.rnorm(iter_spike_train[:, freezingScrSortIdx[i]]) + i, color=cmap[freezingScrSortIdx[i], :3])

            plt.imshow(iter_freezing_train[None, :], extent=[0, iter_spike_train.shape[0], 0, iter_spike_train.shape[1]],
                       aspect='auto', cmap='gray', vmin=-.5, vmax=1.5)
            plt.subplot(122)
            plt.hist(first_freezing_score, twin, color='k')
            plt.title(idx)
            plt.draw()
            plt.show()
            plt.waitforbuttonpress()
        else:
            break

#%% Pool the PSTHs and response scores of all the experiment sessions
shock_PSTH = []
shock_respScore = []
shock_ca = []

freezing_PSTH = []
freezing_ca = []
freezing_respScore = []

for i_data in FC_dset:  # i_data = [i_genotype, i_shock_train, i_freezing_train, i_spike_train, i_Ca_train]
    if any(i_data[1]>0):
        i_shock_PSTH, i_shock_respScore = get_psth(i_data[3],i_data[1],twin)  # i_shock_respScore = (post_FR - pre_FR) / (post_FR + pre_FR)
        i_shock_Ca, _ = get_psth(i_data[4],i_data[1],twin)   # This line is same as above but to get the Ca signal

        shock_PSTH.append([i_shock_PSTH[:,i,:] for i in range(i_shock_PSTH.shape[1])])
        shock_ca.append([i_shock_Ca[:,i,:] for i in range(i_shock_Ca.shape[1])])
        shock_respScore.append([i_shock_respScore[:,i] for i in range(i_shock_respScore.shape[1])])
    else:
        print("No shock in this session")

    if any(i_data[2] > 0):
        i_freezing_PSTH, i_freezing_respScore = get_psth(i_data[3], i_data[2], twin)  # The same as above but for freezing
        i_freezing_Ca, _ = get_psth(i_data[4], i_data[2], twin)
    else:
        i_freezing_PSTH = np.zeros(i_shock_PSTH.shape)
        i_freezing_respScore = np.zeros((i_shock_respScore.shape)) - 100
        i_freezing_Ca = np.zeros(i_shock_Ca.shape)
        print("No freezing in this session")

    freezing_PSTH.append([i_freezing_PSTH[:, i, :] for i in range(i_freezing_PSTH.shape[1])])
    freezing_ca.append([i_freezing_Ca[:, i, :] for i in range(i_freezing_Ca.shape[1])])
    freezing_respScore.append([i_freezing_respScore[:, i] for i in range(i_freezing_respScore.shape[1])])
#%% Find shock responsive cells
# The f_.... is the flattened version of the list which concatenate all the sessions together
f_shock_respScore = flatten_list(shock_respScore)  # i_shock_respScore = (post_FR - pre_FR) / (post_FR + pre_FR)
f_shock_PSTH = flatten_list(shock_PSTH)
f_shock_ca = flatten_list(shock_ca)
f_shock_score = [np.nanmean(i,axis=0) for i in f_shock_respScore]

# Sort the cells based on the shock response score
sort_idx = np.argsort(f_shock_score)
sorted_shock_PSTH = np.vstack([f_shock_PSTH[i] for i in sort_idx])
sorted_shock_PSTH = sorted_shock_PSTH/sorted_shock_PSTH.max(axis=1)[:,None]
sorted_shock_ca = np.vstack([f_shock_ca[i] for i in sort_idx])

# Create a boolean array to indicate the pro/anti/neutral shock responsive cells
arr_shock_score = np.vstack([f_shock_score[i]*np.ones((len(f_shock_PSTH[i]),1)) for i in sort_idx])
arr_shock_score[np.isnan(arr_shock_score)] = 0
pro_cell = arr_shock_score > shock_corr_thre
anti_cell = arr_shock_score < -shock_corr_thre
non_cell = abs(arr_shock_score) < shock_corr_thre

#%%
plt.figure(1)
plt.clf()

plt.subplot(2,2,1)
plt.hist(f_shock_score,twin,weights=np.ones(len(f_shock_score)) / len(f_shock_score), color=(.4,.4,.4))
plt.ylabel('Frequency')
plt.xlabel('Shock response score [(postFR-preFR)/(postFR+preFR)]')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.subplot(2,2,2)
plt.pie((np.sum(pro_cell), np.sum(anti_cell), np.sum(non_cell),),
        labels=('pos corr', 'neg corr', 'no corr'),
        colors=[(1,.7,.2), (0.2,.7,1), (.6,.6,.6)], autopct='%1.1f%%', startangle=90)

plt.subplot(2,2,3)
plt.imshow(sorted_shock_PSTH[::-1,:], aspect='auto', cmap='Greens')  ## Reds for ETV
plt.title('PSTH plots for all neurons sorted by shock response score')

plt.subplot(2,2,4)
for i in range(sorted_shock_ca.shape[0]):
    temp = (sorted_shock_ca[i, :]-sorted_shock_ca[i, :].mean())/np.nanstd(sorted_shock_ca[i, :]) # z-score normalized by the first 10 frames (baseline)
    if pro_cell[i]:
        plt.plot(temp+10, c=[1, 0, 0, 2/np.sum(pro_cell)])
    elif anti_cell[i]:
        plt.plot(temp-10, c=[0, 0, 1, 2/np.sum(anti_cell)])
    else:
        plt.plot(temp, c=[0,0,0, 2/np.sum(non_cell)])

#%% Generate the same plots for freezing
f_freezing_respScore = flatten_list(freezing_respScore)  # i_shock_respScore = (post_FR - pre_FR) / (post_FR + pre_FR)
f_freezing_PSTH = flatten_list(freezing_PSTH)
f_freezing_ca = flatten_list(freezing_ca)
f_freezing_score = [np.nanmean(i,axis=0) for i in f_freezing_respScore]
sort_idx = np.argsort(f_freezing_score)
sorted_freezing_PSTH = np.vstack([f_freezing_PSTH[i] for i in sort_idx])
sorted_freezing_PSTH = sorted_freezing_PSTH/sorted_freezing_PSTH.max(axis=1)[:,None]
sorted_freezing_ca = np.vstack([f_freezing_ca[i] for i in sort_idx])

arr_freezing_score = np.vstack([f_freezing_score[i]*np.ones((len(f_freezing_PSTH[i]),1)) for i in sort_idx])
arr_freezing_score[np.isnan(arr_freezing_score)] = 0
pro_cell = arr_freezing_score > freezing_corr_thre
anti_cell = arr_freezing_score < -freezing_corr_thre
non_cell = abs(arr_freezing_score) < freezing_corr_thre

plt.figure(2)
plt.clf()

plt.subplot(2,2,1)
plt.hist(f_freezing_score,twin,weights=np.ones(len(f_freezing_score)) / len(f_freezing_score), color=(.4,.4,.4))
plt.ylabel('Frequency')
plt.xlabel('Freezing response score [(postFR-preFR)/(postFR+preFR)]')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.subplot(2,2,2)
plt.pie((np.sum(pro_cell), np.sum(anti_cell), np.sum(non_cell),),
        labels=('pos corr', 'neg corr', 'no corr'),
        colors=[(1,.7,.2), (0.2,.7,1), (.6,.6,.6)], autopct='%1.1f%%', startangle=90)

plt.subplot(2,2,3)
plt.imshow(sorted_freezing_PSTH[::-1,:], aspect='auto', cmap='Greens')
plt.title('PSTH plots for all neurons sorted by freezing response score')

plt.subplot(2,2,4)
for i in range(sorted_freezing_ca.shape[0]):
    temp = (sorted_freezing_ca[i, :]-sorted_freezing_ca[i, :10].mean())/np.nanstd(sorted_freezing_ca[i, :])
    if pro_cell[i]:
        plt.plot(temp+10, c=[1, 0, 0, 2/np.sum(pro_cell)])
    elif anti_cell[i]:
        plt.plot(temp-10, c=[0, 0, 1, 2/np.sum(anti_cell)])
    else:
        plt.plot(temp, c=[0,0,0, np.maximum(2/np.sum(non_cell),0.005)])