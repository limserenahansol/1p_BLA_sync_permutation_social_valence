from suite2p.extraction import dcnv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from matplotlib import use
use('qt5agg')

import core.util as cu

#%% Param settings

# Spike detection parameter
ca_tau = 1 # Replace with the real tau time of your calcium indicator
ca_fs = 10  # Replace with the real frame rate of the corresponding recording

# Distance preference score related parameter
max_radius = 10
min_radius = 20
n_dist_pix = 100
# spatial heatmap parameter
nx,ny = 50,50
heatmap_counter_thre = 4
folder_lists = [[
    # ('Etv1_1', 'social_1', '1'), ('Etv1_1', 'social_1', '2'),
    # ('Etv1_2', 'social_1', '1'), ('Etv1_3', 'social_1', '1'),##
    # ('Etv1_3', 'social_2', '1'), ('Etv1_3', 'social_2', '2'),
    # ('Etv1_4', 'social_1', '1'), ('Etv1_4', 'social_2', '1'),
    # #('Etv1_5', 'social_1', '3'), ('Etv1_5', 'social_1', '4'),## cnmfe did not detect well neurons but more artfect
    # #('Etv1_6', 'social_1', '1'), #('Etv1_6', 'social_1', '2'),###
    # ('Etv1_7', 'social_1', '3'), ('Etv1_7', 'social_1', '4')
    # ]]

        #('Lypd1_5', 'social_1', '1'),
        #('Lypd1_6', 'social_1', '1'),
        #('Lypd1_6', 'social_2', '1'),
        #('Lypd1_6', 'social_2', '1'),
        ('Lypd1_7', 'social_2', '1'),
        ('Lypd1_7', 'social_2', '2'),
        ('Lypd1_7', 'social_2', '3'),
        ('Lypd1_8', 'social_1', '1'),
        #('Lypd1_6', 'social_2', '1'),
        #('Lypd1_6', 'social_2', '2'),
        ('Lypd1_9', 'social_1', '1'),
        ('Lypd1_10', 'social_1', '1'),
        ('Lypd1_10', 'social_1', '2'),
        ('Lypd1_11', 'social_1', '3'),
        ('Lypd1_11', 'social_1', '4')
    ]]
    # ('Rspo2_2', 'social_1', '1'),('Rspo2_2', 'socialFeed_1', '1'),  # 3 chamber social feed together
    #  ('Rspo2_3', 'social_1', '1'),
    # ('Rspo2_4', 'socialFeed_1', '1'), ('Rspo2_4', 'socialFeed_1', '2'), ('Rspo2_4', 'socialNovel_1', '1'),
    # ('Rspo2_5', 'social_1', '1')
    # ]]

#%% Batch processing of all sessions
frplots = []
for k, folder_list in enumerate(folder_lists):
    spatial_heatmap = []
    dist_pref_ratio = []
    behav_loc = []
    spike_trains = []
    frplot = []
    for im, ie, isess in folder_list:
        fn = 'J:\\Hansol_Yue\experiment data\{}\{}\session_{}\preprocessed_20230313153213.h5'.format(im, ie, isess)
        behav_df, ca_df = cu.load_data_file(fn)
        iter_behav_loc = np.vstack([cu.rnorm(behav_df['X center']) * (nx - 1), cu.rnorm(behav_df['Y center']) * (ny - 1)])
        # behav_loc.append(iter_behav_loc)

        # spike detection
        iter_spike_train = dcnv.oasis(F=ca_df.to_numpy().T, batch_size=1, tau=ca_tau, fs=ca_fs).T
        dist = behav_df['Distance to point'].to_numpy()
        dist_bins = np.linspace(0,30, 31)
        dist_bin_cen = dist_bins[:-1] + (dist_bins[1:] - dist_bins[:-1]) / 2
        frInBin = np.zeros((len(dist_bins) - 1, iter_spike_train.shape[1]))

        for i in range(len(dist_bins) - 1):
            frInBin[i, :] = np.nanmean(iter_spike_train[(dist >= dist_bins[i]) & (dist < dist_bins[i + 1]), :], axis=0)
        frplot.append(frInBin)
    # spike_train.append(iter_spike_train)
    #
    # # heatmap computation
    # iter_spatial_hmap = np.zeros((nx, ny, iter_spike_train.shape[1]))
    # iter_spatial_counter = iter_spatial_hmap.copy()
    # discrete_location_index = np.floor(iter_behav_loc).astype(int)
    # discrete_location_index[np.isnan(iter_behav_loc)] = 0
    # for ii, ixy in enumerate(discrete_location_index.T):
    #     iter_spatial_hmap[ixy[1], ixy[0], :] += iter_spike_train[ii, :] > 2
    #     iter_spatial_counter[ixy[1], ixy[0], :] += 1
    # iter_spatial_hmap[0, 0, :] = 0
    # iter_spatial_hmap /= iter_spatial_counter
    # iter_spatial_hmap[iter_spatial_counter < heatmap_counter_thre] = 0
    # spatial_heatmap.append(iter_spatial_hmap)
    #
    # # distance preference score computation
    # loc_dist = np.sum(iter_behav_loc ** 2, axis=0) ** .5
    # nn_loc_dist = (loc_dist[~np.isnan(loc_dist)]-min_radius)/(max_radius-min_radius)
    # nn_spks = iter_spike_train[~np.isnan(loc_dist), :]
    # iter_dist_pref_ratio = nn_loc_dist.dot(nn_spks)/(nn_spks.mean(axis=0)[...,np.newaxis]*nn_loc_dist).sum(axis=1)
    # dist_pref_ratio.append(iter_dist_pref_ratio)
    frplot = np.concatenate(frplot,axis=1)
    frplot[np.isnan(frplot)] = 0
    frplot /= frplot.max(axis=0)
    frplots.append(frplot)
#%%

for o in range(3):
    plt.clf()
    ax = plt.subplot(1,2,1)
    idx = np.argsort(np.argmax(frplots[o],axis=0))
    plt.imshow(frplots[o][:,idx].T,aspect='auto',interpolation='nearest',cmap='GnBu', extent=[0,30,0,frplots[o].shape[1]])
    plt.xlabel('Distance from point (cm)')
    plt.ylabel('Neuron #')

    ax = plt.subplot(1,2,2)
    for i in range(frplots[o].shape[1]):
        plt.plot(np.linspace(0,30,30),frplots[o][:,idx[::-1][i]]*2+i,'k')

    plt.xlabel('Distance from point (cm)')
    #plt.ylabel('Firing rate')
    ax.set_ylim([0,frplots[o].shape[1]+2])
    plt.waitforbuttonpress()

#%%
plt.clf()
pidx = 1
sort_idx = np.argsort(iter_spike_train[:,pidx])
plt.scatter(iter_behav_loc[0,sort_idx],iter_behav_loc[1,sort_idx],c=iter_spike_train[sort_idx,pidx],s=iter_spike_train[sort_idx,pidx]*10)
# %% FR plotS
plt.clf()
max_idx = []
for v in frplots:
    max_idx.append(np.linspace(0,30,30)[np.argmax(v,axis=0)])

plt.violinplot(max_idx, positions=[0,1,2], bw_method=0.2, showmeans=True, showextrema=True, showmedians=False)
plt.xticks([0,1,2])
plt.gca().set_xticklabels(['Etv1','Lypd1','Rspo2'])
plt.ylabel('Peak firing distance (cm)')
#%%
temp = [(np.mean(i<10),np.mean(i>20),np.mean((i>10)&(i<20))) for i in max_idx]
labels = 'prococial','antisocial','neutral'
explode = (0.1, 0.1, 0)
colors = ( "magenta", "brown",
          "grey")
sizes = [15, 30, 45, 10]
plt.clf()
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.pie(temp[i],autopct='%1.1f%%',startangle=90,colors=colors)
    plt.legend(labels, loc="best", bbox_to_anchor=(1, 2))
    plt.gca().set_title(['Etv1','Lypd1','Rspo2'][i],fontsize=16)
#%% Spatial heatmap plot

# This part of the code is for indexing, just ignore
spike_train_indice_lists = []
for o in spike_trains:
    spike_train_indices = []
    session_idx = 0
    for i in o:
        spike_train_indices.append(np.vstack((np.ones(i.shape[1])*session_idx,np.arange(i.shape[1]))))
        session_idx += 1
    spike_train_indices = np.hstack(spike_train_indices).astype(int).T
    spike_train_indice_lists.append(spike_train_indices)

sorted_max_idx = [np.argsort(i) for i in max_idx]  # sorted the neurons by their peak firing rate distance bin (nearest to farthest)
sorted_spike_train_indice_lists = [spike_train_indice_lists[i][sorted_max_idx[i]] for i in range(1)]
group_idx = 0
for show_idx in np.arange(200,220):
    session_trace_idx = sorted_spike_train_indice_lists[group_idx][show_idx]
    selected_spike_train = spike_trains[group_idx][session_trace_idx[0]][:,session_trace_idx[1]]
    selected_behav_loc = behav_locs[group_idx][session_trace_idx[0]]
    sorted_scatter_plot_order = np.argsort(selected_spike_train)
    temp = dist_locs[group_idx][session_trace_idx[0]]
    peak_dist = np.sort(max_idx[group_idx])[show_idx]
    frInBin = np.zeros(len(dist_bins)-1)
    for i in range(len(dist_bins) - 1):
        dist_filter = (temp >= dist_bins[i]) & (temp < dist_bins[i + 1])
        frInBin[i] = np.nanmean(selected_spike_train[dist_filter], axis=0)
    fig = plt.figure(3)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(selected_behav_loc[0,:], selected_behav_loc[1,:], '--', color=[0,0,0,.2])
    ax.scatter(selected_behav_loc[0,sorted_scatter_plot_order], selected_behav_loc[1,sorted_scatter_plot_order], c=selected_spike_train[sorted_scatter_plot_order], s=selected_spike_train[sorted_scatter_plot_order] * 10)
    ax.scatter(selected_behav_loc[0,:], selected_behav_loc[1,:], s = 50*(np.abs(temp-peak_dist)<1), c = np.abs(temp-peak_dist)<1, cmap='Reds', vmin=0, vmax=1.5, marker='+')  # show the distance from the peak firing distance
    ax.set_aspect('equal')
    plt.title('Peak distance {}'.format(peak_dist))
    plt.show()
    plt.draw()
    plt.waitforbuttonpress()  # click to continue to the next neuron
