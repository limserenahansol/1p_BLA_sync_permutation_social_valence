from suite2p.extraction import dcnv
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import use
use('qt5agg')

import core.util as cu
## Rspo2 social should be distance point 2!!!!
#%% Param settings

# Spike detection parameter
ca_tau = 1 # Replace with the real tau time of your calcium indicator
ca_fs = 10  # Replace with the real frame rate of the corresponding recording

# Distance preference score related parameter
min_radius = 0   # check the social chamber radius in the video to set this parameter
max_radius = 30  # the maximum distance to the social target in all sessions and all animals
n_dist_bin = 50  # the number of distance bins
n_dist_pix = 100
permutation_times = 1000

# spatial heatmap parameter
nx,ny = 50,50
heatmap_counter_thre = 4

# Please always use the sessions using the same experiment setup
folder_lists = [
    [('Rspo2_2', 'social_1', '1'),('Rspo2_2', 'socialFeed_1', '1'),  # 3 chamber social feed together
     ('Rspo2_3', 'social_1', '1'),
    ('Rspo2_4', 'socialFeed_1', '1'), ('Rspo2_4', 'socialFeed_1', '2'), ('Rspo2_4', 'socialNovel_1', '1'),
    ('Rspo2_5', 'social_1', '1')
    ]]
# folder_lists =[[
#     #('Lypd1_5', 'social_1', '1'),
#     #('Lypd1_6', 'social_1', '1'),
#     #('Lypd1_6', 'social_2', '1'),
#     #('Lypd1_6', 'social_2', '1'),
#     ('Lypd1_7', 'social_2', '1'),
#     ('Lypd1_7', 'social_2', '2'),
#     ('Lypd1_7', 'social_2', '3'),
#     ('Lypd1_8', 'social_1', '1'),
#     #('Lypd1_6', 'social_2', '1'),
#     #('Lypd1_6', 'social_2', '2'),
#     ('Lypd1_9', 'social_1', '1'),
#     ('Lypd1_10', 'social_1', '1'),
#     ('Lypd1_10', 'social_1', '2'),
#     ('Lypd1_11', 'social_1', '3'),
#     ('Lypd1_11', 'social_1', '4')
# ]]
# folder_lists = [[
#     ('Etv1_1', 'social_1', '1'), ('Etv1_1', 'social_1', '2'),
#     ('Etv1_2', 'social_1', '1'), ('Etv1_3', 'social_1', '1'),##
#     ('Etv1_3', 'social_2', '1'), ('Etv1_3', 'social_2', '2'),
#     ('Etv1_4', 'social_1', '1'), ('Etv1_4', 'social_2', '1'),
#     #('Etv1_5', 'social_1', '3'), ('Etv1_5', 'social_1', '4'),## cnmfe did not detect well neurons but more artfect
#     #('Etv1_6', 'social_1', '1'), #('Etv1_6', 'social_1', '2'),###
#     ('Etv1_7', 'social_1', '3'), ('Etv1_7', 'social_1', '4')
#     ]]

# folder_lists = [[
#     ('Etv1_1', 'Feed_1', '1'),
#     ('Etv1_1', 'Feed_1', '2'),
#     ('Etv1_2', 'Feed_1', '1'),
#     ('Etv1_2', 'Feed_1', '2'),
#     ('Etv1_2', 'Feed_2', '1'),  # no food
#     ('Etv1_2', 'Feed_2', '2'),  # no food
#     ('Etv1_3', 'Feed_1', '1'),
#     ('Etv1_3', 'Feed_1', '2'), ('Etv1_3', 'Feed_1', '3'), ('Etv1_4', 'Feed_1', '1'), ('Etv1_5', 'Feed_1', '1'),
#     ('Etv1_5', 'Feed_1', '2'), ('Etv1_5', 'Feed_1', '3'), ('Etv1_5', 'Feed_1', '4'), ('Etv1_6', 'Feed_1', '1'),
#     ('Etv1_6', 'Feed_1', '2'), ('Etv1_7', 'Feed_1', '3'), ('Etv1_7', 'Feed_1', '4'),
#     ]]

# folder_lists = [[('Lypd1_1', 'Feed_1', '1'),
#  ('Lypd1_1', 'Feed_1', '2'),
#  ('Lypd1_2', 'Feed_1', '1'),
#  ('Lypd1_2', 'Feed_1', '2'),  # dont de
#  ('Lypd1_3', 'Feed_1', '1'),
#  ('Lypd1_5', 'Feed_1', '1'),
#  ('Lypd1_5', 'Feed_1', '2'),
#  ('Lypd1_5', 'Feed_1', '3'),
#  ('Lypd1_5', 'Feed_2', '1'),
#  ('Lypd1_5', 'Feed_2', '2'),
#  ('Lypd1_5', 'Feed_2', '3'),
#  ('Lypd1_5', 'Feed_2', '4'),
#  ('Lypd1_6', 'Feed_1', '1'),
#  ('Lypd1_7', 'Feed_1', '1'),
#  # ('Lypd1_7', 'Feed_1', '2'),#dele
#  ('Lypd1_8', 'Feed_1', '2'),
#  ('Lypd1_10', 'Feed_1', '1'),
#  ('Lypd1_10', 'Feed_1', '2'),
#  ('Lypd1_11', 'Feed_1', '3'),
#  ('Lypd1_11', 'Feed_1', '4'),  # 43 del
#
#  ]]
# folder_lists =[[('Rspo2_1', 'Feed_1', '1'), ('Rspo2_1', 'Feed_1', '2'), ('Rspo2_1', 'Feed_1', '3'), ('Rspo2_2', 'Feed_1', '2'),
#  ('Rspo2_5', 'Feed_1', '1'),
#  ('Rspo2_2', 'socialFeed_1', '1'),  # social feed together
#  ('Rspo2_4', 'socialFeed_1', '1'), ('Rspo2_4', 'socialFeed_1', '2')  # social feed together
#  ]]

#%% Batch processing of all sessions
frplots = []
sig_cell_trace_list = []
permutation_times = 1000
# define a function to assign distance bins to spike trains based on the distance of the animal to the social target
def gen_dist_bins(distance, max_radius, min_radius, num_bins):
    bins = np.linspace(min_radius, max_radius, num_bins+1)
    dist_bin_cen = bins[:-1] + (bins[1:] - bins[:-1]) / 2
    dist_bin_idx = np.digitize(distance, bins)
    norm_fac = np.ones_like(dist_bin_idx)
    for i in range(num_bins):
        if np.sum(dist_bin_idx == i) > 0:
            norm_fac[dist_bin_idx == i] = np.maximum(1/np.sum(dist_bin_idx == i),0.01)
    # dist_bin_idx -= dist_bin_idx.min()
    return dist_bin_idx, dist_bin_cen, norm_fac

def firing_rate_per_bin(dist_bins,n_bins,spike_train):
    frInBin = np.zeros((n_bins, spike_train.shape[1]))
    for i in range(n_bins):
        frInBin[i, :] = np.nanmean(spike_train[dist_bins == i, :], axis=0)
    return frInBin

sig_cell_arr = []
for k, folder_list in enumerate(folder_lists):
    spatial_heatmap = []
    dist_pref_ratio = []
    behav_loc = []
    spike_train = []
    frplot = []
    sig_cell_trace = []
    for im, ie, isess in folder_list:
        # fn = 'J:\\Hansol_Yue\experiment data\{}\{}\session_{}\FIXTHISPATHpreprocessed_TEST.h5'.format(im, ie, isess)
        fn = 'J:\\Hansol_Yue\Error\experiment data\{}\{}\session_{}\preprocessed_20230310215144.h5'.format(im, ie, isess)

        behav_df, ca_df = cu.load_data_file(fn)
        iter_behav_loc = np.vstack([cu.rnorm(behav_df['X center']) * (nx - 1), cu.rnorm(behav_df['Y center']) * (ny - 1)])

        # spike detection
        iter_spike_train = dcnv.oasis(F=ca_df.to_numpy().T, batch_size=1, tau=ca_tau, fs=ca_fs).T  # the shape is (n_cell, n_frame)
        dist = behav_df['Distance to point'].to_numpy()
        print(np.nanmin(dist,axis=0))
        dist_bins, dist_bin_cen, norm_fac = gen_dist_bins(dist, max_radius, min_radius, n_dist_bin)
        frInBin = firing_rate_per_bin(dist_bins,n_dist_bin,iter_spike_train)

        # Create null distribution by random shuffling of spike train for each neuron for 1000 times with np.roll
        null_frInBin = np.zeros((n_dist_bin, iter_spike_train.shape[1], permutation_times))
        for i in range(permutation_times):
            null_spike_train = np.roll(iter_spike_train, np.random.randint(0, iter_spike_train.shape[0]), axis=0)
            null_frInBin[:, :, i] = firing_rate_per_bin(dist_bins,n_dist_bin,null_spike_train)

        # shuffle test
        # null_permutation_thre = np.percentile(np.nanmax(null_frInBin,axis=0),95,axis=1) # 95 percentile of the maximum firing rate of each neuron from the null distance distribution
        # nfiB = null_frInBin[:]
        # nfiB[np.isnan(nfiB)] = 0
        # null_permutation_thre = np.nanpercentile(null_frInBin,97.5) # 95 percentile of the maximum firing rate of each neuron from the null distance distribution
        null_permutation_thre = np.nanpercentile(np.nanpercentile(null_frInBin, 95, axis=0), 95,
                                              axis=1)  # 95 percentile of the maximum firing rate of each neuron from the null distance distribution
        frBin_significance = np.nanmax(frInBin,axis=0) > null_permutation_thre # if the maximum firing rate of a neuron is larger than the 95 percentile of the null distribution, then it is considered as a distance preference neuron

        if np.sum(frBin_significance) > 0:
            for i in np.where(frBin_significance)[0]:
                sig_cell_trace.append([iter_spike_train[:,i], iter_behav_loc])
        #
        for i in range(frInBin.shape[1]):
            if not frBin_significance[i]:
                frInBin[:,i] *= -1  # if the neuron does not pass the shuffling test, its firing rate will be negative

        frplot.append(frInBin)
        spike_train.append(iter_spike_train)
        print(f"Significant neurons: {np.sum(frBin_significance)}/{frInBin.shape[1]}")
        sig_cell_arr.append(frBin_significance)

    frplot = np.concatenate(frplot,axis=1)
    frplot[np.isnan(frplot)] = 0
    frplot /= frplot.max(axis=0)
    frplots.append(frplot)
    sig_cell_trace_list.append(sig_cell_trace)
    sig_cell_arr = np.concatenate(sig_cell_arr,axis=0)
#%% Firing rate at each distance bin
for o in range(1):
    plt.clf()
    ax = plt.subplot(1,2,1)
    idx = np.argsort(np.argmax(frplots[o],axis=0))
    plt.imshow(frplots[o][:,idx].T,aspect='auto',interpolation='nearest',cmap='GnBu', extent=[min_radius,max_radius,0,frplots[o].shape[1]])
    plt.xlabel('Distance from point (cm)')
    plt.ylabel('Neuron #')

    ax = plt.subplot(1,2,2)
    for i in range(frplots[o].shape[1]):
        plt.plot(np.linspace(min_radius,max_radius,n_dist_bin),frplots[o][:,idx[::-1][i]]*2+i,'k')

    plt.xlabel('Distance from point (cm)')
    #plt.ylabel('Firing rate')
    ax.set_ylim([0,frplots[o].shape[1]+2])
    # plt.waitforbuttonpress()
    #%%%

#%% Scatter plots for the significant neurons' spatial firing pattern
for o in range(1):
    for i in sig_cell_trace_list[o]:
        behav_trace = i[1]
        spike_trace = i[0]
        sorted_scatter_plot_order = np.argsort(spike_trace)
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.plot(behav_trace[0, :], behav_trace[1, :], '--', color=[0, 0, 0, .2])
        ax.scatter(behav_trace[0, sorted_scatter_plot_order], behav_trace[1, sorted_scatter_plot_order],
                   c=spike_trace[sorted_scatter_plot_order],
                   s=spike_trace[sorted_scatter_plot_order] * 10)
        ax.set_aspect('equal')
        # save current ax.scatter plots to  jpg files in current folder

        # plt.savefig('{}.png'.format(i))

        plt.show()
        plt.draw()
        # save current all plots to one single jpg in current folder

        # fig = 'J:\\Hansol_Yue\\experiment data\\figures\\'{}'.jpg'
        # plt.savefig(fig)
        # plt.close()  # close the file
        #
        #                                                fig = 'J:\\Hansol_Yue\\experiment data\\figures\\'.str(i)'.png'

        # # save current multiple plots to a single pdf file
        #           plt.savefig(fig)
        #           plt.close()  # close the file
        #
        #
        # # plt.savefig('J:\\Hansol_Yue\\experiment data\\figures\\'.str(i)'.png')
        #
        # plt.waitforbuttonpress()  # click to continue to the next neuron
        #


#%%
plt.clf()
pidx = 1
sort_idx = np.argsort(iter_spike_train[:,pidx])
plt.scatter(iter_behav_loc[0,sort_idx],iter_behav_loc[1,sort_idx],c=iter_spike_train[sort_idx,pidx],s=iter_spike_train[sort_idx,pidx]*10)
# %% FR plotS

for o in range(3):
    plt.clf()
    ax = plt.subplot(1,2,1)
    idx = np.argsort(np.argmax(frplots[o],axis=0))
    iter_fr = frplots[o]
    iter_fr[np.isnan(iter_fr)] = 0
    iter_fr[np.isinf(iter_fr)] = 0
    plt.imshow(iter_fr[:,idx].T,aspect='auto',interpolation='nearest',cmap='GnBu', extent=[0,30,0,frplots[o].shape[1]])
    plt.xlabel('Distance from point (cm)')
    plt.ylabel('Neuron #')

    ax = plt.subplot(1,2,2)
    for i in range(frplots[o].shape[1]):
        plt.plot(np.linspace(0,44,44),iter_fr[:,idx[::-1][i]]*2+i,'k')

    plt.xlabel('Distance from point (cm)')
    plt.colorbar()
    #plt.ylabel('Firing rate')
    ax.set_ylim([0,frplots[o].shape[1]+2])
    # plt.waitforbuttonpress()

#%%
plt.clf()
pidx = 1
sort_idx = np.argsort(iter_spike_train[:,pidx])
plt.scatter(iter_behav_loc[0,sort_idx],iter_behav_loc[1,sort_idx],c=iter_spike_train[sort_idx,pidx],s=iter_spike_train[sort_idx,pidx]*10)
# %% FR plotS
plt.clf()
max_idx = []
for v in frplots:
    max_idx.append(np.linspace(0,30,50)[np.argmax(v,axis=0)])
max_idx = np.array(max_idx).squeeze()
plt.violinplot(max_idx, positions=[0], bw_method=0.2, showmeans=True, showextrema=True, showmedians=False)
plt.violinplot(max_idx[sig_cell_arr], positions=[0], bw_method=0.2, showmeans=True, showextrema=True, showmedians=False)  # significant only
plt.xticks([0,1,2])
plt.gca().set_xticklabels(['Etv1','Lypd1','Rspo2'])
plt.ylabel('Peak firing distance (cm)')
plt.savefig('C:/Users/lim/Downloads//rspo real social violn plot .pdf', dpi=1600, format='pdf')
#%%%%
len(max_idx[sig_cell_arr])
# # %% FR plotS
# plt.clf()
# max_idx = []
# for v in frplots:
#     max_idx.append(np.linspace(0, 30, 50)[np.argmax(v, axis=0)])
#
# plt.violinplot(max_idx, positions=[0,1,2], bw_method=0.2, showmeans=True, showextrema=True, showmedians=False)
# plt.xticks([0, 1, 2])
# plt.gca().set_xticklabels(['Etv1', 'Lypd1', 'Rspo2'])
# plt.ylabel('Peak firing distance (cm)')
# %%
temp = [(np.sum(i < 10), np.sum(i > 20), np.sum((i > 10) & (i < 20))) for i in max_idx[sig_cell_arr]]
pro = np.sum(max_idx[sig_cell_arr]<10)
anti = np.sum(max_idx[sig_cell_arr]>20)
neutral = np.sum((max_idx[sig_cell_arr] > 10) & (max_idx[sig_cell_arr] < 20) ) #np.sum(10<max_idx[sig_cell_arr]<20)
#%%
temp = [(np.sum(i < 5), np.sum(i > 17), np.sum((i > 5) & (i < 17))) for i in max_idx[sig_cell_arr]]
pro = np.sum(max_idx[sig_cell_arr]<5)
anti = np.sum(max_idx[sig_cell_arr]>17)
neutral = np.sum((max_idx[sig_cell_arr] > 5) & (max_idx[sig_cell_arr] < 17) )
# labels = 'prococial', 'antisocial', 'neutral'
# explode = (0.1, 0.1, 0)
# colors = ("magenta", "brown",
#           "grey")
# sizes = [15, 30, 45, 10]
# plt.clf()
# for i in range(3):
#     plt.subplot(1, 3, i + 1)
#     plt.pie(temp[i], autopct='%1.1f%%', startangle=90, colors=colors)
#     plt.legend(labels, loc="best", bbox_to_anchor=(1, 2))
#     plt.gca().set_title(['Etv1', 'Lypd1', 'Rspo2'][i], fontsize=16)
#%%
plt.clf()
max_idx[sig_cell_arr] = []
for v in frplots:
    max_idx[sig_cell_arr].append(np.linspace(0,30,50)[np.argmax(v,axis=0)])

plt.violinplot(max_idx[sig_cell_arr], positions=[0,1,2], bw_method=0.2, showmeans=True, showextrema=True, showmedians=False)
plt.xticks([0,1,2])
plt.gca().set_xticklabels(['Etv1','Lypd1','Rspo2'])
plt.ylabel('Peak firing distance (cm)')

#%%
max_idx= max_idx[sig_cell_arr]
#temp = [(np.mean(i<5),np.mean(i>17),np.mean((i>5)&(i<17))) for i in max_idx[sig_cell_arr]]  ## pro <10  # social 10  20 food 5 17
##all number = sum
temp = [(np.sum(i<5),np.sum(i>17),np.sum((i>5)&(i<17))) for i in max_idx]
labels = 'prococial','antisocial','neutral'
explode = (0.1, 0.1, 0)
colors = ( "magenta", "brown",
          "grey")
sizes = [15, 30, 45, 10]
plt.clf()
for i in range(1):
    plt.subplot(1,3,i+1)
    plt.pie(temp[i],autopct='%1.1f%%',startangle=90,colors=colors)
    plt.legend(labels, loc="best", bbox_to_anchor=(1, 2))
    plt.gca().set_title(['Etv1','Lypd1','Rspo2'][i],fontsize=16)
# %% FR plotS
plt.clf()
max_idx[sig_cell_arr] = []
for v in frplots:
    max_idx[sig_cell_arr].append(np.linspace(0,30,50)[np.argmax(v,axis=0)])

plt.violinplot(max_idx[sig_cell_arr], positions=[0,1,2], bw_method=0.2, showmeans=True, showextrema=True, showmedians=False)
plt.xticks([0,1,2])
plt.gca().set_xticklabels(['Etv1','Lypd1','Rspo2'])
plt.ylabel('Peak firing distance (cm)')
#%%
temp = [(np.mean(i<10),np.mean(i>20),np.mean((i>10)&(i<20))) for i in max_idx[sig_cell_arr]]
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