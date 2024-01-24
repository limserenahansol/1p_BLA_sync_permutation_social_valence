from suite2p.extraction import dcnv
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

from matplotlib import use
use('qt5agg')

from core import util

# Declare path variables
ROOT_DATADIR = "J:/Hansol_Yue/new_ca_social_fc_together_for_new/"  # Please change to your Hansol_Yue folder path
animal_ids = ['Etv1_2','Etv1_1']#'Etv1_2','Etv1_1']#, 'Etv1_2']#, 'Etv1_4']#, 'Lypd1_1', 'Lypd1_2']  # Please Change this to the animal id you want to analyze
shock_twin = 20  # Time window for PSTH, in frames, e.g. shock_twin = 30, the PSTH will include the calcium/spike activities in 30 frames before and after the onset of shock/freezing event
ca_tau =0.1  # Please change to the correct tau value for your data
ca_fs = 10  # Please change to the correct sampling rate for your data
exp_id = "FC_1"
session_id = 1
h5fn = 'preprocessed_TEST.h5'
#%%
output_all = {}
for animal_id in animal_ids:
    fn = f'{ROOT_DATADIR}/{animal_id}/{exp_id}/session{session_id}/{h5fn}'
    behav_data,ca_data = util.load_data_file(fn)
    if animal_id == 'Etv1_1':
        social_st, social_ed = 80, 140
        shock_st, shock_ed = 200, 380
    else:
        social_st, social_ed = 80, 200
        shock_st, shock_ed = 260, 480
    social_st_idx, social_ed_idx = np.argmin(np.abs(behav_data.index-social_st)), np.argmin(np.abs(behav_data.index-social_ed))
    shock_st_idx, shock_ed_idx = np.argmin(np.abs(behav_data.index-shock_st)), np.argmin(np.abs(behav_data.index-shock_ed))
    social_spikes = dcnv.oasis(ca_data.iloc[social_st_idx:social_ed_idx,:].to_numpy().T, batch_size = 1, tau=ca_tau, fs=ca_fs).T
    shock_spikes = dcnv.oasis(ca_data.iloc[shock_st_idx:shock_ed_idx,:].to_numpy().T, batch_size = 1, tau=ca_tau, fs=ca_fs).T
    mouse_position = behav_data[['Centre position X', 'Centre position Y']].iloc[social_st_idx:social_ed_idx,:].to_numpy()
    social_idx = behav_data['Investigating social zone'].iloc[social_st_idx:social_ed_idx]>0
    shock_onset = np.where(behav_data['Floor shock active'].iloc[shock_st_idx:shock_ed_idx].diff()>0)[0]
    shock_idx = behav_data['Floor shock active'].iloc[shock_st_idx:shock_ed_idx]*0
    for i in shock_onset:
        shock_idx.iloc[i-shock_twin:i] = -1
        shock_idx.iloc[i:i+shock_twin] = 1
    output_all[animal_id] = {'social_spikes':social_spikes, 'shock_spikes':shock_spikes, 'social_idx':social_idx,
                             'shock_idx':shock_idx, 'mouse_social_position':mouse_position}

#%% (Optional) plot the behavioral label for social and shock session to check if they were properly extracted
# The etv1_1 and lypd1_1 have fewer shock times than 5, also the shock interval are not equal in all 5 animals
make_plot = False
animal_id = 'Lypd1_2'
if make_plot:
    plt.figure(1)
    plt.clf()
    plt.subplot(111)
    plt.plot(output_all[animal_id]['social_idx'], 'b')
    plt.plot(output_all[animal_id]['shock_idx'],'r')
    plt.show()
    plt.pause(10)
#%% Pool all social and shock average FRs and calculate the social and shock score
get_avg_with_crit = lambda spk_key, idx_key, critfun, x: np.nanmean(x[spk_key][critfun(x[idx_key]),:], axis=0)
get_avg = lambda spk_key, x: np.nanmean(x[spk_key], axis=0)

avg_fr_social_on = np.hstack(list(map(partial(get_avg_with_crit, 'social_spikes','social_idx', lambda x: x>0), output_all.values())))
avg_fr_social_off = np.hstack(list(map(partial(get_avg_with_crit, 'social_spikes','social_idx', lambda x: x==0), output_all.values())))
avg_fr_social_all = np.hstack(list(map(partial(get_avg, 'social_spikes'), output_all.values())))
social_scr = (avg_fr_social_on - avg_fr_social_off)/(avg_fr_social_on + avg_fr_social_off)

avg_fr_shock_on = np.hstack(list(map(partial(get_avg_with_crit, 'shock_spikes','shock_idx', lambda x: x>0), output_all.values())))
avg_fr_shock_off = np.hstack(list(map(partial(get_avg_with_crit, 'shock_spikes','shock_idx', lambda x: x<0), output_all.values())))
avg_fr_shock_all = np.hstack(list(map(partial(get_avg, 'shock_spikes'), output_all.values())))
shock_scr = (avg_fr_shock_on - avg_fr_shock_off)/(avg_fr_shock_on + avg_fr_shock_off)

nanfreeidx = ~np.isnan(social_scr) & ~np.isnan(shock_scr)
avg_fr_social_on = avg_fr_social_on[nanfreeidx]
avg_fr_shock_on = avg_fr_shock_on[nanfreeidx]
avg_fr_social_off = avg_fr_social_off[nanfreeidx]
avg_fr_shock_off = avg_fr_shock_off[nanfreeidx]
avg_fr_social_all = avg_fr_social_all[nanfreeidx]
avg_fr_shock_all = avg_fr_shock_all[nanfreeidx]
social_scr = social_scr[nanfreeidx]
shock_scr = shock_scr[nanfreeidx]


#%% Plot the average FRs and social/shock scores, also plot the linear regressions

plt.figure(2)
plt.clf()
plt.subplot(121)
val_min, val_max = np.min([np.min(avg_fr_social_off), np.min(avg_fr_shock_off)]), np.max([np.max(avg_fr_social_off), np.max(avg_fr_shock_off)])
#mask = (avg_fr_social_on >= 0) & (avg_fr_social_on <= 30)
plt.scatter(avg_fr_social_on, avg_fr_shock_on, s=20, c='k')
# linear regression with r2
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(avg_fr_social_on, avg_fr_shock_on)
plt.plot([val_min, val_max],[val_min*slope+intercept, val_max*slope+intercept],'k--')
plt.title(f'Average social vs shock FR  |  $R^{2}$ = {r_value**2:.2f}')# why no p value here?
### same scale for lypd1
plt.figure(2)
plt.clf()
plt.subplot(121)
val_min, val_max = np.min([np.min(avg_fr_social_off), np.min(avg_fr_shock_off)]), np.max([np.max(avg_fr_social_off), np.max(avg_fr_shock_off)])
mask = (avg_fr_social_on >= 0) & (avg_fr_social_on <= 30)
plt.scatter(avg_fr_social_on[mask], avg_fr_shock_on[mask], s=20, c='k')
# linear regression with r2
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(avg_fr_social_on, avg_fr_shock_on)
plt.plot([0, 20],[val_min*slope+intercept, val_max*slope+intercept],'k--')
plt.title(f'Average social vs shock FR  |  $R^{2}$ = {r_value**2:.2f}')# why no p value here?
####


ax = plt.gca()
# ax.set_aspect('equal', 'box')
ax.set_xlabel('Social avg FR')
ax.set_ylabel('Post-Shock avg FR')

plt.subplot(122)
plt.scatter(social_scr, shock_scr, s=20, c='k')
social_scr = social_scr[~np.isnan(shock_scr)]
shock_scr = shock_scr[~np.isnan(shock_scr)]
slope, intercept, r_value, p_value, std_err = linregress(social_scr, shock_scr)
plt.plot([-1, 1],[-slope+intercept, slope+intercept],'k--')
plt.title(f'Average social vs shock FR  |  $R^{2}$ = {r_value**2:.2f}| p_value = {p_value:.2f}')
ax = plt.gca()
ax.set_aspect('equal', 'box')
ax.set_xlabel('Social score (-1 antisocial - 1 pro-social)')
ax.set_ylabel('Shock score (-1 antishock - 1 pro-shock)')
# plt.pause(50)

#%% Plot example traces for "pro-/ anit-social" and "pro-/ anti-shock" neurons
concat_mouse_pos_func = lambda x: list(list(map(lambda y: y['mouse_social_position'], x.values())))
concat_mouse_pos = concat_mouse_pos_func(output_all)
concat_social_idx_func = lambda x: list(list(map(lambda y: y['social_idx'], x.values())))
concat_social_idx = concat_social_idx_func(output_all)
concat_shock_idx_func = lambda x: list(list(map(lambda y: y['shock_idx'], x.values())))
concat_shock_idx = concat_shock_idx_func(output_all)
concat_social_spk_func = lambda x: list(list(map(lambda y: y['social_spikes'], x.values())))
concat_social_spk = concat_social_spk_func(output_all)
concat_shock_spk_func = lambda x: list(list(map(lambda y: y['shock_spikes'], x.values())))
concat_shock_spk = concat_shock_spk_func(output_all)

concate_mouse_idx = []
counter = 0
for k in output_all.keys():
    concate_mouse_idx.append([np.ones(output_all[k]['social_spikes'].shape[1])*counter, np.arange(output_all[k]['social_spikes'].shape[1])])
    counter += 1
concate_mouse_idx = np.hstack(concate_mouse_idx).astype(int)

# get the neuron with the highest social score
social_sorted_idx = np.argsort(social_scr) # sort the neurons based on their social score from low to high
shock_sorted_idx = np.argsort(shock_scr) # sort the neurons based on their shock score from low to high

# Iteratively plot through the top 10 neurons with the highest social score
plt.figure(3)
plt.clf()
for i in range(5):
    plt.subplot(211)
    mouse_idx,cell_idx = concate_mouse_idx[:,social_sorted_idx[-i-1]]
    iter_social_spikes = concat_social_spk[mouse_idx][:,cell_idx]
    iter_social_idx = concat_social_idx[mouse_idx]
    iter_shock_spikes = concat_shock_spk[mouse_idx][:, cell_idx]
    iter_shock_idx = concat_shock_idx[mouse_idx]
    plt.plot(iter_social_idx.index,util.rnorm(iter_social_spikes)+i*1.5,'k')
    plt.plot(iter_social_idx+i*1.5,'r')
    plt.plot(iter_shock_idx.index,util.rnorm(iter_shock_spikes)+i*1.5, 'k')
    plt.plot((iter_shock_idx>0)+i*1.5, 'r')
    plt.title(f'Top 5 "social" neuron')

    plt.subplot(212)
    mouse_idx,cell_idx = concate_mouse_idx[:,social_sorted_idx[i]]
    iter_social_spikes = concat_social_spk[mouse_idx][:,cell_idx]
    iter_social_idx = concat_social_idx[mouse_idx]
    iter_shock_spikes = concat_shock_spk[mouse_idx][:, cell_idx]
    iter_shock_idx = concat_shock_idx[mouse_idx]
    plt.plot(iter_social_idx.index,util.rnorm(iter_social_spikes)+i*1.5,'k')
    plt.plot(iter_social_idx+i*1.5,'r')
    plt.plot(iter_shock_idx.index,util.rnorm(iter_shock_spikes)+i*1.5, 'k')
    plt.plot((iter_shock_idx>0)+i*1.5, 'r')
    plt.title(f'Bottom 5 "social" neuron')

# Iteratively plot through the top 10 neurons with the highest social score
plt.figure(4)
plt.clf()
for i in range(5): # for some reason the first 2 "pro-shock" neurons look funky, so skip them
    plt.subplot(211)
    mouse_idx,cell_idx = concate_mouse_idx[:,shock_sorted_idx[-i-1]]
    iter_social_spikes = concat_social_spk[mouse_idx][:,cell_idx]
    iter_social_idx = concat_social_idx[mouse_idx]
    iter_shock_spikes = concat_shock_spk[mouse_idx][:, cell_idx]
    iter_shock_idx = concat_shock_idx[mouse_idx]
    plt.plot(iter_social_idx.index,util.rnorm(iter_social_spikes)+i*1.5,'k')
    plt.plot(iter_social_idx+i*1.5,'r')
    plt.plot(iter_shock_idx.index,util.rnorm(iter_shock_spikes)+i*1.5, 'k')
    plt.plot((iter_shock_idx>0)+i*1.5, 'r')
    plt.title(f'Top 5 "shock" neuron')

    plt.subplot(212)
    mouse_idx,cell_idx = concate_mouse_idx[:,shock_sorted_idx[i]]
    iter_social_spikes = concat_social_spk[mouse_idx][:,cell_idx]
    iter_social_idx = concat_social_idx[mouse_idx]
    iter_shock_spikes = concat_shock_spk[mouse_idx][:, cell_idx]
    iter_shock_idx = concat_shock_idx[mouse_idx]
    plt.plot(iter_social_idx.index,util.rnorm(iter_social_spikes)+i*1.5,'k')
    plt.plot(iter_social_idx+i*1.5,'r')
    plt.plot(iter_shock_idx.index,util.rnorm(iter_shock_spikes)+i*1.5, 'k')
    plt.plot((iter_shock_idx>0)+i*1.5, 'r')
    plt.title(f'Bottom 5 "shock" neuron')

coactive_neuron_idx = np.where(np.bitwise_and(social_scr>0.5, shock_scr>0.5))[0]
plt.figure(5)
plt.clf()
for i,v in enumerate(coactive_neuron_idx): # for some reason the first 2 "pro-shock" neurons look funky, so skip them
    plt.subplot(111)
    mouse_idx,cell_idx = concate_mouse_idx[:,v]
    iter_social_spikes = concat_social_spk[mouse_idx][:,cell_idx]
    iter_social_idx = concat_social_idx[mouse_idx]
    iter_shock_spikes = concat_shock_spk[mouse_idx][:, cell_idx]
    iter_shock_idx = concat_shock_idx[mouse_idx]
    plt.plot(iter_social_idx.index,util.rnorm(iter_social_spikes)+i*1.5,'k')
    plt.plot(iter_social_idx+i*1.5,'r')
    plt.plot(iter_shock_idx.index,util.rnorm(iter_shock_spikes)+i*1.5, 'k')
    plt.plot((iter_shock_idx>0)+i*1.5, 'r')
    plt.title(f'Shock+/social+ neuron')


#%% Pp value
from scipy.stats import t
import numpy as np
r = np.corrcoef(avg_fr_social_on, avg_fr_shock_on)[0, 1] # get the correlation coefficient
n = len(avg_fr_social_on) # get the number of paired data
t_stat = r * np.sqrt(n-2) / np.sqrt(1-r**2) # calculate the t-statistic
p = 2 * (1 - t.cdf(abs(t_stat), n-2)) # calculate the p-value

print(p)

#%% r squared  value
from scipy.stats import linregress
res=slope, intercept,  r_value, p_value, std_err = linregress(avg_fr_social_on, avg_fr_shock_on)
#print(f"R-squared: {res.rvalue**2:.6f}")
p1=float("{:.9f}".format(p_value))
print(p_value)
# p value is here to small. how I can calculate it correctly?

#%% r
from scipy.stats import t
import numpy as np
r = np.corrcoef(social_scr, shock_scr)[0, 1] # get the correlation coefficient
n = len(avg_fr_social_on) # get the number of paired data
t_stat = r * np.sqrt(n-2) / np.sqrt(1-r**2) # calculate the t-statistic
p = 2 * (1 - t.cdf(abs(t_stat), n-2)) # calculate the p-value

print(p)


plt.figure(2)
plt.clf()
plt.subplot(121)
val_min, val_max = np.min([np.min(avg_fr_social_off), np.min(avg_fr_shock_off)]), np.max([np.max(avg_fr_social_off), np.max(avg_fr_shock_off)])
mask = (avg_fr_social_on >= 0) & (avg_fr_social_on <= 30)
plt.scatter(avg_fr_social_on[mask], avg_fr_shock_on[mask], s=20, c='k')
# linear regression with r2
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(avg_fr_social_on, avg_fr_shock_on)
plt.plot([0, 20],[val_min*slope+intercept, val_max*slope+intercept],'k--')
plt.title(f'Average social vs shock FR  |  $R^{2}$ = {r_value**2:.2f}')# why no p value here?
