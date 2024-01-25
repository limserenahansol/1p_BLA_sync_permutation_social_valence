import numpy as np
import h5py as h5
import pandas as pd


def comput_subplot_layout(n):
    base = np.sqrt(n)
    min_base = int(np.floor(base))
    max_base = min_base+1
    if min_base * max_base >= n:
        return min_base, max_base
    else:
        return max_base, max_base


def load_data_file(fn):
    with h5.File(fn) as data_file:
        behav_data = data_file['Aligned behavior data']
        ca_data = data_file['Raw Ca data']
        behav_df = {}
        for k, v in behav_data.items():
            if 'In zone(' in k:
                nk = f"In zone {k.split('(')[1].strip(' ')}"
                behav_df[nk] = np.array(list(v.values())[0])
            elif k == 'time':
                behav_time = v
            else:
                behav_df[k] = np.array(v)
        behav_df = pd.DataFrame(behav_df, index=behav_time)

        ca_df = {}

        for k, v in ca_data.items():
            if k != 'time':
                if v.attrs['accepted']:
                    ca_df[k] = np.array(v)
            else:
                ca_dataframe_index = np.array(v)
        ca_df = pd.DataFrame(ca_df, index=ca_dataframe_index)
    return behav_df, ca_df

def getPeriEventData(data, eventTrace, preEvtWin=20, postEvtWin=40):
    periEventIdx = np.where(eventTrace)[0]
    periEventData = np.zeros(
        (preEvtWin + postEvtWin, data.shape[1], len(periEventIdx)))  # nt x n_cell x n_trial
    periEventFullIdx = []
    for i, v in enumerate(periEventIdx):
        st_idx = v - preEvtWin
        ed_idx = v + postEvtWin
        pad_st = 0
        pad_ed = preEvtWin + postEvtWin
        if st_idx < 0:
            pad_st = -st_idx
        if ed_idx > data.shape[0]:
            pad_ed -= ed_idx - data.shape[0]
            ed_idx = data.shape[0]
        periEventData[pad_st:pad_ed, :, i] = data.iloc[st_idx:ed_idx, :]
        periEventFullIdx.append(np.concatenate([np.ones(pad_st)*st_idx,np.arange(st_idx, ed_idx),np.ones(preEvtWin+postEvtWin-pad_ed)*ed_idx-1]))
    periEventFullIdx = np.array(periEventFullIdx, dtype=int)
    return periEventData,periEventFullIdx


def rnorm(data, *args, **kwargs):
    d_min = np.min(data, *args, **kwargs)
    d_max = np.max(data, *args, **kwargs)
    if 'axis' in kwargs.keys():
        d_min = np.expand_dims(d_min,axis=kwargs['axis'])
        d_max = np.expand_dims(d_max,axis=kwargs['axis'])
    return (data - d_min) / (d_max - d_min)
