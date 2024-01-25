import numpy as np
# #significant social
# lypd1 = np.array([4,8,17])  # pro neutral anti  13.7 %    27.5%  58.8%
# etv1 = np.array([40,10,17]) # 59.7%  15%  25.3

# rspo2 = np.array([9,16,11])  #25 44.4 30.6


# ##significant food
# lypd1 = np.array([29,7,6])  # pro neutral anti 69.1 16.7 14.2      23.8 19.0
# etv1 = np.array([43,18,57]) # 29.5  15.5  54.9
# rspo2 = np.array([14,37,3]) # 25.9       68.6  5.5
total_pop_num = lypd1 + etv1 + rspo2
total_pop = np.hstack([[i]*v for i,v in enumerate(total_pop_num)])

null_lypd = []
null_etv = []
null_rspo = []
for i in range(5000):
    iter_lypd_null = np.random.choice(total_pop,sum(lypd1))
    iter_lypd = np.array([sum(iter_lypd_null==i) for i in np.unique(iter_lypd_null)])
    null_lypd.append(iter_lypd)
    iter_etv_null = np.random.choice(total_pop,sum(etv1))
    iter_etv = np.array([sum(iter_etv_null==i) for i in np.unique(iter_etv_null)])
    null_etv.append(iter_etv)
    iter_rspo_null = np.random.choice(total_pop,sum(rspo2))
    iter_rspo = np.array([sum(iter_rspo_null==i) for i in np.unique(iter_rspo_null)])
    null_rspo.append(iter_rspo)
#%%%

null_lypd = np.array(null_lypd)
null_etv = np.array(null_etv)
null_rspo = np.array(null_rspo)
#%%
high_lypd = np.percentile(null_lypd,100-5/2/6,axis=0)
low_lypd = np.percentile(null_lypd,5/6,axis=0)
high_etv = np.percentile(null_etv,100-5/2/6,axis=0)
low_etv = np.percentile(null_etv,5/6,axis=0)
high_rspo = np.percentile(null_rspo,100-5/2/6,axis=0)
low_rspo = np.percentile(null_rspo,5/6,axis=0)
print((lypd1>high_lypd).astype(int)-(lypd1<low_lypd).astype(int))
print((etv1>high_etv).astype(int)-(etv1<low_etv).astype(int))
print((rspo2>high_rspo).astype(int)-(rspo2<low_rspo).astype(int))
#%% bar plot: plot the number of cells in each cluster side-by-side with the null distribution
import matplotlib.pyplot as plt

fig = plt.figure(1)
ax = fig.add_subplot(131)
ax.bar(np.arange(3)-0.2,lypd1,width=0.2,color='g',alpha=0.5,label='lypd1')
# plot null distribution mean and add error bar (std)
ax.bar(np.arange(3),null_lypd.mean(axis=0),width=0.2,color='k',alpha=0.5)
ax.errorbar(np.arange(3),null_lypd.mean(axis=0),yerr=null_lypd.std(axis=0),fmt='none',ecolor='k',capsize=5)
# put a star on the cluster that is significantly different from null distribution
ax.scatter(np.arange(3)[(lypd1>high_lypd).astype(int)-(lypd1<low_lypd).astype(int)==1],np.ones(3)[(lypd1>high_lypd).astype(int)-(lypd1<low_lypd).astype(int)==1]*lypd1.max()*1.1,marker='*',s=100,color='k')
ax.scatter(np.arange(3)[(lypd1>high_lypd).astype(int)-(lypd1<low_lypd).astype(int)==-1],np.ones(3)[(lypd1>high_lypd).astype(int)-(lypd1<low_lypd).astype(int)==-1]*lypd1.max()*1.1,marker='*',s=100,color='k')

# same for etv
ax = fig.add_subplot(132)
ax.bar(np.arange(3)-0.2,etv1,width=0.2,color='r',alpha=0.5,label='etv1')
ax.bar(np.arange(3),null_etv.mean(axis=0),width=0.2,color='k',alpha=0.5)
ax.errorbar(np.arange(3),null_etv.mean(axis=0),yerr=null_etv.std(axis=0),fmt='none',ecolor='k',capsize=5)
ax.scatter(np.arange(3)[(etv1>high_etv).astype(int)-(etv1<low_etv).astype(int)==1],np.ones(3)[(etv1>high_etv).astype(int)-(etv1<low_etv).astype(int)==1]*etv1.max()*1.1,marker='*',s=100,color='k')
ax.scatter(np.arange(3)[(etv1>high_etv).astype(int)-(etv1<low_etv).astype(int)==-1],np.ones(3)[(etv1>high_etv).astype(int)-(etv1<low_etv).astype(int)==-1]*etv1.max()*1.1,marker='*',s=100,color='k')

# same for rspo2
ax = fig.add_subplot(133)
ax.bar(np.arange(3)-0.2,rspo2,width=0.2,color='b',alpha=0.5,label='rspo2')
ax.bar(np.arange(3),null_rspo.mean(axis=0),width=0.2,color='k',alpha=0.5)
ax.errorbar(np.arange(3),null_rspo.mean(axis=0),yerr=null_rspo.std(axis=0),fmt='none',ecolor='k',capsize=5)
ax.scatter(np.arange(3)[(rspo2>high_rspo).astype(int)-(rspo2<low_rspo).astype(int)==1],np.ones(3)[(rspo2>high_rspo).astype(int)-(rspo2<low_rspo).astype(int)==1]*rspo2.max()*1.1,marker='*',s=100,color='k')
# # same for low_rspo
# ax.scatter(np.arange(3)[(rspo2>high_rspo).astype(int)-(rspo2<low_rspo).astype(int)==-1],np.ones(3)[(rspo2>high_rspo).astype(int)-(rspo2<low_rspo).astype(int)==-1]*rspo2.max()*1.1,marker='o',s=100,color='k')
plt.savefig('C:/Users/lim/Downloads//bargraph food.pdf',dpi=1200,format='pdf')
