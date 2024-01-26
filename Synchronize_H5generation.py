import core.preprocessing as prep

#%% To process a single session folder
# demo_data_path = r'Z:\\Hansol_Yue\\experiment data\\Etv1_1\\Social_1\\session_2\\'
demo_data_path = 'J:\Hansol_Yue\_specialcase_of_behavior_\Rspo2_noffood'
prep.align_ca_behav_data(demo_data_path,'TEST',use_independent_logger=True)

#%% To process a single session folder
# demo_data_path = r'Z:\\Hansol_Yue\\experiment data\\Etv1_1\\Social_1\\session_2\\'
demo_data_path = 'J:\\Hansol_Yue\\new_ca_social_fc_together\\Lypd1_4\\FC_2\\session1'
prep.align_ca_behav_data(demo_data_path,'TEST',use_independent_logger=True)
#%% To process the entire experiment folder
demo_data_path = 'J:\\Hansol_Yue\\new_ca_social_fc_together\\'
ps = prep.ProjectScreener(demo_data_path,'YZ')
ps.batch_processing(1)
