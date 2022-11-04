%data import
imported_data = importdata('sync_uwb_sig-001.mat');
sleep_Raw_data = imported_data.data{1,1};
sleep_Preprocessing_data = imported_data.data{3,1};