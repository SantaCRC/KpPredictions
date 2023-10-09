import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

# Define the list of file names
filenames = ["/content/drive/MyDrive/SpaceApps/dsc_fc_summed_spectra_2016_v01.csv", "/content/drive/MyDrive/SpaceApps/dsc_fc_summed_spectra_2017_v01.csv", "/content/drive/MyDrive/SpaceApps/dsc_fc_summed_spectra_2018_v01.csv","/content/drive/MyDrive/SpaceApps/dsc_fc_summed_spectra_2019_v01.csv","/content/drive/MyDrive/SpaceApps/dsc_fc_summed_spectra_2020_v01.csv","/content/drive/MyDrive/SpaceApps/dsc_fc_summed_spectra_2021_v01.csv","/content/drive/MyDrive/SpaceApps/dsc_fc_summed_spectra_2022_v01.csv","/content/drive/MyDrive/SpaceApps/dsc_fc_summed_spectra_2023_v01.csv"]  # Add or remove filenames as necessary
column_names = ['datetime'] + ['sensor_data_' + str(i) for i in range(1, 54)]
# Load each file into a DataFrame and store them in a list
dataframes = [pd.read_csv(filename,parse_dates=True, index_col=0,names=column_names) for filename in filenames]

# Concatenate all DataFrames in the list together
mergeded_df = pd.concat(dataframes)

# 1. Load raw data from CSV
raw_df = pd.read_csv('/content/drive/MyDrive/SpaceApps/dsc_fc_summed_spectra_2016_v01.csv', parse_dates=True, index_col=0,names=column_names)

# 2. Load Kp index from TXT and convert to datetime
kp_df = pd.read_csv('/content/drive/MyDrive/SpaceApps/kp.gfz-potsdam.de_kpdata_startdate=2016-01-01&enddate=2022-12-31&format=kp2#kpdatadownload-143.txt', delim_whitespace=True, header=None)

# Define a datetime column from the first four columns
kp_df[3] = kp_df[3].astype(int).astype(str)  # Convert hours from float to integer to string
kp_df[0] = pd.to_datetime(kp_df[0].astype(str) + '-' + kp_df[1].astype(str) + '-' + kp_df[2].astype(str) + ' ' + kp_df[3].astype(int).astype(str) + ':00:00')
kp_df.set_index(0, inplace=True)

# Keep only the 7th column as Kp value and rename it
kp_df = kp_df.iloc[:, 6]
kp_df.name = 'kp_value'

# 3. Resample the raw data to 3-hour intervals
aggregated_data = mergeded_df.resample('3H').mean()

# 4. Merge aggregated data with Kp index
merged_df = aggregated_data.join(kp_df, how='inner')  # This will only keep rows where both dataframes have values

# reanming the DataFrame columns
# Handle any NaNs that might result from the merge
merged_df.dropna(inplace=True)