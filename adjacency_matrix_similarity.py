import os
import numpy as np
import pandas as pd
import calendar
from utils import calculate_neiborhoods
import time
from correlation import crosscorr, pearson_distance_add, normalization_corr
from utils import min_max_norm


# 1.1 read dublin bike data
dublin_bike_file = "./data/bike_dataset/dublinbikes_20200701_20201001.csv"
df = pd.read_csv(dublin_bike_file, dtype={'STATION ID': int, 'BIKE STANDS':int, \
        'AVAILABLE BIKE STANDS': int, 'AVAILABLE BIKES':int, 'LATITUDE': float, \
        'LONGITUDE': float})

# 2.1 dublin bike data timestamp remove second
def remove_second(x):
    return x["TIME"][0:16]
df["TIME"]=df.apply(remove_second, axis=1)

station_ids = list(df["STATION ID"].drop_duplicates().values)
position_LA = list(df["LATITUDE"].drop_duplicates().values)
position_LO = list(df["LONGITUDE"].drop_duplicates().values)
record_dates = list(df["TIME"].drop_duplicates().values)

bike_series = []
print(station_ids)
for i in station_ids:
    bike_series.append(np.array(df[(df["STATION ID"] == i)]["AVAILABLE BIKES"]))
    # print(len(df[(df["STATION ID"] == i)]["AVAILABLE BIKES"]))#26464

# print(bike_series[0])
# 4. create adjacency matrix:
adjacency_matrix = np.zeros((len(station_ids),len(station_ids)))
for i in range(len(station_ids)):
    for j in range(len(station_ids)):
        if i==j:
            adjacency_matrix[i][j] = 1
        else:
            distance = ((position_LA[i]-position_LA[j])**2+ \
                (position_LO[i]-position_LO[j])**2)**0.5
            # print(bike_series[i].shape)
            # print(bike_series[j].shape)
            # print(np.corrcoef(bike_series[i].T, bike_series[j].T))
            # adjacency_matrix[i][j] = np.corrcoef(bike_series[i], bike_series[j])[0,1]
            pearson_corr = pearson_distance_add(bike_series[i], bike_series[j], position_LA[i], \
                position_LA[j], position_LO[i], position_LO[j])
            norm_distance = min_max_norm(distance,0.001,0.1)
            if pearson_corr < 0:
                adjacency_matrix[i][j] = pearson_corr
            else:
                adjacency_matrix[i][j] = max(norm_distance, pearson_corr)
            # adjacency_matrix[i][j] = norm_distance+pearson_corr
            # print("*******************")
            # print(distance)
            # print(pearson_corr)
            # print(adjacency_matrix[i][j])
np.save('./data/adjacency_matrix_selected.npy', adjacency_matrix)

