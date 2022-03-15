import os
import numpy as np
import pandas as pd
import calendar
from utils import calculate_neiborhoods
import time


# 1.1 read dublin bike data
dublin_bike_file = "./data/bike_dataset/dublinbikes_20200701_20201001.csv"
df = pd.read_csv(dublin_bike_file, dtype={'STATION ID': int, 'BIKE STANDS':int, \
        'AVAILABLE BIKE STANDS': int, 'AVAILABLE BIKES':int, 'LATITUDE': float, \
        'LONGITUDE': float})

# 1.2 read dublin historical weather data
dublin_weather_file = "./data/weather_history_data_202001-10.csv"
df_weather = pd.read_csv(dublin_weather_file, dtype={'Temperature': float, \
        'Wind Speed': float, 'Cloud Cover': float, 'Relative Humidity': float})
# and 'Conditions'

# 2.1 dublin bike data timestamp remove second
# def remove_second(x):
#     return x["TIME"][0:16]
# df["TIME"]=df.apply(remove_second, axis=1)

station_ids = list(df["STATION ID"].drop_duplicates().values)
position_LA = list(df["LATITUDE"].drop_duplicates().values)
position_LO = list(df["LONGITUDE"].drop_duplicates().values)
record_dates = list(df["TIME"].drop_duplicates().values)
# print(record_dates.index('2020-07-01 00:05'))
print(len(record_dates))#26752
print(len(station_ids))#110

# 2.2 dublin weather data processing
weather_conditions = list(df_weather["Conditions"].drop_duplicates().values)

# 2.3 find the nodes_neighberhoods
neiborhood_lists = calculate_neiborhoods(position_LA, position_LO)
# print(neiborhood_lists) neighberhood number range from 2 to 8
# for i in range(len(neiborhood_lists)):
#     print(len(neiborhood_lists[i]))
# print(len(neiborhood_lists))

# 3. create history data: 
#       0: available bikes, 1: time, 2: workday, 3: weather_cond, 4: temp, 5: neiborhood
#       6: wind_speed, 7:cloud_cover, 8: humidity
times = 1
record_dates = [record_dates[i] for i in range(len(record_dates)) if i%times==0]
history_node_data = np.zeros((len(record_dates), 8, len(station_ids)), dtype=np.int8)-1 #number of dates * number of nodes, value = 0
length = df.shape[0]
# print(len(record_dates))

for i in range(length):
    now = time.time()
    time_ = df["TIME"][i]
    station_ = df["STATION ID"][i]
    available_bikes = df["AVAILABLE BIKES"][i]
    if time_ in record_dates:
        time_index = record_dates.index(time_)
        station_index = station_ids.index(station_)
        neiborhood_list_ = neiborhood_lists[station_index]
        history_node_data[time_index,0,station_index] = available_bikes
        # 1. daytime
        history_node_data[time_index,1,station_index] = int(time_[11:13])
        # 2. weekday
        year = int(time_[0:4])
        month = int(time_[5:7])
        day = int(time_[8:10])
        history_node_data[time_index,2,station_index] = calendar.weekday(year, month, day)
        # 3. weather
        weather_date = time_[5:7]+'/'+time_[8:10]+'/'+time_[0:4]+" "+time_[11:13]+":00:00"
        #'01/01/2020 04:00:00'
        # print(weather_date)
        index = np.where(df_weather['Date time'].values == weather_date)
        if (len(index)==0):
            continue
        # print(len(index))
        weather_cond = df_weather['Conditions'][index[0]].values[0]
        weather_cond_index = weather_conditions.index(weather_cond)#3
        temperature = df_weather['Temperature'][index[0]].values[0]#4
        wind_speed = df_weather['Wind Speed'][index[0]].values[0]#5
        cloud_cover = df_weather['Cloud Cover'][index[0]].values[0]#6
        humidity = df_weather['Relative Humidity'][index[0]].values[0]#7
        history_node_data[time_index,3,station_index] = weather_cond_index
        history_node_data[time_index,4,station_index] = temperature
        history_node_data[time_index,5,station_index] = wind_speed
        history_node_data[time_index,6,station_index] = cloud_cover
        history_node_data[time_index,7,station_index] = humidity
        # 4. neiborhood available bikes statistic:
        # neiborhood_available_bikes = 0
        # for j in range(len(neiborhood_list_)):
        #     neiborhood = neiborhood_list_[j]
        #     _tmp = (df["TIME"]==time_) & (df['STATION ID']==station_)
        #     if sum(_tmp) == 0:
        #         print("none")
        #         continue
        #     else:
        #         num_values = df['AVAILABLE BIKES'][_tmp].values
        #         if len(num_values) > 0:
        #             neiborhood_available_bikes += num_values[0]
        # history_node_data[time_index,5,station_index] = neiborhood_available_bikes
        # print('neiborhood_available_bikes:')
        # print(neiborhood_available_bikes)
    # print("{}: time consume: {}".format(i, time.time()-now))

# Data clearning
for i in range(len(record_dates)):
    for j in range(len(station_ids)):
        if history_node_data[i,0,j] == -1:
            if i == 0:
                history_node_data[i,0,j] = 0
            else:
                history_node_data[i,0,j] = history_node_data[i-1,0,j]
        
        for k in range(1,8):
            if history_node_data[i,k,j] == -1:
                if j == 0:
                    history_node_data[i,k,j] = 0
                else:
                    history_node_data[i,k,j] = history_node_data[i,k,j-1]

# record_dates_len_3 = (len(record_dates)-1)/3
# history_node_data_3 = np.zeros(record_dates_len_3, 8, len(station_ids)), dtype=np.int8)
# # Data post-processing: average 3 times data, ie. using 12 15mins data to predict 3 15mins
# for i in range(len(record_dates)):


# print(history_node_data[:,-2])
np.save('./data/history_node_all_features_times_60.npy', history_node_data)

# 4. create adjacency matrix:
adjacency_matrix = np.zeros((len(station_ids),len(station_ids)))
for i in range(len(station_ids)):
    for j in range(len(station_ids)):
        if i==j:
            adjacency_matrix[i][j] = 0
        else:
            # adjacency_matrix[i][j] = ((position_LA[i]-position_LA[j])**2+ \
                # (position_LO[i]-position_LO[j])**2)**0.5
            adjacency_matrix[i][j] = 1/(((position_LA[i]-position_LA[j])**2+ \
                (position_LO[i]-position_LO[j])**2)**0.5)
np.save('./data/adjacency_matrix_reversed.npy', adjacency_matrix)

# Target1: nodes: history data, avilable bikes, available bike stands, bike stands
# Target2: adjacency matrix: node distances

# 0: station ID, 1: time, 4:bike stands, 5:available bike stands, 6: avilable bikes, 9,10:LA,LONG


