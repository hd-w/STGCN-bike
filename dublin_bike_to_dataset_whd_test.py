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



# 2.1 dublin bike data timestamp remove second
def remove_second(x):
    return x["TIME"][0:16]
df["TIME"]=df.apply(remove_second, axis=1)

station_ids = list(df["STATION ID"].drop_duplicates().values)
position_LA = list(df["LATITUDE"].drop_duplicates().values)
position_LO = list(df["LONGITUDE"].drop_duplicates().values)
record_dates = list(df["TIME"].drop_duplicates().values)
# print(record_dates.index('2020-07-01 00:05'))
print(len(record_dates))#26752
print(len(station_ids))#110
print(station_ids)#110




