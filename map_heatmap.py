import folium
import pandas as pd
import numpy as np
import webbrowser
from folium.plugins import HeatMap
import xlrd
import branca

#绝对地址或同一目录下相对地址
file_name = "station_mae_location_new.xlsx"
file = xlrd.open_workbook(file_name)
sheet = file.sheet_by_name("Sheet1")
col_value0 = sheet.col_values(2)
col_value1 = sheet.col_values(3)

#获取经纬度数据，使用两个变量存储
LAT_new = col_value0  #纬度 latitude
LNG_new = col_value1  #经度 longitude
LOC = []
#此处必须使用zip构成元组
for lng,lat in zip(list(LNG_new),list(LAT_new)):
    LOC.append([lat, lng])

Center=[np.mean(np.array(LAT_new,dtype='float32')),np.mean(np.array(LNG_new,dtype='float32'))]
m=folium.Map(location=Center,zoom_start=8.5)
HeatMap(LOC).add_to(m)
# folium.map.LayerControl('topleft', collapsed=False).add_to(m)

colorlist = ['#3399FF','#66FFCC','#66FF66','#FFFF99','#FF9900','#FF3300']
colorbar = branca.colormap.StepColormap(colorlist,vmin = 0,vmax = 2.0, caption= 'MAE color indication')
m.add_child(colorbar)


#保存格式为html文件，可使用绝对路径进行保存
name='bike_heatmap_new.html'
m.save(name)

#将结果文件打开进行显示
webbrowser.open(name,new=2)
