import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

times = np.array([])
ng_id = np.array([])
for path, subdirs, files in os.walk(r'E:\img\2020-04-10\Camera_5\NG'):
    for filename in files:
        jpeg_index = filename.find('jpeg')
        if jpeg_index != -1:
            index = filename.find('-') + 1
            date = filename[index - 5:index + 5]
            index = index + 6
            time = filename[index:index + 2] + ":" + filename[index + 3:index + 5] + ":" + filename[index + 6:index + 8]
            times = np.append(times, date+" "+time)
            index_pro = filename.find('Success') - 1
            index_pre = filename.find('_', index_pro-5) + 1
            ng_index = filename[index_pre:index_pro]
            ng_id = np.append(ng_id, ng_index)
print(filename)
print(ng_index)
times = pd.to_datetime(times)
interval = times - times[0]
# print(interval)

ng_id = ng_id.astype(np.int)

print(ng_id)
# print(date)
# print(time)
print("times = ")
print(times)


interval = interval.total_seconds()/60
interval = interval.to_numpy()
interval = interval.astype(int)

print(interval)
# print(type(interval))
length = len(interval)
y = np.linspace(1, length, length)
# plt.plot(interval, x)
print(times[0])
print(times[0]+pd.Timedelta('5 hours'))

# my_x_ticks = np.arange(times[0], times[0]+pd.Timedelta('5 hours'), pd.Timedelta('30 minutes'))
# print(my_x_ticks)

my_y_ticks = np.arange(0, length, int(length/20))
my_x_ticks = pd.date_range(times[0], times[-1], freq="30min")
print("my_x_tick = ")
print(my_x_ticks)
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xticklabels([x.strftime('%H:%M') for x in my_x_ticks])
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.grid()
plt.plot(times, y)
plt.xlabel('Time')
plt.ylabel('NG NUM')

#######################################################################################

# print(ng_id.dtype)
# print(ng_id[0])
# print(ng_id[-1])
# num_max = ng_id[-1]
fig2, ax2 = plt.subplots(figsize=(12, 8))
my_x_ticks = np.arange(1, np.max(ng_id), int(np.max(ng_id)/20))
my_y_ticks = np.arange(0, length, int(length/20))
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.grid()
plt.plot(ng_id, y)
plt.xlabel('Overall NUM')
plt.ylabel('NG NUM')

#########################################################################################
print(np.max(ng_id))
fig3, ax3 = plt.subplots(figsize=(12, 8))
my_x_ticks = pd.date_range(times[0], times[-1], freq="30min")
ax3.set_xticklabels([x.strftime('%H:%M') for x in my_x_ticks])
my_y_ticks = np.arange(1, np.max(ng_id), int(np.max(ng_id)/20))
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.plot(times, ng_id)
plt.grid()
plt.xlabel('Time')
plt.ylabel('Overall NUM')
plt.show()
