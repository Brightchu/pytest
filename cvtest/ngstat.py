import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

times = np.array([])
ng_id = np.array([])
ng_rate_max = 0.06
ng_rate_step = 0.001
sample_date = "2020-04-13"
camera_sel = "5"
c1_c4_step = "step2"
# folder_path = r"E:\img\\"+sample_date+r"\Camera_" + camera_sel + r'\NG'
folder_path = "/media/bright/Repository/img/"+sample_date+"/Camera_"+camera_sel+"/NG"
ng_index_pre = 0
print(folder_path)
if camera_sel == "1" or camera_sel == "4":
    for path, subdirs, files in os.walk(folder_path):
        for filename in files:
            step_index = filename.find(c1_c4_step)
            jpeg_index = filename.find('jpeg')
            if step_index != -1 and jpeg_index != -1:
                index = filename.find('-') + 1
                date = filename[index - 5:index + 5]
                index = index + 6
                time = filename[index:index + 2] + ":" + filename[index + 3:index + 5] + ":" + filename[
                                                                                               index + 6:index + 8]
                index_pre = filename.find('step') + 6
                index_pro = filename.find('Success') - 1
                ng_index = filename[index_pre:index_pro]
                ng_index = int(ng_index)
                if ng_index_pre != ng_index:
                    ng_index_pre = ng_index
                    times = np.append(times, date + " " + time)
                    ng_id = np.append(ng_id, ng_index)
else:
    for path, subdirs, files in os.walk(folder_path):
        for filename in files:
            jpeg_index = filename.find('jpeg')
            if jpeg_index != -1:
                index = filename.find('-') + 1
                date = filename[index - 5:index + 5]
                index = index + 6
                time = filename[index:index + 2] + ":" + filename[index + 3:index + 5] + ":" + filename[
                                                                                               index + 6:index + 8]
                index_pro = filename.find('Success') - 1
                index_pre = filename.find('_', index_pro - 5) + 1
                ng_index = filename[index_pre:index_pro]
                if ng_index_pre != ng_index:
                    ng_index_pre = ng_index
                    times = np.append(times, date + " " + time)
                    ng_id = np.append(ng_id, ng_index)

times = pd.to_datetime(times)
interval = times - times[0]
# print(interval)

ng_id = ng_id.astype(np.int)
print(filename)
print(ng_index)
# print(ng_id)
# print(date)
# print(time)
# print("times = ")
# print(times)


interval = interval.total_seconds() / 60
interval = interval.to_numpy()
interval = interval.astype(int)

# print(interval)
# print(type(interval))
ng_length = len(interval)
print("ng number : ", ng_length)
NG_series = np.linspace(1, ng_length, ng_length)
NG_rate = np.array([])
total_num = np.array([])
ng_i_total_pre = 0
ng_step_num = 0
total_num_step = 0
for ng_i_current, ng_i_total_current in zip(NG_series, ng_id):
    if ng_i_total_current <= ng_i_total_pre:
        total_num_step = total_num_step + ng_i_total_pre
        ng_step_num = ng_i_current - 1
    NG_rate = np.append(NG_rate, (ng_i_current-ng_step_num)/ng_i_total_current)
    total_num = np.append(total_num, ng_i_total_current+total_num_step)
    ng_i_total_pre = ng_i_total_current
# plt.plot(interval, x)
# print(times[0])
# print(times[-1])
# print(NG_rate)

# my_x_ticks = np.arange(times[0], times[0]+pd.Timedelta('5 hours'), pd.Timedelta('30 minutes'))
# print(my_x_ticks)
########################################################################################
# NG NUM vs Time
########################################################################################

my_y_ticks = np.arange(0, ng_length+5, int(ng_length / 20)+1)
my_x_ticks = pd.date_range(times[0], times[-1], freq="30min")
# print("my_x_tick = \n", my_x_ticks)
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xticklabels([x.strftime('%H:%M') for x in my_x_ticks])
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.grid()
plt.plot(times, NG_series)
plt.xlabel('Time')
plt.ylabel('NG NUM')
if camera_sel == "1" or camera_sel == "4":
    title_prefix = "\n%s Camera%s %s" % (sample_date, camera_sel, c1_c4_step)
else:
    title_prefix = "\n%s Camera%s" % (sample_date, camera_sel)
plt.title('NG NUM vs Time'+title_prefix)
########################################################################################
# NG NUM vs Overall NUM
########################################################################################

# print(ng_id.dtype)
# print(ng_id[0])
# print(ng_id[-1])
# num_max = ng_id[-1]
fig2, ax2 = plt.subplots(figsize=(12, 8))
my_x_ticks = np.arange(0, np.max(ng_id), int(np.max(ng_id) / 20))
my_y_ticks = np.arange(0, ng_length+5, int(ng_length / 20)+1)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.grid()
plt.plot(ng_id, NG_series)
plt.xlabel('Overall NUM')
plt.ylabel('NG NUM')
plt.title('NG NUM vs Overall NUM'+title_prefix)
########################################################################################
# Overall NUM vs Time
#########################################################################################
fig4, ax4 = plt.subplots(figsize=(12, 8))
my_x_ticks = pd.date_range(times[0], times[-1], freq="30min")
ax4.set_xticklabels([x.strftime('%H:%M') for x in my_x_ticks])
my_y_ticks = np.arange(1, np.max(ng_id), int(np.max(ng_id) / 20))
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.plot(times, ng_id)
plt.grid()
plt.xlabel('Time')
plt.ylabel('Overall NUM')
plt.title('Overall NUM vs Time'+title_prefix)

########################################################################################
# NG Rate vs Overall NUM
########################################################################################
# print(ng_id.dtype)
# print(ng_id[0])
# print(ng_id[-1])
# num_max = ng_id[-1]
fig3, ax3 = plt.subplots(figsize=(12, 8))
my_x_ticks = np.arange(0, np.max(total_num)+5, int(np.max(total_num) / 20))
plt.xticks(my_x_ticks)
# my_y_ticks = np.arange(0, np.max(NG_rate)*1.1, (np.max(NG_rate)*1.1)/20)
my_y_ticks = np.arange(0, ng_rate_max, ng_rate_step)
plt.ylim((0, ng_rate_max))
ax3.set_yticklabels([("%.2f%%" % (x * 100)) for x in my_y_ticks])
# print("NG_rate max = ", np.max(NG_rate))
# print("NG_rate interval = ", (np.max(NG_rate)*1.1))
plt.yticks(my_y_ticks)
plt.grid()
plt.plot(total_num, NG_rate)
plt.xlabel('Overall NUM')
plt.ylabel('NG Rate')
plt.title('NG Rate vs Overall NUM'+title_prefix)

########################################################################################
# NG Rate vs NUM500
#########################################################################################
NG_rate_500 = np.array([])
for ng_idx, total_idx in zip(NG_series, ng_id):
    total_diff_500_idx = 0
    if total_idx > 500:
        total_diff_500_idx = np.min(np.argwhere(ng_id > (total_idx-500)))
    current_rate_500 = (ng_idx - NG_series[total_diff_500_idx])/(total_idx - ng_id[total_diff_500_idx]+1)
    NG_rate_500 = np.append(NG_rate_500, current_rate_500)
fig5, ax5 = plt.subplots(figsize=(12, 8))
my_x_ticks = np.arange(0, np.max(total_num)+5, int(np.max(total_num) / 20))
plt.xticks(my_x_ticks)
# my_y_ticks = np.arange(0, np.max(NG_rate_500)*1.1, (np.max(NG_rate_500)*1.1)/20)
my_y_ticks = np.arange(0, ng_rate_max, ng_rate_step)
plt.ylim((0, ng_rate_max))
ax5.set_yticklabels([("%.2f%%" % (x * 100)) for x in my_y_ticks])
# print("NG_rate max = ", np.max(NG_rate))
# print("NG_rate interval = ", (np.max(NG_rate)*1.1))
plt.yticks(my_y_ticks)
plt.grid()
plt.plot(total_num, NG_rate_500)
plt.xlabel('Overall NUM')
plt.ylabel('NG Rate')
plt.title('NG Rate vs NUM500'+title_prefix)
plt.show()
