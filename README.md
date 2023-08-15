## **数据说明**

**注意：虽然都是包含时间戳的数据，但是粒度不一样，需要进一步处理；同时建议将“性别”这一项也作为数据加入。**

### **1.ACC.csv:三轴加速度计数据**

数据将包括“时间戳”作为日期时间值，加速度计数据将用于“X”、“Y”、“Z”方向

datetime, acc_x, acc_y, acc_z

例子：2020-02-13 15:28:50.000000, -34.0,17.0, 55.0


### **2.BVP.csv:血容量脉搏数据**

数据将包括“时间戳”作为日期时间值，“血容量脉搏值”作为当时记录的测量值。

datetime, bvp

例子：2020-02-13 15:28:50.000000,-0.0



### **3.Dexcom.csv:间质葡萄糖浓度数据(数据具有许多的缺失值)**

数据将包括“时间戳”作为日期时间值和“间质葡萄糖浓度值”作为当时记录的测量值。

Index,
Timestamp (YYYY-MM-DDThh:mm:ss),
Event Type,
Event Subtype,
Patient Info,
Device Info,
Source Device ID,
Glucose Value (mg/dL),     **#这个指标比较关键**
Insulin Value (u),
Carb Value (grams),
Duration (hh:mm:ss),
Glucose Rate of Change (mg/dL/min),
Transmitter Time (Long Integer)      #

例子：13,2020-02-13 17:23:32,EGV,,,,iPhone G6,61.0,,,,,11101.0

**数据主要包含：下标、时间戳、项目类型（都是EGV）、Source Device ID（都是iPhone G6）、葡萄糖浓度值(mg/dL)、注射时间（？）**

### **4.EDA.csv:皮肤电活动数据**

数据将包括“时间戳”作为日期时间值和“皮肤电活动值”作为当时记录的测量值。

datetime, eda

例子：2020-02-13 15:28:50.250,0.001281

### **5.Food_Log.csv:参与者在整个研究过程中消耗的食物日志（估计不是很重要）**

数据将包括“日期”作为日期值，“time_of_day”作为时间值，
“time_begin”作为日期时间值，“time_end”作为日期时间值，
“logged_food”作为字符串值，“金额”作为数值，“单位”作为字符串值，
“searched_food”作为字符串值，“卡路里”作为数值，“total_carb”作为数值，
“dietary_fiber”、“糖”作为数值，“蛋白质”作为数值，“total_fat”作为数值

date,
time,
time_begin,
time_end,
logged_food,
amount,
unit,
searched_food,
calorie,
total_carb,
dietary_fiber,
sugar,
protein,
total_fat

例子：2020-02-13,18:00:00,2020-02-13 18:00:00,,Berry Smoothie,
20.0,fluid ounce,Strawberry Smoothie,456.0,85.0,1.7,83.0,16.0,3.3

### **6.HR.csv:心率数据**

数据将包括“时间戳”作为日期时间值和“指心率值”作为当时记录的测量值。

datetime, hr

例子：2020-07-12 15:29:00,94.0

### **7.IBI.csv:心跳间隔数据**

数据将包括“时间戳”作为日期时间值和“指心跳间隔值”作为当时记录的测量值。

datetime, ibi

例子：2020-02-13 15:33:22.059328,0.8281629999999999

### **8.TEMP.csv:皮肤温度数据**

数据将包括“时间戳”作为日期时间值和“皮肤温度值”作为当时记录的测量值。

datetime, temp

例子：2020-02-13 15:28:50.000,30.21


### ACC 1/32 s (20296428, 4)  无空值
### BVP 1/64 s (40592838, 2) 无空值
### Dexcom 5 min (2573, 13) 大量空值
### EDA 1/4 s (2537046, 2) 无空值
### Food_Log 时间段不固定 (61, 14) 部分空值
### HR 1 s (634188, 2) 无空值
### IBI 时间段不固定 (266366, 2) 无空值
### TEMP 1/4 (2537040, 2) 无空值

#Food_Log的数据不是血液里的平衡量，而是增量



#高低糖数据分布，分为16个文件
150 2411
827 1292
145 2156
161 2002
88 2470
698 2149
73 2133
235 2269
624 1680
266 1881
677 2165
405 1763
506 1473
335 1904
15 477
153 1981