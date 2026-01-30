import time
import math as cm
import DrDelta as robot
import DrEmpower_can as dr
max_list_temp = [90, 90, 90]  # 关节模型角度最大值
min_list_temp = [-42, -42, -42]  # 关节模型角度最小值
l_temp = [100, 250, 35, 23.4]  # 机器人尺寸参数：[l1 l2 R r] 详见库函数说明
# # 机器人对象初始化函数函数
ro = robot.robot(MAX_list_temp=max_list_temp, MIN_list_temp=min_list_temp, L_temp=l_temp)
#
# # 运动到指定位置函数************************************
# ro.set_position(tip_x_y_z=[0, 0, -210], speed=10, acceleration=10)
# ro.set_position(tip_x_y_z=[31.593, 29.837, -211.099], speed=10, acceleration=10)
# ro.set_position(tip_x_y_z=[-34.62, -0.531, -197.888], speed=10, acceleration=10)
# # *********************************************************
#
# # 运动到相对位置函数*****************************************
# ro.set_relative_position(tip_x_y_z=[0, 0, 15], speed=1, acceleration=1)
# *********************************************************

# # 等待运动到位函数*****************************************
# ro.position_done()
# *********************************************************

# ####### 轨迹跟踪函数***********************************************
# ####画正方形#################
# def draw_rectangle(pl=[283, 0, -240], l=30, h=1):
#     ''''在水平面上画正方形
#     pl: 长方形左上角坐标（起始点），其中pl[2]代表作图平面与全局坐标系z轴的焦点的z坐标
#     l: 宽度
#     h: 高度
#     '''
#     n= 550 # 每条边分割的点数（数量越多，曲线越标准，但画得越慢）
#     l_delta = l/n
#     h_delta = h/n
#     pl_list = []
#     pl_list.append(pl)
#     # l1 = pl[1]
#     for i in range(1, n+1):
#         pl_temp = [pl[0]+i*l_delta, pl[1], pl[2]]
#         pl_list.append(pl_temp)
#     print(pl_temp)
#     for i in range(1, n+1):
#         pl_temp1 = [pl_temp[0], pl_temp[1]-i*h_delta, pl[2]]
#         pl_list.append(pl_temp1)
#     print(pl_temp1)
#     for i in range(1, n+1):
#         pl_temp2 = [pl_temp1[0]-i*l_delta, pl_temp1[1], pl[2]]
#         pl_list.append(pl_temp2)
#     print(pl_temp2)
#     for i in range(1, n+1):
#         pl_temp3 = [pl_temp2[0], pl_temp2[1]+i*h_delta, pl[2]]
#         pl_list.append(pl_temp3)
#     print(pl_temp3)
#     print(pl_list)
#     return pl_list
# #
# #
# pl_list = draw_rectangle(pl=[-100, 100, -190], l=200, h=200) #
# # ########控制机械臂末端连续运动到多个指定位置和姿态函数(必须单独一次性使用)
# # ro.set_positions(tip_x_y_zs=pl_list, t=5) # 控制机械臂末端连续运动到多个指定位置和姿态函数(必须单独一次性使用)
# ro.set_positions_curve_pre(tip_x_y_zs=pl_list) # 预设机械臂末端轨迹函数
# ro.set_position(tip_x_y_z=[0, 0, -200], speed=10, acceleration=10)
# ro.position_done()
# ro.set_positions_curve_start_point(10) # 运动到轨迹起始位置函数
# while True:
#     ro.set_positions_curve_do(10) # 末端轨迹执行函数，参数为大致运行时间

############################################
##############################################
#####画椭圆#################
# def draw_ellipse(pl=[0, 0, -240], a=10, b=10):
#     ''''椭圆方程: (x-pl[0])²/a²+(y-pl[1])²/b²=1
#         pl: 椭圆中心点坐标,其中pl[2]代表作图水平面在z轴上的位置
#         a: x轴对应轴长
#         b: x轴对应轴长y
#         '''
#     n = 800 # 每条边分割的点数（数量越多画得越慢）, n过小会在末尾与起始之间有明显停顿
#     angle_delta = cm.pi/n * 2
#     pl_list = []
#     for i in range(0, n+1):
#         x = pl[0] + a * cm.cos(angle_delta*i)
#         y = pl[1] + b * cm.sin(angle_delta*i)
#         pl_list.append([x, y, pl[2]])
#     print(pl_list)
#     return pl_list

# ro.set_position(tip_x_y_z=[0, 0, -200], speed=10, acceleration=10)
# ro.position_done()

# for i in range(20):
#     pl_list = draw_ellipse(pl=[0, 0, -210], a=50, b=100)  # 求点
#     ro.set_positions_curve_pre(tip_x_y_zs=pl_list)  # 预设机械臂末端轨迹函数
#     ro.set_positions_curve_start_point(10)  # 运动到轨迹起始位置函数
#     ro.detect_position()
#     ro.set_positions_curve_do(5)  # 末端轨迹执行函数，参数为大致运行时间

# for i in range(3):
#     ro.set_positions_curve_do(5)  # 末端轨迹执行函数，参数为大致运行时间
# while True:
#     ro.set_positions_curve_do(5) # 末端轨迹执行函数，参数为大致运行时间
# # ************************************************************

# 控制机器人关节角度函数*******************************************
# ro.set_joints(angle_list=[0, 0, 0], speed=10, acceleration=10)
# *********************************************************

# 查看内存中当前位置函数*******************************************
# ro.show_position()
# *********************************************************

# 查看实际中各关节模型角度（通过回读关节电机角度）*******************************************
# ro.detect_joints()
# *********************************************************

# 查看实际中当前位置函数（通过回读关节电机角度）*******************************************
# ro.detect_position()
# *********************************************************

# 查看关节参数*******************************************
# ro.read_property(joint_num=1, property='axis.motor.config.current_control_bandwidth') # 电流控制带宽
# *********************************************************

# 查看关节PID*******************************************
# ro.read_pid(joint_num=1)
# *********************************************************

# 设置关节参数*******************************************
# ro.set_property(joint_num=1, property='axis.motor.config.current_control_bandwidth', value=20) # 电流控制带宽
# *********************************************************

# 设置关节pid*******************************************
# ro.set_pid(joint_num=1, P=20, I=20, D=0.5)
# *********************************************************

# 保存参数*******************************************
# ro.save_config()
# *********************************************************

# 恢复出厂参数*******************************************
# ro.init_config()
# *********************************************************

# 设置零点姿态（谨慎！！确认好零点姿态之后再使用）*******************************************
# ro.set_zero_pose()
# *********************************************************

# # 画图 *****************************************
# import xlrd # 需要在 windows 终端中运行 pip install xlrd==1.2.0 来安装这个库，不要在pycharm软件中直接安装（因为安装的版本较高会导致识别excel文件报错）
# # import xlwt
#
# z = -256
# trans = [0, -50] # 工件坐标系相对全局坐标系的偏移量（默认两个坐标系坐标轴之间相互平行）
# workBook = xlrd.open_workbook('作图/hua.xls')
# sheet1_content1 = workBook.sheet_by_index(0) # sheet索引从0开始，标记sheet1的内容
# row_num = sheet1_content1.nrows # 数据行数
# col_num = sheet1_content1.ncols # 数据列数
# rows = [] # 记录每一行数据的列表
# for i in range(row_num-1): # 去掉第一行数据
#     rows.append(sheet1_content1.row_values(i+1))
#     # print(sheet1_content1.row_values(i+1))
# print(rows)  # 打印出表格数据
# for j in range(len(rows)):
#     rows[j][0] = rows[j][0] * 1 + trans[0]
#     rows[j][1] = rows[j][1] * 1 + trans[1]
# print(rows) # 打印转换后的数据
# print(len(rows)) # 打印转换后的数据
# rows_new = []
# for j in range(len(rows)):
#     rows_new.append([rows[j][0], rows[j][1], z])
# print(rows_new) # 打印转换后的数据
# ro.set_positions_curve_pre(tip_x_y_zs=rows_new) # 预设机器人末端轨迹函数
# while True:
#     ro.set_position(tip_x_y_z=[0, 0, -200], speed=10, acceleration=10)  # 控制机器人末端运动到一定高度的安全位置，避免笔尖在纸面划动
#     ro.position_done()
#     ro.set_positions_curve_start_point(10)  # 运动到轨迹起始位置函数
#     ro.set_positions_curve_do(5)  # 5-20

# # *********************************************************
# # 单独测试关节运动以读取关节电机编号************************************
# ro.set_joints(angle_list=[0, 0, 0], speed=30, acceleration=50)
# ro.position_done()
# ro.set_joints(angle_list=[30, 0, 0], speed=30, acceleration=50)
# ro.position_done()
# ro.set_joints(angle_list=[30, 30, 0], speed=30, acceleration=50)
# ro.position_done()
# ro.set_joints(angle_list=[30, 30, 30], speed=30, acceleration=50)
# ro.position_done()


ro.set_joints(angle_list=[0, 0, 0], speed=10, acceleration=10)