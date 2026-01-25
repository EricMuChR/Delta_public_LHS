#!/usr/bin/python3

import Delta_3 as am
import DrEmpower_can as dr
import math
import os
import time

current_path = os.path.dirname(os.path.abspath(__file__))

# 角度和弧度转换
DEG_RAD = math.pi / 180
RAD_DEG = 180 / math.pi
P2_LIST_PATH = current_path + '/p2_list.txt'


class robot(am.arm):
    """五连杆并联机器人类（继承arm类）

    Attributes:
        c_angle_list: 机器人所有关节当前模型角度（角度制）组成的列表；
        position_list: 用来存放机器人动作序列的列表；
        ID_list: 机器人关节所对应编号组成的列表；
        P1_list: 机器人关节模型角度与电机机实际角度转换的第一个系数的列表；两者的变换关系为：舵机角度=P1*模型角度+P2（模型关节轴线与电机转动轴线同向为1，反向为-1）
        P2_list: 机器人关节模型角度与电机实际角度转化的第二个系数的列表；两者的变换关系为：舵机角度=P1*模型角度+P2（电机处于模型初始位置安装时对应的实际角度）
        MAX_list: 由机器人各个关节模型角最大值组成的列表；
        MIN_list: 由机器人各个关节模型角最小值组成的列表；
        init_model_angle: 机器人安装位置各关节模型角度值（固定值，不可改变）；
        max_speed: 机器人最大运动速度，一般取各个一体化关节中转速最低的那个；
        tutorial_t：示教编程中位姿停留时间
        tutorial_t_list：每个位姿的停留时间组成的列表
        bit_time：系统发送一次运动指令所需的时间
        torque_limits：机器人各一体化关节最大限制力矩组成的列表
        tip_x_y_z_start：轨迹规划起始点坐标
    Methods:
        __init__: 对象初始化函数；
        ###############运动控制函数######################
        set_position: 控制机器人末端运动到指定位置函数（不改变姿态角）
        set_relative_position：控制机器人相对当前位姿运动一定距离函数（不改变姿态角）
        set_positions：控制机器人连续运动多个位置和姿态函数，即执行一定轨迹
        set_positions_curve_pre：预先设置机器人末端运动轨迹函数
        set_positions_curve_start_point：控制机器人运动到预设轨迹起点的函数，需与在set_arm_positions_curve_pre之后使用
        set_positions_curve_do：控制机器人执行预设轨迹函数
        set_joints：控制机器人各个关节运动到指定模型角度函数
        ###############数据回读函数####################
        show_position：根据内存中的模型角度反馈机器人当前位姿函数
        detect_position：根据回读的电机角度反馈机器人当前位姿函数
        ###############辅助功能函数####################
        save_position: 基于机器人当前末端位姿求解各关节模型角度，并保存position_list中；
        model_to_servo: 将position_list中保存的机器人关节模型角度转换成电机实际角度；
        do_motion: 动作执行函数，让机器人依次执行之前保存的动作；
        range_init: 机器人关节转动范围设置函数，设置的是模型角度范围；
        add_position: 将当前机器人姿态保存到position_list中
        read_joints：读取机器人所有一体化关节当前角度；
        read_joints_p_v：读取机器人所有一体化关节当前角度和转速
        servo_to_model: 将读取的舵机角度转换成关节模型角度；
        free: 将所有关节设置成待机模式函数，便于手动掰动关节，注意待机之前需要手扶；
        lock：将所有关节固定在当前角度函数，让机器人保持在当前位置，保持刚度；
        clear_position: 删除position_list中某个position或清空所有position；
        position_done：检查并等待机器人运动到指定位置
    """
    c_angle_list = []
    position_list = []
    servo_angle_list_list = []
    position_list_servo = []
    position_list_temp = []
    ID_list = [1, 2, 3]
    P1_list = [1, 1, 1]  # 对应一体化关节输出轴与模型中z轴方向一致则为1，不一致则为-1
    P2_list = [0, 0, 0] # 电机在机器人初始位置安装时的实际角度，本机器人关节皆为在零位安装
    MAX_list = [90, 90, 90]  # 关节模型角度最大值
    MIN_list = [-42, -42, -42]  # 关节模型角度最小值
    init_model_angle = [0, 0, 0]
    max_speed = 30
    tutorial_t = 1 # 示教编程中位姿停留时间
    tutorial_t_list = [0] # 每个位姿的停留时间组成的列表
    bit_time = 0 # 系统发送一次运动指令所需的时间
    torque_limits = [] # 机器人各一体化关节最大限制力矩组成的列表
    tip_x_y_z_start = [0, 0, 0] # 轨迹规划起始点坐标

    # 机器人初始化函数
    def __init__(self, MAX_list_temp=[], MIN_list_temp=[], L_temp=[100, 250, 35, 23.4]):
        """机器人对象初始化函数，初始化机器人对象的属性参数

        Args:
            MAX_list_temp: 机器人关节模型角度最大值组成的列表
            MIN_list_temp: 机器人关节模型角度最小值组成的列表
            L_temp：机器人尺寸参数：[l1 l2 R r] 详见库函数说明
        Returns:
            无
        Raises:
            无
        """

        am.arm.__init__(self, l=L_temp)
        self.range_init(MAX_list_temp, MIN_list_temp) # 关节模型角度范围初始化
        servo_angle_list = self.read_joints() # 读取机器人一体化关节实际角度
        if servo_angle_list != False:
            self.c_angle_list = self.servo_to_model(servo_angle_list=servo_angle_list) # 根据读取到的电机角度初始化关节模型角度（角度制）
        else:
            self.c_angle_list = self.init_model_angle[:] # 若一体化关节读取失败则关节模型角度取默认值
        self.theta = [DEG_RAD * i for i in self.c_angle_list].copy() # 初始化关节模型角度（弧度制）
        self.forward_kinematics_position(self.c_angle_list) # 根据初始化的关节模型角度计算初始的末端位置tip_x_y_z和姿态phi
        position = [self.theta[0] / math.pi * 180, self.theta[1] / math.pi * 180, self.theta[2] / math.pi * 180]
        self.position_list.append(position) # 记录初始关节模型角度
        position_list_servo_temp = self.read_joints() # 再读取一次一体化关节角度，以进行后面的操作
        dr.set_angles(self.ID_list, position_list_servo_temp, 20, 20, 0)
        start = time.time()
        for k in range(len(self.ID_list)):
            dr.set_angles(self.ID_list, position_list_servo_temp, 20, 20, 0)
        self.bit_time = (time.time() - start) / (k + 1)  # 计算得到系统发送一次位置控制指令所需时间
        print('初始化成功')
        time.sleep(0.2)

    # 一、机器人位置和姿态控制函数
    def set_position(self, tip_x_y_z=[0, 0, 0], speed=1.0, acceleration=10):
        """控制机器人运动到指定位置和姿态

        Args:
            tip_x_y_z: 机器人末端位置坐标组成的列表[x, y, z]
            speed: 当前姿态被执行时转速最快关节的转动速度
            acceleration：关节转动加速度大小
        Returns:
            无
        Raises:
            无
        """
        tip_x_y_z_bk = self.tip_x_y_z.copy()  # 记录此前末端位置坐标，此处不能直接使用 tip_x_y_z_bk = self.tip_x_y_z 否则 tip_x_y_z_bk 会始终随着 self.tip_x_y_z 变化
        self.tip_x_y_z = tip_x_y_z.copy()  # 将目标位置赋值给对象位置属性 tip_x_y_z
        self.clear_position()
        if self.save_position(tip_x_y_z=tip_x_y_z):  # 运动学逆解并将结果保存进 position_list
            self.do_motion(speed=speed, acceleration=acceleration)  # 执行动作
        else:
            self.tip_x_y_z = tip_x_y_z_bk.copy()  # 若运动学逆解失败，则恢复此前的末端位置坐标

    def set_relative_position(self, tip_x_y_z=[0, 0, 0], speed=1.0, acceleration=10):
        """控制机器人运动到相对当前位置的指定位置，不改变机器人角姿态

        Args:
            tip_x_y_z: 机器人末端相对部坐标组成的列表[x, y, z]
            speed: 当前姿态被执行时转速最快关节的转动速度
            acceleration：关节转动加速度大小
        Returns:
            无
        Raises:
            无
        """
        tip_x_y_z_bk = self.tip_x_y_z.copy()  # 记录此前末端坐标，此处不能直接使用 tip_x_y_z_bk = self.tip_x_y_z 否则 tip_x_y_z_bk 会始终随着 self.tip_x_y_z 变化
        for i in range(len(tip_x_y_z)):
            self.tip_x_y_z[i] = self.tip_x_y_z[i] + tip_x_y_z[i]  # 机器人末端坐标值加上相对坐标值
        tip_x_y_z = self.tip_x_y_z.copy() # 将更新后的末端坐标赋值给tip_x_y_z
        self.clear_position()
        if self.save_position(tip_x_y_z=tip_x_y_z):  # 运动学逆解，并将结果存进 position_list
            self.do_motion(speed=speed, acceleration=acceleration)  # 执行动作
        else:
            self.tip_x_y_z = tip_x_y_z_bk.copy()  # 若运动学逆解失败，则恢复此前的末端姿态角
    #
    # 二、机器人轨迹控制函数
    def set_positions(self, tip_x_y_zs=[], t=1):
        """控制机器人末端按顺序连续运动到多个指定位置和姿态

        Args:
            tip_x_y_zs: 机器人末端连续多个坐标组成的列表[[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], ...]
            t: 运动执行的大致时间
        Returns:
            无
        Raises:
            无
        """

        n = len(tip_x_y_zs)
        self.clear_position()
        for i in range(n): # 将输入的坐标值和姿态角全部带入逆解函数求解出关节模型角度
            tip_x_y_z = tip_x_y_zs[i].copy() # 将轨迹点赋值给 tip_x_y_z
            tip_x_y_z_bk = self.tip_x_y_z.copy()  # 记录此前末端坐标值，此处不能直接使用 tip_x_y_z_bk = self.tip_x_y_z 否则 tip_x_y_z_bk 会始终随着 self.tip_x_y_z 变化
            self.tip_x_y_z = tip_x_y_z.copy()  # 记录此后末端坐标值
            if not self.save_position(tip_x_y_z=tip_x_y_z):  # 运动学逆解，并将求解结果存进 position_list
                self.tip_x_y_z = tip_x_y_z_bk.copy()  # 如果逆解失败则取此前记录的末端坐标值
        self.position_list_servo = []  # position_list_servo 在此置空，用于存放计算得到的一体化关节实际角度列表
        for j in range(len(self.position_list)):
            self.position_list_servo.append(self.model_to_servo(model_angle_list=self.position_list[j])) # 将关节模型角度转换成一体化关节实际角度并添加进 position_list_servo
        self.set_position(tip_x_y_z=tip_x_y_zs[0], speed=10)
        self.position_done()
        n = len(self.position_list_servo)
        if t >= n * self.bit_time: # 轨迹要求的执行时间大于系统发送 n 个动作指令（所有轨迹点）所需的时间
            print('t >= n * bit_time') # 提示
            bit_wideth = n / t / 2 # 指令发送带宽的一半
            start = time.time() # 记录指令发送开始时间
            dr.set_angles(self.ID_list, self.position_list_servo[0], 20, bit_wideth, 0) # 采用轨迹跟踪模式发送首条位置指令，根据轨迹跟踪模式要求，bit_wideth 需为实际指令发送带宽的一半
            while (time.time() - start) < (t / n): # 将轨迹执行时间按照点的数量均分
                time.sleep(0.001)
            bit_wideth1 = 1 / (time.time() - start) / 2 # 计算在 t>n 情况下的指令发送频率的一半
            for k in range(n):
                start = time.time()
                dr.set_angles(self.ID_list, self.position_list_servo[k], 20, bit_wideth1, 0) # 采用轨迹跟踪模式，根据轨迹跟踪模式要求，bit_wideth1 需为实际指令发送带宽的一半
                while (time.time() - start) < (t / n):
                    time.sleep(0.001)
                bit_wideth1 = 1 / (time.time() - start) / 2 # 时刻监控在 t>n 情况下单条指令发送的时间
        else:
            print('t < n * bit_time') # 提示轨迹要求的时间小于系统发送 n 个指令（所有轨迹点）所需的时间
            bit_wideth = 1 / self.bit_time / 2 # 指令发送频率取初始化阶段获得的系统单条指令发送所需时间的倒数，再取其一半作为 set_angles() 函数参数
            for k in range(n):
                dr.set_angles(self.ID_list, self.position_list_servo[k], 20, bit_wideth, 0) # 使用轨迹追踪模式控制机器人运动
    #
    def set_positions_curve_pre(self, tip_x_y_zs=[]):
        """预先计算机器人按顺序连续运动到多个指定位置所对应的关节角度

        Args:
            tip_x_y_zs: 机器人末端连续多个坐标组成的列表[[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], ...]
        Returns:
            无
        """

        n = len(tip_x_y_zs)
        self.clear_position()
        self.tip_x_y_z_start = tip_x_y_zs[0].copy() # 记录轨迹初始位置
        for i in range(n):
            tip_x_y_z = tip_x_y_zs[i].copy() # 将轨迹点赋值给 tip_x_y_z
            tip_x_y_z_bk = self.tip_x_y_z.copy()  # 记录此前末端坐标值，此处不能直接使用 tip_x_y_z_bk = self.tip_x_y_z 否则 tip_x_y_z_bk 会始终随着 self.tip_x_y_z 变化
            self.tip_x_y_z = tip_x_y_z.copy()  # 记录此前末端坐标值
            if not self.save_position(tip_x_y_z=tip_x_y_z):  # 运动学逆解，并将求解结果存进 position_list
                self.tip_x_y_z = tip_x_y_z_bk.copy()  # 如果逆解失败则取此前记录的末端坐标值
        self.position_list_servo = []  # position_list_servo 在此置空，用于存放计算得到的一体化关节实际角度列表
        for j in range(len(self.position_list)):
            self.position_list_servo.append(self.model_to_servo(model_angle_list=self.position_list[j]))

    def set_positions_curve_start_point(self, speed=10):
        """控制机器人运动到轨迹中的首个位置, 用于 set_arm_positions_cure_pre 之后

        Args:
            speed: 机器人动作执行时转速最快关节的转动速度
        Returns:
            无
        Raises:
            无
        """

        if speed > self.max_speed:
            speed = self.max_speed # 最大速度限制，保证安全性和动作一致性
        self.set_position(tip_x_y_z=self.tip_x_y_z_start, speed=speed) # 运动到初始位置
        # print(self.tip_x_y_z_start)
        self.position_done() # 监控并等待动作执行结束

    def set_positions_curve_do(self, t=1):
        """控制机器人按顺序连续运动到多个指定位置和姿态, 与 set_arm_positions_cure_pre 连用

        Args:
            t: 用来指定当前姿态被执行时的大致时长
        Returns:
            无
        Raises:
            无
        """
        for i in range(len(dr.cur_angle_list)):
            DETA_angle_list = list(
                map(lambda x: abs(x[0] - x[1]),
                    zip(self.position_list_servo[0], dr.cur_angle_list)))
        if max(DETA_angle_list) > 10.0:  # 如果轨迹起点离当前位置比较远，则自动调用set_positions_curve_start_point(speed=10)，否则无需调用
            self.set_positions_curve_start_point(speed=10) # 防止用户忘记调用 set_positions_curve_start_point(self, speed)
        n = len(self.position_list_servo)
        if t >= n * self.bit_time: # 轨迹要求的执行时间大于系统发送 n 个动作指令（所有轨迹点）所需的时间
            print('t >= n * bit_time')  # 提示
            bit_wideth = n / t / 2 # 指令发送带宽的一半
            start = time.time() # 记录指令发送开始时间
            dr.set_angles(self.ID_list, self.position_list_servo[0], 20, bit_wideth, 0) # 采用轨迹跟踪模式发送首条位置指令，根据轨迹跟踪模式要求，bit_wideth 需为实际指令发送带宽的一半
            while (time.time() - start) < (t / n): # 将轨迹执行时间按照点的数量均分
                time.sleep(0.001)
            bit_wideth1 = 1 / (time.time() - start) / 2 # 计算在 t>n 情况下的指令发送频率的一半
            for k in range(1, n):   # 前面已经发送过轨迹起点位置，这里无需重复，所以从1开始
                start = time.time()
                dr.set_angles(self.ID_list, self.position_list_servo[k], 20, bit_wideth1, 0) # 采用轨迹跟踪模式，根据轨迹跟踪模式要求，bit_wideth1 需为实际指令发送带宽的一半
                while (time.time() - start) < (t / n):
                    time.sleep(0.001)
                bit_wideth1 = 1 / (time.time() - start) / 2 # 时刻监控在 t>n 情况下单条指令发送的时间
        else:
            print('t < n * bit_time')  # 提示轨迹要求的时间小于系统发送 n 个指令（所有轨迹点）所需的时间
            bit_wideth = 1 / self.bit_time / 2 # 指令发送频率取初始化阶段获得的系统单条指令发送所需时间的倒数，再取其一半作为 set_angles() 函数参数
            for k in range(n):
                dr.set_angles(self.ID_list, self.position_list_servo[k], 20, bit_wideth, 0) # 使用轨迹追踪模式控制机器人运动

    # 三、机器人示教编程函数
    def tutorial_program(self):
        """在记录机器人连续姿态信息，以实现示教编程

        Args:
            无
        Returns:
            示教过程中，1~6号关节电机角度组成的列表的列表
        Raises:
            无
        """

        self.servo_angle_list_list = []
        self.free() # 设置一体化关节为待机状态，以便掰动
        while 1:
            servo_angle_list = self.read_joints()  # 读取并记录关节电机角度
            self.servo_angle_list_list.append(servo_angle_list) # 将读取到的角度添加到关节电机角度列表的列表中
            print(self.servo_angle_list_list) # 显示一体化关节位置数据
            print('n: ', len(self.servo_angle_list_list)) # 显示一体化关节位置列表长度

    def tutorial_do(self, t=0):
        """执行机器人连续姿态（轨迹）

        Args:
            t: 轨迹执行的大致时间
        Returns:
            无
        Raises:
            无
        """

        if t <= 0:
            t = self.tutorial_t
        n = len(self.servo_angle_list_list) # 记录动作序列的长度
        if t >= n * self.bit_time: # 轨迹要求的执行时间大于系统发送 n 个动作指令（所有轨迹点）所需的时间
            print('t >= n * bit_time')  # 提示
            bit_wideth = n / t / 2 # 指令发送带宽的一半
            servo_angle_list = self.read_joints()
            time.sleep(0.01)
            dr.set_angles(self.ID_list, servo_angle_list, 20, 20, 1) # 将关节设置为位置闭环控制模式
            dr.set_angles(self.ID_list, self.servo_angle_list_list[0], 20, 20, 1)
            self.position_done()
            dr.set_angles(self.ID_list, self.servo_angle_list_list[0], 20, bit_wideth, 0) # 采用轨迹跟踪模式发送首条位置指令，根据轨迹跟踪模式要求，bit_wideth 需为实际指令发送带宽的一半
            start = time.time()  # 记录指令发送开始时间
            while (time.time() - start) < (t / n): # 将轨迹执行时间按照点的数量均分
                time.sleep(0.001)
            bit_wideth1 = 1 / (time.time() - start) / 2 # 计算在 t>n * self.bit_time 情况下的指令发送频率的一半
            start_time = time.time()
            for k in range(n):
                start = time.time()
                dr.set_angles(self.ID_list, self.servo_angle_list_list[k], 20, bit_wideth1, 0) # 采用轨迹跟踪模式，根据轨迹跟踪模式要求，bit_wideth1 需为实际指令发送带宽的一半
                while (time.time() - start) < (t / n):
                    time.sleep(0.001)
                bit_wideth1 = 1 / (time.time() - start) / 2 # 时刻监控在 t>n 情况下单条指令发送的时间
            print('运行时长：', time.time() - start_time)
        else:
            print('t < n * bit_time')  # 提示轨迹要求的时间小于系统发送 n 个指令（所有轨迹点）所需的时间
            bit_wideth = 1 / self.bit_time / 2 # 指令发送频率取初始化阶段获得的系统单条指令发送所需时间的倒数，再取其一半作为 set_angles() 函数参数
            for k in range(n):
                dr.set_angles(self.ID_list, self.servo_angle_list_list[k], 20, bit_wideth, 0) # 使用轨迹追踪模式控制机器人运动

    def add_position(self, t=0):
        """读取到的当前机器人所有关节角度并转换成模型角度，然后保存到position_list中。

        Args:
            t: 该姿态与下一姿态之间的时间间隔（即姿态保持时间）
        Returns:
            True or False ；
        Raises:
            无；
        """

        servo_list = self.read_joints() # 读取一体化关节角度
        if servo_list != False:
            self.position_list.append(self.servo_to_model(servo_angle_list=servo_list)) # 将一体化关节角度转换成关节模型角度，并添加进 position_list 中
            self.tutorial_t_list.append(t) # 设置该姿态与下一姿态之间的时间间隔（即姿态保持时间）
            print("保存当前姿态成功！ 当前position_list中共有" + str(len(self.position_list)) + "个position")
            return True
        else:
            print("当前姿态读取失败，请再试一次！")
            return False

    def do_position(self, speed=10, acceleration=10, o_r=0, n=0):
        """依次执行机器人通过示教保存position；

        Args:
            speed: 当前姿态被执行时转速最快关节的转动速度
            acceleration：关节加速度大小
            o_r: 用来选择执行的顺序，
                o_r =0：从前往后执行
                o_r =1: 从后往前执行
            n: 用来控制执行的细节，如果n=0，这执行所有保存的姿态；如果n>0 则执行第n个姿态
        Returns:
            无
        Raises:
            无
        """
        self.do_motion(speed=speed, acceleration=acceleration, o_r=o_r, n=n, flg=1)
    #
    # 四、机器人关节角度控制函数
    def set_joints(self, angle_list=[0, 0, 0], speed=1.0, acceleration=10):
        """控制机器人关节运动到指定模型角度

        Args:
            angle_list: 机器人三个关节模型角度组成的列表[joint1, joint2, joint3]
            speed: 当前姿态被执行时转速最快关节的转动速度
        Returns:
            无
        Raises:
            无
        """

        self.clear_position()
        if len(angle_list) == len(self.ID_list):
            position = angle_list[:]
            self.theta = [math.pi / 180 * i for i in angle_list] # 更新内存中的关节模型角度（弧度制）
            for i in range(len(position)):
                if position[i] < self.MIN_list[i]:
                    print("第" + str(i + 1) + "个关节角度超出了最小极限角度") # 检查关节模型角度是否小于最小允许值
                    return False
                if position[i] > self.MAX_list[i]:
                    print("第" + str(i + 1) + "个关节角度超出了最大极限角度") # 检查关节模型角度是否大于最大允许值
                    return False
            self.position_list.append(position) # 将输入的关节角度列表保存进 position_list
            self.tutorial_t_list.append(0) # 指定该组姿态与下组姿态的时间间隔为 0
            self.do_motion(speed=speed, acceleration=acceleration) # 执行动作
            self.forward_kinematics_position(angle_list) # 使用运动学正解计算指定机器人关节模型角度后的位置和姿态
            return True
        else:
            print("角度参数有误！")
            return False
    #
    # 五、机器人位置参数回读函数
    def show_position(self):
        """根据内存中的关节模型角度值，显示当前末端位置和姿态

        Args:
            无
        Returns:
            末端位置和姿态坐标
        Raises:
            无
        """

        model_angle_list = [RAD_DEG * i for i in self.theta] # 用于安全观察，因此使用内存参数快速计算
        self.forward_kinematics_position(model_angle_list) # 调用运动学正解函数
        print("当前机器人末端 x 坐标为: " + str(self.tip_x_y_z[0]))
        print("当前机器人末端 y 坐标为: " + str(self.tip_x_y_z[1]))
        print("当前机器人末端 z 坐标为: " + str(self.tip_x_y_z[2]))
        return self.tip_x_y_z

    def detect_joints(self):
        """根据读取到的一体化关节角度值，显示当前关节模型角度

        Args:
            无
        Returns:
            机器人关节模型角度
        Raises:
            无
        """

        servo_angle_list = self.read_joints() # 读取各一体化关节角度（用于查询未知姿态，因此读取关节角度）
        model_angle_list = self.servo_to_model(servo_angle_list=servo_angle_list) # 将一体化关节角度转换为关节模型角度
        print("关节 1 模型角度为: " + str(model_angle_list[0]))
        print("关节 2 模型角度为: " + str(model_angle_list[1]))
        print("关节 3 模型角度为: " + str(model_angle_list[2]))
        return model_angle_list

    def detect_position(self):
        """根据读取到的一体化关节角度值，显示当前末端位置和姿态

        Args:
            无
        Returns:
            末端位置和姿态坐标
        Raises:
            无
        """

        servo_angle_list = self.read_joints() # 读取各一体化关节角度（用于查询未知姿态，因此读取关节角度）
        model_angle_list = self.servo_to_model(servo_angle_list=servo_angle_list) # 将一体化关节角度转换为关节模型角度
        self.forward_kinematics_position(model_angle_list) # 调用运动学正解函数
        print("当前机器人末端 x 坐标为: " + str(self.tip_x_y_z[0]))
        print("当前机器人末端 y 坐标为: " + str(self.tip_x_y_z[1]))
        print("当前机器人末端 z 坐标为: " + str(self.tip_x_y_z[2]))
        return self.tip_x_y_z

    # 六、关节阻抗控制函数
    def impedance_control_joints(self, vel=1, tff=0, kp=5, kd=0.1):
        """设置机器人每个关节的阻抗系数

        Args:
            vel: 关节目标速度(r/min)
            tff: 前馈扭矩(Nm)
            kp: 刚度系数(rad/Nm)
            kd: 阻尼系数(rad/s/Nm)
        Returns:
            1~5号电机扭矩
        Raises:
            无
        """

        for j in range(len(self.ID_list)): # 提前 position_list 中的最后一组关节模型角度列表（即机器人最后姿态），设置此姿态下关节阻抗系数
            dr.impedance_control(id_num=self.ID_list[j], pos=self.model_to_servo(self.position_list[len(self.position_list) - 1])[j],
                                 vel=vel, tff=tff, kp=kp, kd=kd)

    # 七、功能辅助函数
    def save_position(self,  tip_x_y_z=[0, 0, -240]):
        """求解并保存当前机器人姿态下各个关节模型角度，并将角度制的角度列表添加进 position_list

        Args:
            tip_x_y_z：末端在全局坐标系下的坐标值[x, y, z]
        Returns:
            True: 逆解过程顺利通过时，返回True
            False: 机器人末端超出工作空间、机器人末端超出安全范围、关节模型角度超出安全范围
        Raises:
            无
        """

        theta_bk = self.theta.copy()
        if not self.inverse_kinematics(tip_x_y_z=tip_x_y_z):  # 在对应支链中进行运动学逆解
            self.theta = theta_bk.copy() # 记录内存中此前的驱动关节模型角度（弧度制）
            return False
        position = [self.theta[0] / math.pi * 180, self.theta[1] / math.pi * 180, self.theta[2] / math.pi * 180]
        for i in range(len(position)):
            if position[i] < self.MIN_list[i]:
                print("第" + str(i + 1) + "个关节角度超出了最小极限角度")  # 检查关节模型角度是否小于最小允许值
                return False
            if position[i] > self.MAX_list[i]:
                print("第" + str(i + 1) + "个关节角度超出了最大极限角度")  # 检查关节模型角度是否大于最大允许值
                return False
        self.position_list.append(position) # 记录目标位姿下驱动关节角度（角度制）
        self.tutorial_t_list.append(0) # 位姿停留时间，单位秒
        return True

    # 动作执行
    def do_motion(self, speed=1.0, acceleration=10, o_r=0, n=0, flg=0):
        """依次执行机器人此前保存的position；

        Args:
            speed: 当前姿态被执行时转速最快关节的转动速度
            acceleration：关节加速度大小
            o_r: 用来选择执行的顺序，
                o_r =0：从前往后执行
                o_r =1: 从后往前执行
            n: 用来控制执行的细节，如果n=0，这执行所有保存的姿态；如果n>0 则执行第n个姿态
            flg: 用于判定是否需要确保每一个姿态都执行到位，1代表是，0代表否
        Returns:
            无
        Raises:
            无
        """

        # 最大速度限制，保证安全性和动作一致性
        if speed > self.max_speed:
            speed = self.max_speed
        try:
            if n == 0:
                if o_r == 0:
                    for i in range(len(self.position_list)):
                        dr.set_angles(self.ID_list, self.model_to_servo(self.position_list[i]), speed, acceleration, 1) # 依次执行 position_list 中的position（姿态）
                        if flg == 1:
                            self.position_done()  # 等待并监控动作执行结束
                        time.sleep(self.tutorial_t_list[i])  # 设置每个姿态的停留时长
                else:
                    for j in range(len(self.position_list)):
                        i = len(self.position_list) - 1 - j  # 倒序执行
                        dr.set_angles(self.ID_list, self.model_to_servo(self.position_list[i]), speed, acceleration, 1) # 倒序执行 position_list 中的position（姿态）
                        if flg == 1:
                            self.position_done()  # 等待并监控动作执行结束
                        time.sleep(self.tutorial_t_list[i])  # 设置每个姿态的停留时长
            else:
                dr.set_angles(self.ID_list, self.model_to_servo(self.position_list[n-1]), speed, acceleration, 1)
                if flg == 1:
                    self.position_done()  # 等待并监控动作执行结束
                time.sleep(self.tutorial_t_list[n-1])  # 设置每个姿态的停留时长
        except Exception as result:
            print('检测出异常 in do_motion{}'.format(result))
    # def do_motion(self, speed=1.0, acceleration=10, o_r=0, n=0):
    #     """依次执行机器人此前保存的position；
    #
    #     Args:
    #         speed: 当前姿态被执行时转速最快关节的转动速度
    #         acceleration：关节转动加速度大小
    #         mode: 控制电机转动模式，0为轨迹追踪模式，1为梯形轨迹模式，2为前馈模式
    #         o_r: 用来选择执行的顺序，
    #             o_r =0：从前往后执行
    #             o_r =1: 从后往前执行
    #         n: 用来控制执行的细节，如果n=0，这执行所有保存的姿态；如果n>0 则执行第n个姿态
    #     Returns:
    #         无
    #     Raises:
    #         无
    #     """
    #
    #     # 最大速度限制，保证安全性和动作一致性
    #     if speed > self.max_speed:
    #         speed = self.max_speed
    #     try:
    #         if n == 0:
    #             if o_r == 0:
    #                 for i in range(len(self.position_list)):
    #                     dr.set_angles(self.ID_list, self.model_to_servo(self.position_list[i]), speed, acceleration, 1) # 依次执行 position_list 中的position（姿态）
    #                     # self.position_done() # 等待并监控动作执行结束，默认不开
    #                     time.sleep(self.tutorial_t_list[i])  # 设置每个姿态的停留时长
    #             else:
    #                 for j in range(len(self.position_list)):
    #                     i = len(self.position_list) - 1 - j  # 倒序执行
    #                     dr.set_angles(self.ID_list, self.model_to_servo(self.position_list[i]), speed, acceleration, 1) # 倒序执行 position_list 中的position（姿态）
    #                     # self.position_done() # 等待并监控动作执行结束，默认不开
    #                     time.sleep(self.tutorial_t_list[i])  # 设置每个姿态的停留时长
    #         else:
    #             dr.set_angles(self.ID_list, self.model_to_servo(self.position_list[n-1]), speed, acceleration, 1)
    #             # self.position_done() # 等待并监控动作执行结束，默认不开
    #             time.sleep(self.tutorial_t_list[n-1])  # 设置每个姿态的停留时长
    #     except Exception as result:
    #         print('检测出异常 in do_motion{}'.format(result))

    # 关节角度范围设置
    def range_init(self, max_list=[], min_list=[]):
        """设置机器人各个关节的模型角度范围[min, max]

        Args:
            max_list: 由机器人所有关节转动范围最大值组成的列表
            min_list: 由机器人所有关节转动范围最小值组成的列表
        Returns:
            无；
        Raises:
            无；
        """

        servo_number = len(self.ID_list)
        if len(max_list) == servo_number:
            self.MAX_list = max_list[:]
        else:
            self.MAX_list = [90, 215, 159]
        if len(min_list) == servo_number:
            self.MIN_list = min_list[:]
        else:
            self.MIN_list = [-36, -15, -159]
#
    # 读取一体化关节角度
    def read_joints(self):
        """读取机器人当前姿态下各关节的电机角度。

        Args:
            无。
        Returns:
            False or servo_list.
            若有一个或者多个关节读取出错，则返回False, 反之，如果一切正常，则返回所有关节角度组成的列表。
        Raises:
            无。
        """

        servo_list = []
        flag = []
        for i in range(len(self.ID_list)):
            servo = dr.get_state(id_num=self.ID_list[i]) # 使用一体化关节的 get_state() 函数依次读取关节角度和速度
            if servo != False:
                servo_list.append(servo[0]) # 将读取到的角度添加进 servo_list
            else:
                servo = dr.get_state(id_num=self.ID_list[i]) # 如果首次读取失败就再读一次
                if servo != False:
                    servo_list.append(servo[0]) # 将读取到的角度添加进 servo_list
                else:
                    flag.append(self.ID_list[i]) # 如果持续读取失败则记录读取失败的一体化关节编号
                    print("ID号为：" + str(self.ID_list[i]) + "的电机读取角度失败！")
        if len(flag) != 0:
            return False
        return servo_list

    def read_joints_p_v(self):
        """读取机器人当前姿态下各关节的角度和转速。

        Args:
            无。
        Returns:
            False or servo_list.
            若有一个或者多个关节读取出错，则返回False, 反之，如果一切正常，则返回所有关节角度和速度组成的列表。
        Raises:
            无。
        """
        servo_p_v_list = []
        flag = []
        for i in range(len(self.ID_list)):
            servo = dr.get_state(id_num=self.ID_list[i]) # 使用一体化关节的 get_state() 函数依次读取关节角度和速度
            if servo != False:
                servo_p_v_list.append(servo) # 将读取到的角度和速度添加进 servo_p_v_list
            else:
                servo = dr.get_state(id_num=self.ID_list[i]) # 如果首次读取失败就再读一次
                if servo != False:
                    servo_p_v_list.append(servo) # 将读取到的角度和速度添加进 servo_p_v_list
                else:
                    flag.append(self.ID_list[i]) # 如果持续读取失败则记录读取失败的一体化关节编号
                    print("ID号为：" + str(self.ID_list[i]) + "的电机取角度和速度失败！")
        if len(flag) != 0:
            return False
        return servo_p_v_list

    # 角度转换
    def servo_to_model(self, servo_angle_list=[]):
        """将机器人一体化关节角度转换成模型角度。
        将当前姿态的舵机角度转换成模型角度，,两者之间的变换关系为
        舵机角度=P1*模型角度+P2，其中P1表示两者的正方向是否相同，相同为1，相反为-1
        第二个参数为差值项，装配时将初始模型角度和舵机角度代入求出。

        Args:
            servo_angle_list: 机器人某个姿态下所有一体化关节角度（角度制）组成的列表
        Returns:
            该姿态下所有关节模型角度组成的列表；
        Raises:
            无
        """

        if servo_angle_list != False:
            model_angle_list = servo_angle_list[:] # 目的是让 servo_angle_list 拥有与 model_angle_list 相同的长度
            for i in range(len(servo_angle_list)):
                model_angle_list[i] = round((servo_angle_list[i] - self.P2_list[i]) / self.P1_list[i], 1) # 通过关节模型角度与一体化关节角度之间的关系进行转换
            return model_angle_list
        else:
            return False

    def model_to_servo(self, model_angle_list=[]):
        """将机器人关节模型角度转换成一体化关节角度。

        Args:
            model_angle_list: 机器人某个姿态下所有关节模型角度（角度制）组成的列表
        Returns:
            该姿态下所有关机电机角度组成的列表；
        Raises:
            无
        """

        servo_angle_list = model_angle_list[:] # 目的是让 servo_angle_list 拥有与 model_angle_list 相同的长度
        for i in range(len(model_angle_list)):
            servo_angle_list[i] = self.P1_list[i] * servo_angle_list[i] + self.P2_list[i] # 通过关节模型角度与一体化关节角度之间的关系进行转换
        return servo_angle_list
#
    # 将所有关节设置成待机模式
    def free(self):
        """将机器人所有一体化关节设置成待机模式，以便节省能源。若要恢复使用必须先调用 lock()函数。
        Args:
            无；
        Returns:
            无；
        Raises:
            无；
        """

        dr.set_mode(0, 1)
#
    # 将所有关节固定在当前位置
    def lock(self):
        """将机器人所有一体化关节设置成锁死模式，保持当前姿态。

        Args:
            无；
        Returns:
            无；
        Raises:
            无；
        """

        dr.set_mode(0, 2)
#
    # 删除或清空position_list里的动作
    def clear_position(self, n=0):
        """清除机器人position_list中保存的姿态。
        当需要重新计算一个轨迹或新姿态时，如果不想保留原有的轨迹或姿态，可以调用此函数将原有保存的所有姿态清空。

        Args:
            n: 指定需要清除的position编号
                n=0: 表示清空position_list中所有position
                n>0: 删除第n个position
                n<0, 删除倒数第n个position
        Returns:
            无。
        Raises:
            无。
        """

        LEN = len(self.position_list)
        if n == 0:
            self.position_list = [] # 直接置空
            self.tutorial_t_list = [] # 直接置空
        elif n > 0:
            if n <= LEN:
                del self.position_list[n - 1] # 删除第 n 个姿态
                del self.tutorial_t_list[n - 1] # 删除第 n 个姿态保持的时间
            else:
                print("元素引用序号超出position_list长度")
        else:
            if LEN + n >= 0:
                del self.position_list[LEN + n] # 删除倒数第 n 个姿态
                del self.tutorial_t_list[LEN + n] # 删除倒数第 n 个姿态保持的时间
            else:
                print("元素引用序号超出position_list长度")

    def position_done(self):
        """检查并等待机器人末端运动到指定位置和姿态。

        Args:
            无。
        Returns:
            无
        Raises:
            无。
        """

        dr.positions_done(self.ID_list)
        # dr.position_done(id_num=self.ID_list[0])

    def read_property(self, joint_num=1, property=''):
        """读取机器人关节参数。

        Args:
            joint_num：关节编号。
             property：参数名称（见于 paramter_interface.py 文件）。
        Returns:
             property：参数的值。
        Raises:
            无。
        """

        if joint_num < 1 or joint_num > 3:
            print("请输入正确的关节编号：1~3")
            return False
        else:
            property = dr.read_property(id_num=self.ID_list[joint_num-1], property=property)
            print(property)
            return property

    def read_pid(self, joint_num=1):
        """读取机器人关节 PID。

        Args:
            joint_num：关节编号。
        Returns:
                  pid：关节 P 值 I 值与 D 值组成的列表 [P, I, D]。
        Raises:
            无。
        """
        if joint_num < 1 or joint_num > 3:
            print("请输入正确的关节编号：1~3")
            return False
        else:
            pid = dr.get_pid(id_num=self.ID_list[joint_num-1])
            return pid

    def set_property(self, joint_num=1, property='', value=0):
        """设置机器人关节参数。

        Args:
            joint_num：关节编号。
             property：参数名称（见于 paramter_interface.py 文件）。
                value：要设置的参数的值。
        Returns:
                 True：设置成功。
                False：设置失败。
        Raises:
            无。
        """

        if joint_num < 1 or joint_num > 3:
            print("请输入正确的关节编号：1~3")
            return False
        else:
            dr.write_property(id_num=self.ID_list[joint_num-1], property=property, value=value)
            time.sleep(0.1)
            value_return = dr.read_property(id_num=self.ID_list[joint_num - 1], property=property)
            if value_return == value:
                print("机器人第 ", joint_num, " 号关节的 ", str(property) + " 修改为：", value_return)
                return True
            else:
                print("机器人第 ", joint_num, " 号关节的 ", str(property) + " 修改失败，请重试")
                return False

    def set_pid(self, joint_num=1, P=10, I=10, D=10):
        """设置机器人关节 PID。

        Args:
            joint_num：关节编号。
                    P：PID 参数的 P 值。
                    I：PID 参数的 I 值。
                    D：PID 参数的 D 值。
        Returns:
                 True：设置成功。
                False：设置失败。
        Raises:
            无。
        """

        if joint_num < 1 or joint_num > 3:
            print("请输入正确的关节编号：1~3")
            return False
        else:
            dr.set_pid(id_num=self.ID_list[joint_num - 1], P=P, I=I, D=D)
            time.sleep(0.1)
            pid = dr.get_pid(id_num=self.ID_list[joint_num - 1])
            if pid == [P, I, D]:
                print("机器人第 ", joint_num, " 号关节的 PID 修改为：", pid)
                return True
            else:
                print("机器人第 ", joint_num, " 号关节的 PID 修改失败，请重试")
                return False

    def save_config(self):
        """保存机器人关节参数。

        Args:
            无。
        Returns:
            无。
        Raises:
            无。
        """

        time.sleep(0.2)
        dr.save_config(0)

    def init_config(self):
        """恢复机器人关节参数出厂设置。

        Args:
            无。
        Returns:
            无。
        Raises:
            无。
        """

        time.sleep(0.2)
        dr.init_config(0)

    def set_zero_pose(self):
        """设置机器人零点姿态（谨慎！！确认好零点姿态之后再使用）。

        Args:
            无。
        Returns:
            True：设置成功。
           False：设置失败。
        Raises:
            无。
        """

        time.sleep(0.2)
        dr.set_zero_position(0)
        angle = 0
        for i in range(len(self.ID_list)):
            angle += dr.get_angle(self.ID_list[i])
        if angle < 0.01:
            print("零点姿态设置成功")
            return True
        else:
            print("零点姿态设置失败，请重试")
            return False