
import math

RAD_DEG = 180 / math.pi


class arm(object):
    """一个五连杆并联机器人类，关节皆为转动副，专注运动模型

    Attributes:
        L: 机器人杆件长度组成的列表，前三个为支链杆件长度，第四个为支链原点所在圆的半径；
        tip_x_y_z: 机器人末端坐标系原点在全局坐标系下的坐标[x, y, z]；
        theta: 驱动关节角度，弧度制，关节角度相对于初始姿态逆时针为正，顺时针为负（正方向为使末端位置向上运动的方向）；
    Methods:
        __init__: 对象初始化函数；
        inverse_kinematics：根据末端件坐标系位姿求解关节角度的函数（逆解）；
        forward_kinematics_position: 根据关节角度求解末端点坐标系位姿函数（正解）；
    """
    L = [0, 0, 0, 0] # 前三个为支链杆件长度，第四个为支链原点所在圆的半径
    tip_x_y_z = [0, 0, -240] # 定义tip_x_y
    theta = [0, 0, 0] # 驱动关节角度，弧度制

    def __init__(self, l=[100, 100, 100, 100, 200]):
        """初始化函数，初始化机器人对象的属性参数 L

        Args:
            l: 机器人杆件长度组成的列表；
        Returns:
            无。
        Raises:
            无。
        """
        self.L = l[:]

    def inverse_kinematics(self, tip_x_y_z=[0, 0, 0], ud=0):
        """根据末端件坐标系原点位置求解关节角的函数，并以弧度制保存在theta列表中(运动学逆解)。

        Args:
            tip_x_y_z：末端点在全局坐标系中的坐标[x, y, z]
            ud: (up/down)用来选择逆解中的第几种解：
                ud = 0 时(默认值)，机器人第三个关节在末端点上方,对应课程中的alpha构型；
                ud = 1 时，机器人第三个关节在末端点下方，对应课程中的beta构型.
        Returns:
            True: 求解过程顺利通过时，返回True；
            False: 当求解过程出现异常或者末端点超出工作空间时，返回False。
        Raises:
            无。
        """

        theta0_bk = self.theta[0]
        theta1_bk = self.theta[1]
        theta2_bk = self.theta[2]
        x = tip_x_y_z[0] - 0 # 消除模型坐标与实际坐标的误差
        y = tip_x_y_z[1] - 0 # 消除模型坐标与实际坐标的误差
        z = tip_x_y_z[2] - 0 # 消除模型坐标与实际坐标的误差
        if z >= 0:
            print("z 轴坐标输入有误，请输入负值")
            return False
        l1 = self.L[0]
        l2 = self.L[1]
        R = self.L[2]
        r = self.L[3]
        M1 = math.sqrt(3)/2 * (R - r) - x
        N1 = -1/2 * (R - r) - y
        A1 = math.sqrt(3) * M1 - N1
        B1 = 2 * z
        C1 = - (M1 ** 2 + N1 ** 2 + z ** 2 + l1 ** 2 - l2 ** 2) / l1
        delta1 = A1 ** 2 + B1 ** 2 - C1 ** 2  # 设一个中间变量，以减少程序运算量
        if delta1 >= 0:
            if A1 == 0 and B1 == 0 and C1 == 0:
                self.theta[0] = theta0_bk # 此时机器人末端点在z轴上，关节1取前一次的值（即不变）
            else:
                if A1 + C1 == 0:
                    t1 = - (A1 - C1) / (2 * B1)
                else:
                    if ud == 0:
                        t1 = (B1 + math.sqrt(delta1)) / (A1 + C1) # 经实验选取该值
                    else:
                        t1 = (B1 - math.sqrt(delta1)) / (A1 + C1) # 经实验选取该值
                self.theta[0] = 2 * math.atan(t1)  # 求出seta1
        else:
            print('支链1末端位姿超出工作空间')
            return False
        M2 = - math.sqrt(3)/2 * (R - r) - x
        N2 = - 1/2 * (R - r) - y
        A2 = math.sqrt(3) * M2 + N2
        B2 = - 2 * z
        C2 = (M2 ** 2 + N2 ** 2 + z ** 2 + l1 ** 2 - l2 ** 2) / l1
        delta2 = A2 ** 2 + B2 ** 2 - C2 ** 2  # 设一个中间变量，以减少程序运算量
        if delta2 >= 0:
            if A2 == 0 and B2 == 0 and C2 == 0:
                self.theta[1] = theta1_bk  # 此时机器人末端点在z轴上，关节1取前一次的值（即不变）
            else:
                if A2 + C2 == 0:
                    t2 = - (A2 - C2) / (2 * B2)
                else:
                    if ud == 0:
                        t2 = (B2 - math.sqrt(delta2)) / (A2 + C2) # 经实验选取该值
                    else:
                        t2 = (B2 + math.sqrt(delta2)) / (A2 + C2) # 经实验选取该值
                self.theta[1] = 2 * math.atan(t2)  # 求出seta2
        else:
            print('支链2末端位姿超出工作空间')
            return False
        M3 = R - r - y
        A3 = M3
        B3 = z
        C3 = - (M3 ** 2 + x ** 2 + z ** 2 + l1 ** 2 - l2 ** 2) / l1 / 2
        delta3 = A3 ** 2 + B3 ** 2 - C3 ** 2  # 设一个中间变量，以减少程序运算量
        # print("delta3 = ", delta3)
        # print("A3 = ", A3)
        # print("B3 = ", B3)
        # print("C3 = ", C3)
        if delta3 >= 0:
            if A3 == 0 and B3 == 0 and C3 == 0:
                self.theta[2] = theta2_bk  # 此时机器人末端点在z轴上，关节1取前一次的值（即不变）
            else:
                if A3 + C3 == 0:
                    t3 = - (A3 - C3) / (2 * B3)
                else:
                    if ud == 0:
                        t3 = (B3 + math.sqrt(delta3)) / (A3 + C3) # 经实验选取该值
                    else:
                        t3 = (B3 - math.sqrt(delta3)) / (A3 + C3)  # 经实验选取该值
                self.theta[2] = 2 * math.atan(t3)  # 求出seta3
        else:
            print('支链3末端位姿超出工作空间')
            return False
        return True

    def forward_kinematics_position(self, angle_list=[0, 0, 0]):
        """根据关节角求解末端点位置和姿态函数(运动学正解)

        Args:
            angle_list: 关节角度组成的列表[joint1, joint2, joint3]（角度制）
        Returns:
            无
        Raises:
            无
        """

        tip_x_y_z_bk = self.tip_x_y_z.copy()
        angle = [math.pi / 180 * i for i in angle_list]
        seta1 = angle[0]
        seta2 = angle[1]
        seta3 = angle[2]
        l1 = self.L[0]
        l2 = self.L[1]
        R = self.L[2]
        r = self.L[3]
        x_D1 = math.sqrt(3)/2 * (R - r + l1 * math.cos(seta1))
        y_D1 = -1/2 * (R - r + l1 * math.cos(seta1))
        z_D1 = -l1 * math.sin(seta1)
        x_D2 = - math.sqrt(3) / 2 * (R - r + l1 * math.cos(seta2))
        y_D2 = -1/2 * (R - r + l1 * math.cos(seta2))
        z_D2 = -l1 * math.sin(seta2)
        x_D3 = 0
        y_D3 = R - r + l1 * math.cos(seta3)
        z_D3 = -l1 * math.sin(seta3)
        M1 = x_D1 ** 2 + y_D1 ** 2 + z_D1 ** 2 - l2 ** 2
        M2 = x_D2 ** 2 + y_D2 ** 2 + z_D2 ** 2 - l2 ** 2
        M3 = x_D3 ** 2 + y_D3 ** 2 + z_D3 ** 2 - l2 ** 2
        N1 = (z_D1 - z_D3) * (x_D1 - x_D2) - (z_D1 - z_D2) * (x_D1 - x_D3)
        N2 = (z_D1 - z_D3) * (y_D1 - y_D2) - (z_D1 - z_D2) * (y_D1 - y_D3)
        N3 = 1/2 * ((z_D1 - z_D3) * (M1 - M2) - (z_D1 - z_D2) * (M1 - M3))
        N4 = (y_D1 - y_D3) * (x_D1 - x_D2) - (y_D1 - y_D2) * (x_D1 - x_D3)
        N5 = (y_D1 - y_D3) * (z_D1 - z_D2) - (y_D1 - y_D2) * (z_D1 - z_D3)
        N6 = 1 / 2 * ((y_D1 - y_D3) * (M1 - M2) - (y_D1 - y_D2) * (M1 - M3))
        if N2 == 0 or N5 == 0:
            self.tip_x_y_z[0] = self.tip_x_y_z[0] # 尚未得到解析解，暂用内存中的数据，进行位置控制后该情况将与实际相符
            self.tip_x_y_z[1] = self.tip_x_y_z[1] # 尚未得到解析解，暂用内存中的数据，进行位置控制后该情况将与实际相符
            self.tip_x_y_z[2] = self.tip_x_y_z[2] # 尚未得到解析解，暂用内存中的数据，进行位置控制后该情况将与实际相符
        else:
            R1 = - N1 / N2
            R2 = N3 / N2
            R3 = - N4 / N5
            R4 = N6 / N5
            A = 1 + R1 ** 2 + R3 ** 2
            B = 2 * (R1 * R2 + R3 * R4 - R1 * y_D1 - R3 * z_D1 - x_D1)
            C = R2 ** 2 + R4 ** 2 + M1 - 2 * (R2 * y_D1 + R4 * z_D1)
            delta = B ** 2 - 4 * A * C
            if delta >= 0:
                if A == 0 and B == 0 and C == 0:
                    self.tip_x_y_z[0] = tip_x_y_z_bk[0]  # 此时机器人末端点在z轴上，关节1取前一次的值（即不变）
                else:
                    if A == 0:
                        t = - C / B
                        self.tip_x_y_z[0] = t  # 求出x
                        self.tip_x_y_z[1] = R1 * self.tip_x_y_z[0] + R2
                        self.tip_x_y_z[2] = R3 * self.tip_x_y_z[0] + R4
                    else:
                        t = (- B + math.sqrt(delta)) / (2 * A)
                        self.tip_x_y_z[0] = t  # 求出x
                        self.tip_x_y_z[1] = R1 * self.tip_x_y_z[0] + R2
                        self.tip_x_y_z[2] = R3 * self.tip_x_y_z[0] + R4
                        if self.tip_x_y_z[2] >= 0:
                            t = (- B - math.sqrt(delta)) / (2 * A)  # 经实验选取该值
                            self.tip_x_y_z[0] = t  # 求出x
                            self.tip_x_y_z[1] = R1 * self.tip_x_y_z[0] + R2
                            self.tip_x_y_z[2] = R3 * self.tip_x_y_z[0] + R4
            else:
                print("无解")
                return False
        return True






