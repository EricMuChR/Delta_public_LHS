# Axis.AxisState
AXIS_STATE_UNDEFINED = 0
AXIS_STATE_IDLE = 1
AXIS_STATE_STARTUP_SEQUENCE = 2
AXIS_STATE_FULL_CALIBRATION_SEQUENCE = 3
AXIS_STATE_MOTOR_CALIBRATION = 4
AXIS_STATE_SENSORLESS_CONTROL = 5
AXIS_STATE_ENCODER_INDEX_SEARCH = 6
AXIS_STATE_ENCODER_OFFSET_CALIBRATION = 7
AXIS_STATE_CLOSED_LOOP_CONTROL = 8
AXIS_STATE_LOCKIN_SPIN = 9
AXIS_STATE_ENCODER_DIR_FIND = 10
AXIS_STATE_HOMING = 11

# Controller.ControlMode
CONTROL_MODE_VOLTAGE_CONTROL = 0
CONTROL_MODE_TORQUE_CONTROL = 1
CONTROL_MODE_VELOCITY_CONTROL = 2
CONTROL_MODE_POSITION_CONTROL = 3
CONTROL_MODE_IMPEDANCE_CONTROL = 4

# Controller.InputMode
INPUT_MODE_INACTIVE = 0
INPUT_MODE_PASSTHROUGH = 1
INPUT_MODE_VEL_RAMP = 2
INPUT_MODE_POS_FILTER = 3
INPUT_MODE_MIX_CHANNELS = 4
INPUT_MODE_TRAP_TRAJ = 5
INPUT_MODE_TORQUE_RAMP = 6
INPUT_MODE_MIRROR = 7

# Motor.MotorType
MOTOR_TYPE_HIGH_CURRENT = 0
MOTOR_TYPE_GIMBAL = 2
MOTOR_TYPE_ACIM = 3

# Encoder.Mode
ENCODER_MODE_INCREMENTAL = 0
ENCODER_MODE_HALL = 1
ENCODER_MODE_SINCOS = 2
ENCODER_MODE_SPI_ABS_CUI = 256
ENCODER_MODE_SPI_ABS_AMS = 257
ENCODER_MODE_SPI_ABS_AEAT = 258
ENCODER_MODE_SPI_ABS_RLS = 259
ENCODER_MODE_SPI_ABS_EDR = 271

# Axis.Error
AXIS_ERROR_NONE = 0x00000000
AXIS_ERROR_INVALID_STATE = 0x00000001
AXIS_ERROR_DC_BUS_UNDER_VOLTAGE = 0x00000002
AXIS_ERROR_DC_BUS_OVER_VOLTAGE = 0x00000004
AXIS_ERROR_CURRENT_MEASUREMENT_TIMEOUT = 0x00000008
AXIS_ERROR_BRAKE_RESISTOR_DISARMED = 0x00000010
AXIS_ERROR_MOTOR_DISARMED = 0x00000020
AXIS_ERROR_MOTOR_FAILED = 0x00000040
AXIS_ERROR_SENSORLESS_ESTIMATOR_FAILED = 0x00000080
AXIS_ERROR_ENCODER_FAILED = 0x00000100
AXIS_ERROR_CONTROLLER_FAILED = 0x00000200
AXIS_ERROR_POS_CTRL_DURING_SENSORLESS = 0x00000400
AXIS_ERROR_WATCHDOG_TIMER_EXPIRED = 0x00000800
AXIS_ERROR_MIN_ENDSTOP_PRESSED = 0x00001000
AXIS_ERROR_MAX_ENDSTOP_PRESSED = 0x00002000
AXIS_ERROR_ESTOP_REQUESTED = 0x00004000
AXIS_ERROR_UNKOWN_ERROR_CODE1 = 0x00008000
AXIS_ERROR_UNKOWN_ERROR_CODE2 = 0x00010000
AXIS_ERROR_HOMING_WITHOUT_ENDSTOP = 0x00020000
AXIS_ERROR_OVER_TEMP = 0x00040000

# Motor.Error
MOTOR_ERROR_NONE = 0x00000000
MOTOR_ERROR_PHASE_RESISTANCE_OUT_OF_RANGE = 0x00000001
MOTOR_ERROR_PHASE_INDUCTANCE_OUT_OF_RANGE = 0x00000002
MOTOR_ERROR_ADC_FAILED = 0x00000004
MOTOR_ERROR_DRV_FAULT = 0x00000008
MOTOR_ERROR_CONTROL_DEADLINE_MISSED = 0x00000010
MOTOR_ERROR_NOT_IMPLEMENTED_MOTOR_TYPE = 0x00000020
MOTOR_ERROR_BRAKE_CURRENT_OUT_OF_RANGE = 0x00000040
MOTOR_ERROR_MODULATION_MAGNITUDE = 0x00000080
MOTOR_ERROR_BRAKE_DEADTIME_VIOLATION = 0x00000100
MOTOR_ERROR_UNEXPECTED_TIMER_CALLBACK = 0x00000200
MOTOR_ERROR_CURRENT_SENSE_SATURATION = 0x00000400
MOTOR_ERROR_UNKOWN_ERROR_CODE = 0x00000800
MOTOR_ERROR_CURRENT_LIMIT_VIOLATION = 0x00001000
MOTOR_ERROR_BRAKE_DUTY_CYCLE_NAN = 0x00002000
MOTOR_ERROR_DC_BUS_OVER_REGEN_CURRENT = 0x00004000
MOTOR_ERROR_DC_BUS_OVER_CURRENT = 0x00008000
MOTOR_ERROR_STALL_LIMIT_VIOLATION = 0x00010000

# Encoder.Error
ENCODER_ERROR_NONE = 0x00000000
ENCODER_ERROR_UNSTABLE_GAIN = 0x00000001
ENCODER_ERROR_CPR_POLEPAIRS_MISMATCH = 0x00000002
ENCODER_ERROR_NO_RESPONSE = 0x00000004
ENCODER_ERROR_UNSUPPORTED_ENCODER_MODE = 0x00000008
ENCODER_ERROR_ILLEGAL_HALL_STATE = 0x00000010
ENCODER_ERROR_INDEX_NOT_FOUND_YET = 0x00000020
ENCODER_ERROR_ABS_SPI_TIMEOUT = 0x00000040
ENCODER_ERROR_ABS_SPI_COM_FAIL = 0x00000080
ENCODER_ERROR_ABS_SPI_NOT_READY = 0x00000100
ENCODER_ERROR_EDR_OVER_RANGE = 0x00000200
ENCODER_ERROR_EDR_POWER_FAILURE = 0x00000400
ERROR_EDR_BATTERY_LOW = 0x00000800,
ERROR_OFFSET_CALIBRATE_FAILURE = 0x00001000,

# Controller.Error
CONTROLLER_ERROR_NONE = 0x00000000
CONTROLLER_ERROR_OVERSPEED = 0x00000001
CONTROLLER_ERROR_INVALID_INPUT_MODE = 0x00000002
CONTROLLER_ERROR_UNSTABLE_GAIN = 0x00000004
CONTROLLER_ERROR_INVALID_MIRROR_AXIS = 0x00000008
CONTROLLER_ERROR_INVALID_LOAD_ENCODER = 0x00000010
CONTROLLER_ERROR_INVALID_ESTIMATE = 0x00000020

# Can.Error
CAN_ERROR_NONE = 0x00000000
CAN_ERROR_DUPLICATE_CAN_IDS = 0x00000001

# Fet_thermistor.Error
THERMISTOR_CURRENT_LIMITER_ERROR_NONE = 0x00000000
THERMISTOR_CURRENT_LIMITER_ERROR_OVER_TEMP = 0x00000001

# Motor_thermistor.Error
MOTOR_CURRENT_LIMITER_ERROR_NONE = 0x00000000
MOTOR_CURRENT_LIMITER_ERROR_OVER_TEMP = 0x00000001

N = 1


def plus_plus(n=0):
    global N
    if n == 0:
        N = 1
    else:
        N = N + 1
    return N


property_addresss = {
    #     #
    'vbus_voltage': 00000 + plus_plus(0), # 只读
    'ibus': 00000 + plus_plus(1), # 只读
    # config. #
    # can. #
    'can.error': 20000 + plus_plus(0), # 只读
    # can.config. #
    'can.config.baud_rate': 21000 + plus_plus(0), # CAN波特率
    'can.config.enable_state_feedback': 22000 + plus_plus(0), # 是否开启CAN实时状态反馈
    # axis. #
    'axis.error': 30000 + plus_plus(0), # 只读
    'axis.current_state': 30000 + plus_plus(1), # 只读
    'axis.requested_state': 30000 + plus_plus(1), # 设置运行状态
    # axis.config. #
    'axis.config.can_node_id': 31000 + plus_plus(0), # CAN总线节点 ID 号
    'axis.config.state_feedback_rate_ms': 31000 + plus_plus(1), # CAN心跳信号发送的时间间隔，默认为 2，单位 ms，当总线中不同 ID 号关节数量为 n 时，需将该值设置为 2n
    # axis.config.extra_setting. #
    'axis.config.extra_setting.enable_angle_limit': 31200 + plus_plus(0), # 是否开启角度限制属性，默认不开启
    'axis.config.extra_setting.angle_min': 31200 + plus_plus(1), # 最小角度限位 默认为-180.5°
    'axis.config.extra_setting.angle_max': 31200 + plus_plus(1),# 最大角度限位 默认为180.5°
    'axis.config.extra_setting.gear_ratio': 31200 + plus_plus(1), # 只读 减速比
    'axis.config.extra_setting.stall_current_limit': 31200 + plus_plus(1), # 堵转电流
    'axis.config.extra_setting.enable_crash_detect': 31200 + plus_plus(1), # 是否开启碰撞检测，默认开启
    'axis.config.extra_setting.crash_detect_sensitivity': 31200 + plus_plus(1), # 碰撞检测灵敏度，该值需大于 0，越小越灵敏
    # axis.controller. #
    'axis.controller.error': 32000 + plus_plus(0), # 只读
    'axis.controller.trajectory_done': 32000 + plus_plus(1), # 只读
    # axis.controller.config. #
    'axis.controller.config.enable_vel_limit': 32100 + plus_plus(0),# 是否启用转速限制，默认启用
    'axis.controller.config.pos_gain': 32100 + plus_plus(1), # 位置增益
    'axis.controller.config.vel_gain': 32100 + plus_plus(1), # 转速增益
    'axis.controller.config.vel_integrator_gain': 32100 + plus_plus(1), # 转速积分增益
    'axis.controller.config.vel_limit': 32100 + plus_plus(1), # 最大限制转速，此处转速为电机转速，输出端需要考虑减速比
    'axis.controller.config.vel_limit_tolerance': 32100 + plus_plus(1), # 超出最大转速的忍耐度，比如设定为 1.2 时表示当转速超过设定的最大转速值的 1.2 倍才会触发 超速错误：ERROR_OVERSPEED
    'axis.controller.config.inertia': 32100 + plus_plus(1), # 负载转动惯量 默认为0
    'axis.controller.config.input_filter_bandwidth': 32100 + plus_plus(1), # 输入滤波带宽，反应外部控制信号输入的快慢
    # axis.motor. #
    'axis.motor.error': 33000 + plus_plus(0), # 只读
    # axis.motor.config. #
    'axis.motor.config.pole_pairs': 33100 + plus_plus(0), # 只读
    'axis.motor.config.phase_inductance': 33100 + plus_plus(1), # 只读
    'axis.motor.config.phase_resistance': 33100 + plus_plus(1), # 只读
    'axis.motor.config.torque_constant': 33100 + plus_plus(1), # 只读
    'axis.motor.config.current_lim': 33100 + plus_plus(1), # 最大电流限制
    'axis.motor.config.current_lim_margin': 33100 + plus_plus(1), # 最大电流限制忍耐度，如：此值设置为 3 表示当关节电流超过限制电流 3A 时停止关节并报错
    'axis.motor.config.torque_lim': 33100 + plus_plus(1), # 最大力矩限制，此处转速为电机力矩，输出端需要考虑减速比，默认为无穷
    'axis.motor.config.current_control_bandwidth': 33100 + plus_plus(1), # 电流控制带宽
    # axis.motor.current_control. #
    'axis.motor.current_control.Iq_measured': 33200 + plus_plus(0), # 只读
    'axis.motor.current_control.Id_measured': 33200 + plus_plus(1), # 只读
    # axis.encoder. #
    'axis.encoder.error': 34000 + plus_plus(0), # 只读
    # axis.fet_thermistor. #
    'axis.fet_thermistor.error': 36000 + plus_plus(0), # 只读
    'axis.fet_thermistor.temperature': 36000 + plus_plus(1), #只读
    # axis.fet_thermistor.config. #
    'axis.fet_thermistor.config.enabled': 36100 + plus_plus(0), # 是否开启驱动板温度保护，默认开启
    'axis.fet_thermistor.config.temp_limit_lower': 36100 + plus_plus(1), # 驱动板温度保护下限，即开始进行保护的温度
    'axis.fet_thermistor.config.temp_limit_upper': 36100 + plus_plus(1), # 驱动板温度保护上限，即完全保护待机的温度
    # axis.motor_thermistor. #
    'axis.motor_thermistor.error': 37000 + plus_plus(0), # 只读
    'axis.motor_thermistor.temperature': 37000 + plus_plus(1), #只读
    # axis.motor_thermistor.config. #
    'axis.motor_thermistor.config.enabled': 37100 + plus_plus(0), # 是否开启电机温度保护，默认开启
    'axis.motor_thermistor.config.temp_limit_lower': 37100 + plus_plus(1), # 电机温度保护下限，即开始进行保护的温度，一般不宜超过80℃
    'axis.motor_thermistor.config.temp_limit_upper': 37100 + plus_plus(1), # 电机温度保护上限，即完全保护待机的温度
    # axis.output_shaft. #
    'axis.output_shaft.pos_estimate': 38000 + plus_plus(0), # 只读
    'axis.output_shaft.vel_estimate': 38000 + plus_plus(1), # 只读
    'axis.output_shaft.torque_estimate': 38000 + plus_plus(1), # 只读
    'axis.output_shaft.angle_min': 38000 + plus_plus(1), # 输出端最小临时角度限位，默认为-180°
    'axis.output_shaft.angle_max': 38000 + plus_plus(1), # 输出端最大临时角度限位，默认为180°
    'axis.output_shaft.enable_angle_limit': 38000 + plus_plus(1), # 是否开启输出端临时角度限位，默认不开启


}

property_type = {
    #     #
    'vbus_voltage': 'f', # 只读
    'ibus': 'f', # 只读
    # can. #
    'can.error': 'u32',
    # can.config. #
    'can.config.baud_rate': 'u32',
    'can.config.enable_state_feedback': 'u32',
    # axis. #
    'axis.error': 'u32',
    'axis.current_state': 'u32',
    'axis.requested_state': 'u32',
    # axis.config. #
    'axis.config.can_node_id': 'u32',
    'axis.config.state_feedback_rate_ms': 'u32',
    # axis.config.extra_setting. #
    'axis.config.extra_setting.enable_angle_limit': 'u32',
    'axis.config.extra_setting.angle_min': 'f',
    'axis.config.extra_setting.angle_max': 'f',
    'axis.config.extra_setting.gear_ratio': 'f',
    'axis.config.extra_setting.stall_current_limit': 'f',
    'axis.config.extra_setting.enable_crash_detect': 'u32',
    'axis.config.extra_setting.crash_detect_sensitivity': 'u32',
    # axis.controller. #
    'axis.controller.error': 'u32',
    'axis.controller.trajectory_done': 'u32',
    # axis.controller.config. #
    'axis.controller.config.enable_gain_scheduling': 'u32',
    'axis.controller.config.enable_vel_limit': 'u32',
    'axis.controller.config.pos_gain': 'f',
    'axis.controller.config.vel_gain': 'f',
    'axis.controller.config.vel_integrator_gain': 'f',
    'axis.controller.config.vel_limit': 'f',
    'axis.controller.config.vel_limit_tolerance': 'f',
    'axis.controller.config.inertia': 'f',
    'axis.controller.config.input_filter_bandwidth': 'f',
    # axis.motor. #
    'axis.motor.error': 'u32',
    # axis.motor.config. #
    'axis.motor.config.pole_pairs': 's32',
    'axis.motor.config.phase_inductance': 'f',
    'axis.motor.config.phase_resistance': 'f',
    'axis.motor.config.torque_constant': 'f',
    'axis.motor.config.current_lim': 'f',
    'axis.motor.config.current_lim_margin': 'f',
    'axis.motor.config.torque_lim': 'f',
    'axis.motor.config.current_control_bandwidth': 'f',
    # axis.motor.current_control. #
    'axis.motor.current_control.Iq_measured': 'f',
    'axis.motor.current_control.Id_measured': 'f',
    # axis.encoder. #
    'axis.encoder.error': 'u32',
    # axis.fet_thermistor. #
    'axis.fet_thermistor.error': 'u32',
    'axis.fet_thermistor.temperature': 'f',
    # axis.fet_thermistor.config. #
    'axis.fet_thermistor.config.enabled': 'u32',
    'axis.fet_thermistor.config.temp_limit_lower': 'f',
    'axis.fet_thermistor.config.temp_limit_upper': 'f',
    # axis.motor_thermistor. #
    'axis.motor_thermistor.error': 'u32',
    'axis.motor_thermistor.temperature': 'f',
    # axis.motor_thermistor.config. #
    'axis.motor_thermistor.config.enabled': 'u32',
    'axis.motor_thermistor.config.temp_limit_lower': 'f',
    'axis.motor_thermistor.config.temp_limit_upper': 'f',
    # axis.output_shaft. #
    'axis.output_shaft.pos_estimate': 'f',
    'axis.output_shaft.vel_estimate': 'f',
    'axis.output_shaft.torque_estimate': 'f',
    'axis.output_shaft.angle_min': 'f',
    'axis.output_shaft.angle_max': 'f',
    'axis.output_shaft.enable_angle_limit': 'u32',
}


# 通过属性编码找到名称
def value_find_key(value=1):
    dict_temp = property_addresss
    if value in dict_temp.values():
        return list(dict_temp.keys())[list(dict_temp.values()).index(value)]
    else:
        return 0


# 通过名称找到属性编码
def key_find_value(key=''):
    if 'axis' in key:
        temp = key.split('.', 1)[-1]
        key = 'axis.' + temp
    dict_temp = property_addresss
    if key in dict_temp.keys():
        return dict_temp[key]
    else:
        return 0


error_codes = {
    'axis_error': [
        'AXIS_ERROR_NONE',
        'AXIS_ERROR_INVALID_STATE',
        'AXIS_ERROR_DC_BUS_UNDER_VOLTAGE',
        'AXIS_ERROR_DC_BUS_OVER_VOLTAGE',
        'AXIS_ERROR_CURRENT_MEASUREMENT_TIMEOUT',
        'AXIS_ERROR_BRAKE_RESISTOR_DISARMED',
        'AXIS_ERROR_MOTOR_DISARMED',
        'AXIS_ERROR_MOTOR_FAILED',
        'AXIS_ERROR_SENSORLESS_ESTIMATOR_FAILED',
        'AXIS_ERROR_ENCODER_FAILED',
        'AXIS_ERROR_CONTROLLER_FAILED',
        'AXIS_ERROR_POS_CTRL_DURING_SENSORLESS',
        'AXIS_ERROR_WATCHDOG_TIMER_EXPIRED',
        'AXIS_ERROR_MIN_ENDSTOP_PRESSED',
        'AXIS_ERROR_MAX_ENDSTOP_PRESSED',
        'AXIS_ERROR_ESTOP_REQUESTED',
        'AXIS_ERROR_UNKOWN_ERROR_CODE1',
        'AXIS_ERROR_UNKOWN_ERROR_CODE2',
        'AXIS_ERROR_HOMING_WITHOUT_ENDSTOP',
        'AXIS_ERROR_OVER_TEMP'],

    'motor_error': [
        'MOTOR_ERROR_NONE',
        'MOTOR_ERROR_PHASE_RESISTANCE_OUT_OF_RANGE',
        'MOTOR_ERROR_PHASE_INDUCTANCE_OUT_OF_RANGE',
        'MOTOR_ERROR_ADC_FAILED',
        'MOTOR_ERROR_DRV_FAULT',
        'MOTOR_ERROR_CONTROL_DEADLINE_MISSED',
        'MOTOR_ERROR_NOT_IMPLEMENTED_MOTOR_TYPE',
        'MOTOR_ERROR_BRAKE_CURRENT_OUT_OF_RANGE',
        'MOTOR_ERROR_MODULATION_MAGNITUDE',
        'MOTOR_ERROR_BRAKE_DEADTIME_VIOLATION',
        'MOTOR_ERROR_UNEXPECTED_TIMER_CALLBACK',
        'MOTOR_ERROR_CURRENT_SENSE_SATURATION',
        'MOTOR_ERROR_UNKOWN_ERROR_CODE',
        'MOTOR_ERROR_CURRENT_LIMIT_VIOLATION',
        'MOTOR_ERROR_BRAKE_DUTY_CYCLE_NAN',
        'MOTOR_ERROR_DC_BUS_OVER_REGEN_CURRENT',
        'MOTOR_ERROR_DC_BUS_OVER_CURRENT',
        'MOTOR_ERROR_STALL_LIMIT_VIOLATION'],

    'thermistor_error': [
        'THERMISTOR_CURRENT_LIMITER_ERROR_NONE',
        'THERMISTOR_CURRENT_LIMITER_ERROR_OVER_TEMP'],

    'encoder_error': [
        'ENCODER_ERROR_NONE',
        'ENCODER_ERROR_UNSTABLE_GAIN',
        'ENCODER_ERROR_CPR_POLEPAIRS_MISMATCH',
        'ENCODER_ERROR_NO_RESPONSE',
        'ENCODER_ERROR_UNSUPPORTED_ENCODER_MODE',
        'ENCODER_ERROR_ILLEGAL_HALL_STATE',
        'ENCODER_ERROR_INDEX_NOT_FOUND_YET',
        'ENCODER_ERROR_ABS_SPI_TIMEOUT',
        'ENCODER_ERROR_ABS_SPI_COM_FAIL',
        'ENCODER_ERROR_ABS_SPI_NOT_READY',
        'ENCODER_ERROR_EDR_OVER_RANGE',
        'ENCODER_ERROR_EDR_POWER_FAILURE',
        'ERROR_EDR_BATTERY_LOW ',
        'ERROR_OFFSET_CALIBRATE_FAILURE'],

    'controller_error': [
        'CONTROLLER_ERROR_NONE',
        'CONTROLLER_ERROR_OVERSPEED',
        'CONTROLLER_ERROR_INVALID_INPUT_MODE',
        'CONTROLLER_ERROR_UNSTABLE_GAIN',
        'CONTROLLER_ERROR_INVALID_MIRROR_AXIS',
        'CONTROLLER_ERROR_INVALID_LOAD_ENCODER',
        'CONTROLLER_ERROR_INVALID_ESTIMATE'],

    'can_error': [
        'CAN_ERROR_NONE',
        'CAN_ERROR_DUPLICATE_CAN_IDS'],

    'fet_thermistor_error': [
        'FET_THERMISTOR_CURRENT_LIMITER_ERROR_NONE',
        'FET_THERMISTOR_CURRENT_LIMITER_ERROR_OVER_TEMP'],

    'motor_thermistor_error': [
        'MOTOR_THERMISTOR_CURRENT_LIMITER_ERROR_NONE',
        'MOTOR_THERMISTOR_CURRENT_LIMITER_ERROR_OVER_TEMP']

}


def error_decode(data=''):
    line_list = data.split('\r\n')
    for line in line_list:
        if line != '':
            print(line)
            error_name = line.split(': ')[0] + '_error'
            code_num = int(line.split()[1])
            error_list = []
            for i in range(len(error_codes[error_name]) - 1):
                if code_num & (0x01 << i):
                    print(error_codes[error_name][i + 1])
