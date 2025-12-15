import math
import numpy as np
import matplotlib.pyplot as plt
import copy
import csv
import os
from datetime import datetime  # 导入时间戳模块
from scipy.spatial.distance import pdist

from part import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup

pi = math.pi
"""
    1.这是最新开发的abaqus 虚拟编织仿真脚本：
        它能够根据行列式三维四向编织机原理，根据用户输入的行列数、虚拟纤维数、纱线长度、纤维材料性能参数等，自动完成
    纱线的初始化建模、材料赋予、网格划分、分析步创建、边界条件设定。
        相较于之前的版本，它有以下几点更新：
            (1)、改进了初始建模的算法。现在所有纱线长度相等，根据“纱线长度”参数进行设定，解决了较大行列数下内部纱线远短于外部纱线的问题；
        然而这一改动对线性弹簧拉力作用的影响暂未被研究。
            (2)、现在额外的可以设置编织机的高度。这个高度值为纱线长度的4倍以上，越长的模拟高度会带来越小的纱线间拉力差异。
            (3)、新增了log打印功能，现在可以记录更多内容，妈妈再也不用担心记不住仿真的内容啦~
    2.单位制：
        ABAQUS采用自适应单位，用户不需要为变量指定单位，但需根据指定的单位组进行使用；
        下面是该脚本采用的单位制，用户应当与其保持一致。
            Length:     mm                  
            Force:      N
            Mass:       t
            Time:       s
            Stress:     MPa (N/mm²)             1 GPa  =  1e3 MPa  =  1e9 Pa
            Energy:     mJ (10^-3 J)
            Density:    tonne/mm³               1 kg/m^3  =  1e-12 tonne/mm^3
            
    3.建模方法：
    
    3.1 行列式编织机节点定义
        行列式编织机的每一个控制点具有两个参数，其一为编号、其二为当前分析步的实际二维坐标。
        实际二维坐标 用于计算当前分析步中该控制节点的位移边界条件方向（奇/偶 行/列）
        编号 用于在abaqus中索引该控制节点对应的节点对象，从而施加边界条件，用初始二维坐标表征，
        
    3.2 纤维直径与网格划分
        单根虚拟纤维截面积 由 纤维材料的线密度 / 纤维材料的体密度 / 虚拟纤维数 计算而成，开方、计算可得纤维直径
        使用T3D2单元，默认情况下，网格单元长度为虚拟纤维直径的2.5倍，
            用于保证桁架单元beam-to-beam接触的有效检测半径为实际半径，不发生穿透
    
    3.3 
"""

# --------------------------
# 1.CAE初始化
# --------------------------


executeOnCaeStartup()
model_name = "BraidedSimulia"
FFMModel = mdb.Model(name=model_name)
Amplitude = 'Amp-smoothstep-T20'


# --------------------------
# 2.CAE参数输入窗口设置
# --------------------------


# 编织参数输入
# number of fibers, 1, 7, 19, 37, 61, 91, 127, 169, 217, 271
braid_para_fields = (
    ('row_count', '4'),
    ('col_count', '4'),
    ('virtual_fiber_count', '1'),
    ('yarn_length /mm', '24'),                      # 纱线长度（不变）         mm
    ('height (mm)', '48'),                          # 编织机高度（不变）       mm
    ('control_node_spacing (mm)', '4'),             # 控制节点间距            mm
    ('spring_k (N/mm)', '0.1'),                     # 弹簧刚度               N/mm
    ('spring_rl (mm)', '10'),                       # 弹簧参考长度            mm
    ('yarn_interval_factor', '1'),                  # 纱线间隔系数
    ('fiber_interval_factor', '1.0001'),            # 纱线间隔系数
    ('friction_factor', '0.3')                      # 摩擦系数
)

# 材料参数输入
material_fields = (
    ('youngs_module (MPa)', '230e3'),               # 弹性模量              MPa
    ('poissons_ratio', '0.35'),                     # 泊松比
    ('yarn_density_v (t/mm3)', '1.81e-9'),          # 体积密度              t/mm3
    ('yarn_density_l (t/mm)', '0.51e-9')            # 线密度                t/mm
)

# 计算参数输入（不变）
compute_para_fields = (
    ('cpus', '12'),                                  # 并行计算核心数
    ('analysis_steps', '16'),                       # 分析步数
    ('time_period', '1.0'),
    ('min_time_increment', '-1'),
    ('length_of_element', '-1')
)

# 弹出输入窗口获取参数
braid_para_list = getInputs(
    fields=braid_para_fields,
    label='Specify Braided Parameters:',
    dialogTitle='Braided Parameter Setting'
)
material_para_list = getInputs(
    fields=material_fields,
    label='Specify Material Parameters:',
    dialogTitle='Material Parameter Setting'
)
compute_para_list = getInputs(
    fields=compute_para_fields,
    label='Specify Computing Parameters:',
    dialogTitle='Computing Parameter Setting'
)

# --------------------------
# 3. 参数解析与类型转换
# --------------------------


row_count = int(braid_para_list[0])                         # 行数
col_count = int(braid_para_list[1])                         # 列数
virtual_fiber_count = int(braid_para_list[2])               # 虚拟纤维数
yarn_length = float(braid_para_list[3])                     # 纱线长度
height = float(braid_para_list[4])                          # 编织机高度
control_node_spacing = float(braid_para_list[5])            # 控制节点间距
spring_k = float(braid_para_list[6])                        # 弹簧刚度
spring_rl = float(braid_para_list[7])                       # 弹簧参考长度
yarn_interval_factor = float(braid_para_list[8])            # 束结端纱线间隔系数
fiber_interval_factor = float(braid_para_list[9])           # 纤维间隔系数
friction_factor = float(braid_para_list[10])                # 摩擦系数

# 材料参数解析（顺序与material_fields一致）
youngs_module = float(material_para_list[0])                # 弹性模量
poissons_ratio = float(material_para_list[1])               # 泊松比
yarn_density_v = float(material_para_list[2])               # 体积密度
yarn_density_l = float(material_para_list[3])               # 线密度

# 计算参数解析
cpus = int(compute_para_list[0])                            # 核心数
analysis_steps = int(compute_para_list[1])                  # 分析步数
time_period = float(compute_para_list[2])
min_time_increment = float(compute_para_list[3])
length_of_element = float(compute_para_list[4])

# 衍生参数：单根纤维直径计算
# 计算原则：体密度/线密度 得到纱线截面积，总截面积不变确定虚拟纤维直径
d_fiber = math.sqrt(4 * yarn_density_l / (pi * virtual_fiber_count * yarn_density_v))       # 单位：mm


"""
    最小稳定时间增量、最大接触惩罚刚度估算，以调整参数保证计算效率
"""
def optimize(fiber_radius_, youngs_module_, length_of_element_, target_time_increment_):
    """
        计算ABAQUS/Explicit所需的最小稳定时间增量、最大接触惩罚刚度；
        同时根据目标时间增量，对模型进行参数优化；
        这主要是通过调整材料密度同时等比增大弹簧张力实现的；
        这种方式相比于使用质量缩放，虽然在最小稳定时间增量的精度上有一定损失，
            但是可以根据计算值自动确定刚度需要增长的倍数，不再需要用户多次尝试运算
    Args:
        fiber_radius_:
        youngs_module_:
        length_of_element_:
        target_time_increment_:

    Returns:

    """
    area = pi * fiber_radius_ ** 2
    c_d = np.sqrt(youngs_module_ / yarn_density_v)
    t_stable_element = length_of_element_ / c_d
    m_element = yarn_density_v * area * length_of_element_
    k_penalty_theory = youngs_module_ * area / length_of_element_
    k_penalty_max = pi ** 3 * youngs_module_ * fiber_radius_ / 2000

    if target_time_increment_ > t_stable_element:
        k_scale_factor = target_time_increment_ / t_stable_element
        density_scale_factor = k_scale_factor ** 2
    else:
        k_scale_factor = 1
        density_scale_factor = 1

    log_filename_ = "fibered_yarn_log.txt"  # 固定Log文件名，后续可追加
    log_file_exists_ = os.path.exists(log_filename_)
    with open(log_filename_, "a", encoding="utf-8") as log_file_:
        # 首次创建文件时写入头部
        if not log_file_exists_:
            log_file_.write("=" * 80 + "\n")
            log_file_.write("           纱线纤维化建模日志（支持后续追加内容）\n")
            log_file_.write(f"          日志创建时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file_.write("=" * 80 + "\n\n")

        log_file_.write("0.ABAQUS / Explicit 计算性能优化参数计算\n")
        log_file_.write(f"{'最小稳定时间增量':<15}: {t_stable_element:>6}\n")
        log_file_.write(f"{'惩罚接触刚度理论值':<15}: {k_penalty_theory:>6}\n")
        log_file_.write(f"{'惩罚接触刚度最大值':<15}: {k_penalty_max:>6}\n\n")
        log_file_.write(f"{'外部拉力优化放大系数':<15}: {k_scale_factor:>6}\n")
        log_file_.write(f"{'密度优化放大系数':<15}: {density_scale_factor:>6}\n")
    return k_scale_factor, density_scale_factor


"""  Predefined variables  """
rp_dict = {}
rp_dict_copy = {}
n_step = 0

# --------------------------
# 4. 纱线纤维坐标初始化（仅计算）
# --------------------------


def if_number_of_fiber_is_valid(virtual_fiber_count_):
    """
    判断虚拟纤维数是否有效（符合六边形密排规则），有效则返回密排层数。
    Args:
        virtual_fiber_count_ (int): 虚拟纤维数
    Returns:
        int/False: 有效则返回层数，无效则返回False
    """
    if virtual_fiber_count > 0 and virtual_fiber_count % 6 == 1:
        fiber_quotient = virtual_fiber_count // 6
        # 循环计算三角形数，i从1开始（直接用i作为层数计算基准）
        for i in range(1, 100):  # i是生成三角形数的变量
            triangle_num = (i * (i + 1)) / 2
            if triangle_num == fiber_quotient:
                return i + 1  # 层数 = i + 1（核心修正）
        return False
    return False


def hexagonal_packing(diameter, virtual_fiber_count_, fiber_interval_factor_=1, plot=False):
    """
    生成虚拟纤维的六边形密排坐标（2D），用于后续纤维化平移。
    Args:

        diameter (float): 单根纤维直径
        virtual_fiber_count_ (int): 虚拟纤维数
        fiber_interval_factor_ (float): 纱线间隔系数（默认1.25）
        plot (bool): 是否可视化密排结果（默认False）
    Returns:
        list[tuple[float, float]]: 纤维圆心坐标列表（2D）
    """
    # 先判断纤维数有效性，获取密排层数
    layers = if_number_of_fiber_is_valid(virtual_fiber_count_)
    if not layers:
        raise ValueError(f"Invalid virtual_fiber_count: {virtual_fiber_count_}. Must satisfy count%6==1.")

    coordinates_ = [(0.0, 0.0)]  # 中心纤维坐标（初始2D）

    # 循环计算每一层的纤维坐标（从第2层开始）
    for layer_ in range(2, layers + 1):
        base_angle_ = math.radians(30)  # 基础角度（60°间隔的起始角）
        base_points_ = []  # 存储当前层6个顶点坐标

        # 计算当前层6个顶点坐标（2D）
        for i_ in range(6):
            angle_ = base_angle_ + math.radians(60) * i_  # 每个顶点的角度（间隔60°）
            # 半径 = (层数-1) * 纤维直径 * 间隔系数
            radius_ = (layer_ - 1) * diameter * fiber_interval_factor_
            x_ = radius_ * math.cos(angle_)
            y_ = radius_ * math.sin(angle_)
            coordinates_.append((x_, y_))
            base_points_.append((x_, y_))

        # 闭合顶点列表（最后一个点=第一个点，方便后续插值）
        base_points_.append(base_points_[0])

        # 层数>2时，计算每层边中间的纤维坐标（线性插值）
        if layer_ > 2:
            for i_ in range(6):  # 遍历6条边
                # 每条边的中间纤维数 = 层数-2（总点数=层数-1，减去两端顶点）
                for j_ in range(1, layer_ - 1):
                    # 线性插值公式：起点 + j*(终点-起点)/(层数-1)
                    x_new_ = base_points_[i_][0] + (base_points_[i_ + 1][0] - base_points_[i_][0]) * j_ / (layer_ - 1)
                    y_new_ = base_points_[i_][1] + (base_points_[i_ + 1][1] - base_points_[i_][1]) * j_ / (layer_ - 1)
                    coordinates_.append((x_new_, y_new_))

    # 可视化密排结果（可选）
    if plot:
        fig_, ax_ = plt.subplots(figsize=(6, 6))
        radius_ = diameter / 2  # 纤维半径
        for x_, y_ in coordinates_:
            circle_ = plt.Circle((x_, y_), radius_, color='blue', fill=False, linewidth=1)
            ax_.add_artist(circle_)
        # 调整坐标轴范围（包含所有纤维，留10%余量）
        all_x_ = [x for x, y in coordinates_]
        all_y_ = [y for x, y in coordinates_]
        margin_ = radius_ * 1.2
        ax_.set_xlim(min(all_x_) - margin_, max(all_x_) + margin_)
        ax_.set_ylim(min(all_y_) - margin_, max(all_y_) + margin_)
        ax_.set_aspect('equal')
        ax_.set_title(f'Hexagonal Packing (Fibers: {virtual_fiber_count_}, Layers: {layers})')
        ax_.set_xlabel('X (mm)')
        ax_.set_ylabel('Y (mm)')
        plt.show()

    return coordinates_


def parity_judgement(row_count_, col_count_):
    """
    根据行列数的奇偶性生成判断列表，用于控制边缘控制点的生成规则。
    Args:
        row_count_ (int): 行数
        col_count_ (int): 列数
    Returns:
        list[int]: 奇偶判断列表（[底部边缘, 右侧边缘, 顶部边缘, 左侧边缘]）
    """
    # 核心逻辑：仅列数奇偶性决定结果（保留原注释逻辑）
    if col_count_ % 2 == 0:
        judgement_list_ = [1, 1, 0, 0]
    else:
        judgement_list_ = [1, 0, 0, 1]
    return judgement_list_


def generate_control_points(spacing, row_count_, col_count_, z_coordinate_):
    """
    生成编织机的控制点坐标（3D），包含中心控制点和边缘控制点。
    下述以 x 代表 列， y 代表 行。
    Args:
        spacing (float): 控制点间距
        row_count_ (int): 行数
        col_count_ (int): 列数
        z_coordinate_ (float): 所有控制点的Z向坐标
    Returns:
        list[tuple[float, float, float]]: 控制点坐标列表（3D）
    """
    z_coord_ = z_coordinate_
    center_points_ = []
    outer_points_ = []

    # 1. 生成中心区域控制点（修复原逻辑：i循环行数，j循环列数）
    for j_ in range(1, col_count_ + 1):  # j：列方向（Y轴）
        for i_ in range(1, row_count_ + 1):  # i：行方向（X轴）- 原bug修复：用row_count_而非col_count_
            x_ = (i_ - row_count_ / 2 - 0.5) * spacing
            y_ = (j_ - col_count_ / 2 - 0.5) * spacing
            center_points_.append((x_, y_, z_coord_))

    # 2. 生成边缘区域控制点（基于奇偶判断列表）
    judgement_list_ = parity_judgement(row_count_, col_count_)
    # 循环范围：行列数+2（包含边缘外侧的虚拟行/列）
    for i_ in range(row_count_ + 2):
        for j_ in range(col_count_ + 2):
            # 左侧边缘（i=0，排除首尾列）
            if i_ == 0 and j_ % 2 == judgement_list_[0] and 0 < j_ < col_count_ + 1:
                x_ = (i_ - row_count_ / 2 - 0.5) * spacing
                y_ = (j_ - col_count_ / 2 - 0.5) * spacing
                outer_points_.append((x_, y_, z_coord_))
            # 右侧边缘（j=col_count+1，排除首尾行）
            if j_ == col_count_ + 1 and i_ % 2 == judgement_list_[1] and 0 < i_ < row_count_ + 1:
                x_ = (i_ - row_count_ / 2 - 0.5) * spacing
                y_ = (j_ - col_count_ / 2 - 0.5) * spacing
                outer_points_.append((x_, y_, z_coord_))
            # 顶部边缘（i=row_count+1，排除首尾列）
            if i_ == row_count_ + 1 and j_ % 2 == judgement_list_[2] and 0 < j_ < col_count_ + 1:
                x_ = (i_ - row_count_ / 2 - 0.5) * spacing
                y_ = (j_ - col_count_ / 2 - 0.5) * spacing
                outer_points_.append((x_, y_, z_coord_))
            # 左侧边缘（j=0，排除首尾行）
            if j_ == 0 and i_ % 2 == judgement_list_[3] and 0 < i_ < row_count_ + 1:
                x_ = (i_ - row_count_ / 2 - 0.5) * spacing
                y_ = (j_ - col_count_ / 2 - 0.5) * spacing
                outer_points_.append((x_, y_, z_coord_))

    # 合并中心和边缘控制点
    return center_points_ + outer_points_


def initialize_yarn_coords(row_count_, col_count_, control_node_spacing_, fiber_diameter_,
                           yarn_length_, virtual_fiber_count_, height_, yarn_interval_factor_=1, plot=False):
    """
    初始化纱线的基准坐标（3D），返回束结端点、控制端点、连接端坐标。
    Args:
        row_count_ (int): 行数
        col_count_ (int): 列数
        control_node_spacing_ (float): 控制节点间距
        fiber_diameter_ (float): 单根纤维直径
        yarn_length_ (float): 纱线长度
        virtual_fiber_count_ (int): 虚拟纤维数（仅用于日志，不影响计算）
        height_ (float): 束结端点的Z向高度
        yarn_interval_factor_ (float): 纱线间隔系数（默认1）；函数自带保守估计，不会出现初始模型穿透，如果间距过大可调整该系数，使其小于1
        plot (bool): 是否可视化纱线基准坐标（默认False）
    Returns:
        tuple: (bundle_knot_coords, control_end_coords, yarn_connection_coords)
            - bundle_knot_coords: 束结端点坐标列表（3D）
            - control_end_coords: 控制端点坐标列表（3D，后续建模用）
            - yarn_connection_coords: 纱线连接端坐标列表（3D）
    """

    # 1. 计算束结端点间距（基于纤维直径和间隔系数）
    layers_ = if_number_of_fiber_is_valid(virtual_fiber_count_)
    yarn_diameter_ = (2 * layers_ - 1) * fiber_diameter_
    diff = np.sqrt(
        (control_node_spacing_ - yarn_diameter_)**2 / 2**2 *
        ((row_count_ + 2 ) ** 2 + (col_count_ + 2 ) ** 2)
    )
    estimate_fiber_interval_factor_ = 1 / np.sin(np.arctan2(height_, diff))

    bundle_spacing_ = estimate_fiber_interval_factor_ * yarn_diameter_ * yarn_interval_factor_ if virtual_fiber_count_ != 1 \
        else fiber_diameter_ * yarn_interval_factor_

    # 2. 生成束结端点坐标（Z向=height_）
    bundle_knot_coords_ = generate_control_points(
        spacing=bundle_spacing_,
        row_count_=row_count_,
        col_count_=col_count_,
        z_coordinate_=height_
    )

    # 3. 生成控制端点坐标（Z向=0，后续用于边界条件）
    control_end_coords_ = generate_control_points(
        spacing=control_node_spacing_,
        row_count_=row_count_,
        col_count_=col_count_,
        z_coordinate_=0.0
    )

    # 4. 计算纱线连接端坐标（从束结端沿束结→控制端方向延伸纱线长度）
    yarn_connection_coords_ = []
    direction_vectors_ = []
    for bundle_knot_, control_end_ in zip(bundle_knot_coords_, control_end_coords_):
        # 计算束结端到控制端的方向向量
        dx_ = control_end_[0] - bundle_knot_[0]
        dy_ = control_end_[1] - bundle_knot_[1]
        dz_ = control_end_[2] - bundle_knot_[2]

        # 计算方向向量模长（避免除零）
        distance_ = math.sqrt(dx_ ** 2 + dy_ ** 2 + dz_ ** 2)
        if distance_ < 1e-6:  # 两点重合（数值误差容忍）
            connection_ = bundle_knot_
            direction_ = (0, 0, 0)
        else:
            # 单位方向向量 + 延伸纱线长度
            unit_dx_ = dx_ / distance_
            unit_dy_ = dy_ / distance_
            unit_dz_ = dz_ / distance_
            connection_ = (
                bundle_knot_[0] + unit_dx_ * yarn_length_,
                bundle_knot_[1] + unit_dy_ * yarn_length_,
                bundle_knot_[2] + unit_dz_ * yarn_length_
            )
            direction_ = (dx_, dy_, dz_)
        yarn_connection_coords_.append(connection_)
        direction_vectors_.append(direction_)

    # 5. 可视化纱线基准坐标（可选）
    if plot:
        fig_, ax_ = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})

        # 解包坐标
        bundle_x_ = [x for x, y, z in bundle_knot_coords_]
        bundle_y_ = [y for x, y, z in bundle_knot_coords_]
        bundle_z_ = [z for x, y, z in bundle_knot_coords_]
        control_x_ = [x for x, y, z in control_end_coords_]
        control_y_ = [y for x, y, z in control_end_coords_]
        control_z_ = [z for x, y, z in control_end_coords_]
        conn_x_ = [x for x, y, z in yarn_connection_coords_]
        conn_y_ = [y for x, y, z in yarn_connection_coords_]
        conn_z_ = [z for x, y, z in yarn_connection_coords_]

        # 绘制各点
        ax_.scatter(bundle_x_, bundle_y_, bundle_z_, c='red', marker='o', s=50, label='Bundle Knots')
        ax_.scatter(control_x_, control_y_, control_z_, c='blue', marker='^', s=40, label='Control Ends')
        ax_.scatter(conn_x_, conn_y_, conn_z_, c='green', marker='s', s=30, label='Yarn Connections')

        # 绘制纱线（束结端→连接端）
        for b_knot_, conn_ in zip(bundle_knot_coords_, yarn_connection_coords_):
            ax_.plot(
                [b_knot_[0], conn_[0]],
                [b_knot_[1], conn_[1]],
                [b_knot_[2], conn_[2]],
                'k--', linewidth=0.8, alpha=0.6
            )

        # 图表配置
        ax_.set_xlabel('X (mm)')
        ax_.set_ylabel('Y (mm)')
        ax_.set_zlabel('Z (mm)')
        ax_.set_title(f'Yarn Base Coordinates (Rows: {row_count_}, Cols: {col_count_})')
        ax_.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    return bundle_knot_coords_, control_end_coords_, yarn_connection_coords_, direction_vectors_, estimate_fiber_interval_factor_*yarn_interval_factor_


def translate_points_with_visualization(points_, translation_vectors_, visualize=False):
    """
    3D点集平移 + 集成可视化：将原始点集按平移向量批量平移，返回平移后点集。
    Args:
        points_ (list/tuple/np.ndarray): 原始点集（2D/3D，2D点自动补Z=0）
        translation_vectors_ (list/tuple/np.ndarray): 平移向量集（2D/3D，每个向量为3D坐标）
        visualize (bool): 是否可视化原始点与平移后点（默认False）
    Returns:
        np.ndarray: 平移后的点集（3D数组，shape=[向量组数, 单组向量数, 点数量, 3]）
    """
    # 1. 原始点集标准化（转为3D numpy数组）
    points_np_ = np.array(points_, dtype=np.float64)
    if points_np_.ndim == 1:
        points_np_ = points_np_.reshape(1, -1)  # 1D→2D
    if points_np_.shape[1] == 2:
        # 2D点补Z=0，转为3D
        points_np_ = np.hstack((points_np_, np.zeros((points_np_.shape[0], 1), dtype=np.float64)))
    elif points_np_.shape[1] != 3:
        raise ValueError(f"Points must be 2D or 3D, got {points_np_.shape[1]}D.")

    # 2. 平移向量集标准化（转为3D numpy数组）
    vecs_np_ = np.array(translation_vectors_, dtype=np.float64)
    if vecs_np_.ndim == 2:
        # 2D向量集→3D（新增“向量组”维度）
        vecs_np_ = vecs_np_[np.newaxis, :, :]
    elif vecs_np_.ndim != 3:
        raise ValueError(f"Translation vectors must be 2D or 3D, got {vecs_np_.ndim}D.")
    if vecs_np_.shape[-1] != 3:
        raise ValueError(f"Each translation vector must be 3D, got {vecs_np_.shape[-1]}D.")

    # 3. 批量计算平移后点集
    # 输出shape：[向量组数, 单组向量数, 点数量, 3]
    translated_np_ = np.zeros(
        (vecs_np_.shape[0], vecs_np_.shape[1], points_np_.shape[0], 3),
        dtype=np.float64
    )
    for i_, vec_group_ in enumerate(vecs_np_):
        for j_, single_vec_ in enumerate(vec_group_):
            translated_np_[i_][j_] = points_np_ + single_vec_  # 广播机制：点集+单个向量

    # 4. 可视化（可选）
    if visualize:
        fig_, ax_ = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})

        # 绘制原始点
        ax_.scatter(
            points_np_[:, 0], points_np_[:, 1], points_np_[:, 2],
            c='red', marker='o', s=50, label='Original Fiber Points'
        )

        # 绘制平移后点（不同组用不同颜色，避免标签重复）
        colors_ = ['blue', 'green', 'orange', 'purple']
        for i_, vec_group_ in enumerate(translated_np_):
            group_color_ = colors_[i_ % len(colors_)]  # 循环取色
            for j_, single_trans_ in enumerate(vec_group_):
                # 每组只标1个标签（避免重复）
                label_ = f'Translated Group {i_ + 1}' if j_ == 0 else ""
                ax_.scatter(
                    single_trans_[:, 0], single_trans_[:, 1], single_trans_[:, 2],
                    c=group_color_, marker='^', s=40, alpha=0.8, label=label_
                )

        # 图表配置
        ax_.set_xlabel('X (mm)')
        ax_.set_ylabel('Y (mm)')
        ax_.set_zlabel('Z (mm)')
        ax_.set_title('3D Visualization: Original & Translated Points')
        ax_.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    return translated_np_


def generate_fibered_yarn_all_points(
        row_count_, col_count_, control_node_spacing_, fiber_diameter_,
        yarn_length_, virtual_fiber_count_, height_, yarn_interval_factor_=1,fiber_interval_factor_=1.0001,
        plot_hex_pack=False, plot_translate=False, plot_final=False
):
    """
    纱线纤维化总函数：整合纤维密排、坐标平移、可视化、Log日志（追加模式）与表格保存。
    核心功能：根据输入参数生成纤维化纱线坐标，支持后续建模所需的控制点、纤维点数据输出，
    Log文件支持后续追加内容（无时间戳，固定命名），CSV表格防乱码且适配后续数据处理。

    Args:
        row_count_ (int): 编织机控制点行数，需为正整数，决定横向控制点数量。
        col_count_ (int): 编织机控制点列数，需为正整数，决定纵向控制点数量。
        control_node_spacing_ (float): 编织机相邻控制点的间距（单位：mm），用于计算控制点坐标。
        fiber_diameter_ (float): 单根虚拟纤维的直径（单位：mm），由材料线密度、体积密度计算得出。
        yarn_length_ (float): 单根纱线的长度（单位：mm），解决原版本中内外纱线长度不一致问题。
        virtual_fiber_count_ (int): 单根纱线包含的虚拟纤维数量，需满足count%6==1（如1,7,19...），符合六边形密排规则。
        height_ (float): 纱线束结端点的Z向高度（单位：mm），与控制端点（Z=0）形成垂直距离。
        yarn_interval_factor_ (float, optional): 纱线间隔系数，降低该系数以缩小束结端纱线之间的的间距。
        fiber_interval_factor_ (float, optional): 纤维间隔系数， 函数自带保守估计，多纤维建模时不会发生模型穿透，可以降低该系数以缩小间距。
        plot_hex_pack (bool, optional): 是否可视化纤维的六边形密排效果，默认False，True时弹出2D密排图形。
        plot_translate (bool, optional): 是否可视化纤维坐标平移过程，默认False，True时弹出3D原始/平移点对比图形。
        plot_final (bool, optional): 是否可视化最终纤维化纱线的3D效果，默认False，True时弹出含所有纱线、控制点的3D图形。

    Returns:
        dict: 包含所有建模所需数据的字典，键值说明如下：
            - fibered_bundle_knot: 纤维化后的纱线束结端点坐标列表，每个元素为(x,y,z)元组（单位：mm）。
            - fibered_yarn_connection: 纤维化后的纱线连接端坐标列表，每个元素为(x,y,z)元组（单位：mm）。
            - control_end_coords: 编织机控制点坐标列表（Z=0），用于后续边界条件设定，每个元素为(x,y,z)元组（单位：mm）。
            - raw_fiber_coords: 虚拟纤维的原始六边形密排坐标（2D），每个元素为(x,y)元组（单位：mm），供校验密排效果。
            - translated_fiber_np: 纤维平移后的numpy数组（shape=[2, 纱线数, 纤维数, 3]），供后续数值计算使用。
            - d_fiber: 单根虚拟纤维的直径（单位：mm），与输入参数fiber_diameter_一致，便于后续调用。
            - d_yarn: 纤维化后的纱线直径（单位：mm），单纤维时等于d_fiber，多纤维时=（2*密排层数-1）*d_fiber。
            - virtual_fiber_count: 单根纱线的虚拟纤维数量，与输入参数virtual_fiber_count_一致，供日志记录与校验。

    """

    yarn_count_ = 0  # 初始化纱线总数，后续赋值

    # --------------------------
    # 修改1：固定文件命名（去掉时间戳，Log支持追加）
    # --------------------------
    log_filename_ = "fibered_yarn_log.txt"          # 固定Log文件名，后续可追加
    yarn_table_filename_ = "fibered_yarn_coords.csv"# 固定纱线坐标表名
    control_table_filename_ = "control_points_coords.csv"# 固定控制点坐标表名

    # --------------------------
    # 步骤1：初始化返回结果字典
    # --------------------------
    result_dict_ = {
        "fibered_bundle_knot": [],
        "fibered_yarn_connection": [],
        "control_end_coords": [],
        "raw_fiber_coords": [],
        "translated_fiber_np": None,
        "d_fiber": fiber_diameter_,
        "d_yarn": 0.0,
        "virtual_fiber_count": virtual_fiber_count_
    }

    # --------------------------
    # 步骤2：判断纤维数有效性
    # --------------------------
    fiber_layers_ = if_number_of_fiber_is_valid(virtual_fiber_count_)
    if not fiber_layers_ and virtual_fiber_count_ != 1:
        raise ValueError(
            f"Virtual fiber count {virtual_fiber_count_} is invalid! "
            "Must satisfy count%6==1 (e.g., 1,7,19,37...)."
        )

    # --------------------------
    # 步骤3：生成纱线基准坐标
    # --------------------------
    bundle_knot_coords_, control_end_coords_, yarn_connection_coords_, direction_vectors_, estimate_fiber_interval_factor = initialize_yarn_coords(
        row_count_=row_count_,
        col_count_=col_count_,
        control_node_spacing_=control_node_spacing_,
        fiber_diameter_=fiber_diameter_,
        yarn_length_=yarn_length_,
        virtual_fiber_count_=virtual_fiber_count_,
        height_=height_,
        yarn_interval_factor_=yarn_interval_factor_,
        plot=True
    )
    result_dict_["control_end_coords"] = control_end_coords_
    yarn_count_ = len(bundle_knot_coords_)  # 确定纱线总数

    # --------------------------
    # 步骤4：分情况处理纤维化
    # --------------------------
    if virtual_fiber_count_ == 1:
        # 单纤维场景
        result_dict_["fibered_bundle_knot"] = bundle_knot_coords_
        result_dict_["fibered_yarn_connection"] = yarn_connection_coords_
        result_dict_["raw_fiber_coords"] = [(0.0, 0.0)]
        d_yarn = fiber_diameter_
    else:
        # 多纤维场景：遍历所有纱线提取坐标
        fiber_interval_values = []
        max_factor = 0.0
        max_index = -1
        for idx, vec in enumerate(direction_vectors_):
            dx, dy, dz = vec
            x_len = np.sqrt(dx ** 2 + dy ** 2)
            z_diff = abs(dz)
            # 防止除零
            if x_len < 1e-9:
                continue
                # 根据公式计算
            try:
                factor = 1.0 / np.sin(np.arctan2(z_diff, x_len))
            except Exception:
                factor = float('inf')

            fiber_interval_values.append(factor)

            # 记录最大值
            if factor > max_factor:
                max_factor = factor
                max_index = idx + 1  # 第几次计算，可视为第几根纤维

        max_factor = max_factor * fiber_interval_factor_

        # 计算纱线直径
        d_yarn = (2 * fiber_layers_ - 1) * fiber_diameter_ * max_factor
        result_dict_["d_yarn"] = d_yarn

        # if max_factor * 1.02 > estimate_fiber_interval_factor:
        #     raise ValueError("参数错误：当前编织角度过大，您可以选择增加编织机高度，或者增大\'yarn_interval_factor_\'的值来进行调整")
        raw_fiber_coords_ = hexagonal_packing(
            diameter=fiber_diameter_,
            virtual_fiber_count_=virtual_fiber_count_,
            fiber_interval_factor_=max_factor,
            plot=plot_hex_pack
        )
        result_dict_["raw_fiber_coords"] = raw_fiber_coords_

        translation_vectors_ = [bundle_knot_coords_, yarn_connection_coords_]
        translated_fiber_np_ = translate_points_with_visualization(
            points_=raw_fiber_coords_,
            translation_vectors_=translation_vectors_,
            visualize=plot_translate
        )
        result_dict_["translated_fiber_np"] = translated_fiber_np_

        # 遍历所有纱线，收集纤维化坐标
        fibered_bundle_knot_ = []
        fibered_yarn_conn_ = []
        for yarn_idx_ in range(translated_fiber_np_.shape[1]):
            yarn_bundle_fibers_ = translated_fiber_np_[0][yarn_idx_].tolist()
            yarn_conn_fibers_ = translated_fiber_np_[1][yarn_idx_].tolist()
            fibered_bundle_knot_.extend(yarn_bundle_fibers_)
            fibered_yarn_conn_.extend(yarn_conn_fibers_)
        result_dict_["fibered_bundle_knot"] = fibered_bundle_knot_
        result_dict_["fibered_yarn_connection"] = fibered_yarn_conn_

    # --------------------------
    # 修改2：Log日志（追加模式，支持后续写入更多内容）
    # --------------------------
    # 判断Log文件是否存在：不存在则创建并写头部，存在则直接追加内容
    log_file_exists_ = os.path.exists(log_filename_)
    with open(log_filename_, "a", encoding="utf-8") as log_file_:
        # 首次创建文件时写入头部
        if not log_file_exists_:
            log_file_.write("=" * 80 + "\n")
            log_file_.write("           纱线纤维化建模日志（支持后续追加内容）\n")
            log_file_.write(f"          日志创建时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file_.write("=" * 80 + "\n\n")

        # 追加当前运行的信息（带时间戳，区分每次运行记录）
        log_file_.write(f"【运行记录 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}】\n")
        log_file_.write("-" * 50 + "\n")

        # 1. 输入参数（左对齐15字符，右对齐数值，保持整齐）
        log_file_.write("1. 输入参数\n")
        log_file_.write(f"{'行数':<15}: {row_count_:>6}\n")
        log_file_.write(f"{'列数':<15}: {col_count_:>6}\n")
        log_file_.write(f"{'虚拟纤维数':<15}: {virtual_fiber_count_:>6}\n")
        log_file_.write(f"{'单纤维直径(mm)':<15}: {fiber_diameter_:>10.6f}\n")
        log_file_.write(f"{'纱线长度(mm)':<15}: {yarn_length_:>10.2f}\n")
        log_file_.write(f"{'编织机高度(mm)':<15}: {height_:>10.2f}\n")
        log_file_.write(f"{'控制节点间距(mm)':<15}: {control_node_spacing_:>10.2f}\n")
        log_file_.write(f"{'纱线间隔系数':<15}: {yarn_interval_factor_:>10.2f}\n")
        log_file_.write(f"{'是否可视化密排':<15}: {'是' if plot_hex_pack else '否':>6}\n")
        log_file_.write(f"{'是否可视化平移':<15}: {'是' if plot_translate else '否':>6}\n")
        log_file_.write(f"{'是否可视化最终结果':<15}: {'是' if plot_final else '否':>6}\n\n")

        # 2. 计算结果
        log_file_.write("2. 计算结果\n")
        log_file_.write(f"{'纤维密排层数':<15}: {fiber_layers_ if fiber_layers_ else 1:>6}\n")
        log_file_.write(f"{'纤维化纱线直径(mm)':<15}: {d_yarn:>10.6f}\n")
        if virtual_fiber_count_ != 1:
            log_file_.write(f"Yarn Interval Factor = {estimate_fiber_interval_factor:.6f} \n")
            log_file_.write(f"最大 Fiber Interval Factor = {max_factor:.6f} （出现在第 {max_index} 根纤维）\n")
        log_file_.write(f"{'总纱线数':<15}: {yarn_count_:>6}\n")
        log_file_.write(f"{'总控制点数量':<15}: {len(control_end_coords_):>6}\n")
        log_file_.write(f"{'单纱线纤维数':<15}: {virtual_fiber_count_:>6}\n")
        log_file_.write(f"{'总纤维点数量':<15}: {len(result_dict_['fibered_bundle_knot']):>6}\n\n")

        # 3. 当前运行的文件关联信息
        log_file_.write("3. 关联文件\n")
        log_file_.write(f"{'纱线坐标表格':<15}: {yarn_table_filename_}\n")
        log_file_.write(f"{'控制点坐标表格':<15}: {control_table_filename_}\n")
        log_file_.write(f"{'文件保存路径':<15}: {os.getcwd()}\n")
        log_file_.write("\n" + "=" * 80 + "\n\n")  # 分隔符，便于后续追加内容

    # --------------------------
    # 修改3：CSV表格（固定命名，utf-8-sig编码防乱码，英文表头）
    # --------------------------
    # 纱线坐标表格
    with open(yarn_table_filename_, "w", newline="", encoding="utf-8-sig") as yarn_csv_:
        yarn_header_ = [
            "Yarn_ID", "Fiber_ID",
            "Bundle_Knot_X(mm)", "Bundle_Knot_Y(mm)", "Bundle_Knot_Z(mm)",
            "Yarn_Connection_X(mm)", "Yarn_Connection_Y(mm)", "Yarn_Connection_Z(mm)"
        ]
        csv_writer_ = csv.writer(yarn_csv_)
        csv_writer_.writerow(yarn_header_)

        total_fibers_ = len(result_dict_["fibered_bundle_knot"])
        fibers_per_yarn_ = virtual_fiber_count_
        for yarn_idx_ in range(yarn_count_):
            start_idx_ = yarn_idx_ * fibers_per_yarn_
            end_idx_ = start_idx_ + fibers_per_yarn_
            if end_idx_ > total_fibers_:
                end_idx_ = total_fibers_

            for fiber_idx_ in range(start_idx_, end_idx_):
                bundle_x_ = result_dict_["fibered_bundle_knot"][fiber_idx_][0]
                bundle_y_ = result_dict_["fibered_bundle_knot"][fiber_idx_][1]
                bundle_z_ = result_dict_["fibered_bundle_knot"][fiber_idx_][2]
                conn_x_ = result_dict_["fibered_yarn_connection"][fiber_idx_][0]
                conn_y_ = result_dict_["fibered_yarn_connection"][fiber_idx_][1]
                conn_z_ = result_dict_["fibered_yarn_connection"][fiber_idx_][2]

                csv_writer_.writerow([
                    yarn_idx_ + 1,
                    fiber_idx_ - start_idx_ + 1,
                    round(bundle_x_, 6), round(bundle_y_, 6), round(bundle_z_, 6),
                    round(conn_x_, 6), round(conn_y_, 6), round(conn_z_, 6)
                ])

    # 控制点坐标表格
    with open(control_table_filename_, "w", newline="", encoding="utf-8-sig") as ctrl_csv_:
        ctrl_header_ = ["Control_Point_ID", "X_Coord(mm)", "Y_Coord(mm)", "Z_Coord(mm)"]
        csv_writer_ = csv.writer(ctrl_csv_)
        csv_writer_.writerow(ctrl_header_)

        for ctrl_idx_ in range(len(control_end_coords_)):
            x_ = control_end_coords_[ctrl_idx_][0]
            y_ = control_end_coords_[ctrl_idx_][1]
            z_ = control_end_coords_[ctrl_idx_][2]
            csv_writer_.writerow([
                ctrl_idx_ + 1,
                round(x_, 6),
                round(y_, 6),
                round(z_, 6)
            ])


    # --------------------------
    # 步骤5：最终可视化（原逻辑保留）
    # --------------------------
    if plot_final:
        fig_, ax_ = plt.subplots(figsize=(10, 8), subplot_kw={'projection': '3d'})
        yarn_colors_ = plt.cm.tab10(np.linspace(0, 1, min(yarn_count_, 10)))

        for yarn_idx_ in range(yarn_count_):
            fiber_start_ = yarn_idx_ * virtual_fiber_count_
            fiber_end_ = fiber_start_ + virtual_fiber_count_
            curr_yarn_bundle_ = result_dict_["fibered_bundle_knot"][fiber_start_:fiber_end_]
            curr_yarn_conn_ = result_dict_["fibered_yarn_connection"][fiber_start_:fiber_end_]

            bundle_x_ = [x for x, y, z in curr_yarn_bundle_]
            bundle_y_ = [y for x, y, z in curr_yarn_bundle_]
            bundle_z_ = [z for x, y, z in curr_yarn_bundle_]
            conn_x_ = [x for x, y, z in curr_yarn_conn_]
            conn_y_ = [y for x, y, z in curr_yarn_conn_]
            conn_z_ = [z for x, y, z in curr_yarn_conn_]

            label_bundle_ = f'Yarn {yarn_idx_ + 1}' if fiber_start_ == 0 else ""
            ax_.scatter(bundle_x_, bundle_y_, bundle_z_, c=[yarn_colors_[yarn_idx_ % 10]],
                       marker='o', s=40, alpha=0.8, label=label_bundle_)
            ax_.scatter(conn_x_, conn_y_, conn_z_, c=[yarn_colors_[yarn_idx_ % 10]],
                       marker='s', s=30, alpha=0.8)

            for knot_, conn_ in zip(curr_yarn_bundle_, curr_yarn_conn_):
                ax_.plot([knot_[0], conn_[0]], [knot_[1], conn_[1]], [knot_[2], conn_[2]],
                        color=yarn_colors_[yarn_idx_ % 10], linewidth=0.6, alpha=0.5)

        # 绘制控制点
        ctrl_x_ = [x for x, y, z in result_dict_["control_end_coords"]]
        ctrl_y_ = [y for x, y, z in result_dict_["control_end_coords"]]
        ctrl_z_ = [z for x, y, z in result_dict_["control_end_coords"]]
        ax_.scatter(ctrl_x_, ctrl_y_, ctrl_z_, c='gray', marker='x', s=50, alpha=0.8, label='Control Points')

        ax_.set_xlabel('X (mm)')
        ax_.set_ylabel('Y (mm)')
        ax_.set_zlabel('Z (mm)')
        ax_.set_title(f'Final Fibered Yarns (Total Yarns: {yarn_count_}, Fibers/Yarn: {virtual_fiber_count_})')
        ax_.legend(fontsize=9, loc='upper right')
        plt.tight_layout()
        plt.show()

    # 控制台提示（告知文件保存状态）
    print(f"\n=== Modeling Completed ===")
    print(f"Log file updated (append mode): {os.path.join(os.getcwd(), log_filename_)}")
    print(f"Yarn coordinates table saved to: {os.path.join(os.getcwd(), yarn_table_filename_)}")
    print(f"Control points table saved to: {os.path.join(os.getcwd(), control_table_filename_)}")
    print(f"Note: Log file supports subsequent appending of content.\n")

    return result_dict_


# --------------------------
# 5.ABAQUS 创建模型、赋予属性、划分网格、定义接触
# --------------------------


def abaqus_material(model_name_, material_name_="Material-fiber", elastic_params_=(340e3, 0.3), density_=2e-3):
    """
    创建Abaqus材料（原ABAQUSmaterial，适配新命名+参数规范）。
    Args:
        model_name_ (str): 模型名称
        material_name_ (str): 材料名称，默认"Material-fiber"
        elastic_params_ (tuple): 弹性参数（弹性模量MPa、泊松比），默认碳纤维参数(340e3, 0.3)
        density_ (float): 材料密度（单位：tonne/mm³），默认2e-3
    """
    # 创建材料
    mdb.models[model_name_].Material(name=material_name_)
    # 定义弹性属性
    mdb.models[model_name_].materials[material_name_].Elastic(table=(elastic_params_,))
    # 定义密度属性
    mdb.models[model_name_].materials[material_name_].Density(table=((density_,),))


def abaqus_section(model_name_, fiber_diameter_, truss_material_="Material-fiber"):
    """
    创建Abaqus桁架单元截面（原ABAQUSsection，适配新命名+纤维直径参数）。
    Args:
        model_name_ (str): 模型名称
        fiber_diameter_ (float): 单根纤维直径（单位：mm），用于计算截面积
        truss_material_ (str): 截面使用的材料名称，默认"Material-fiber"
    """
    # 计算桁架单元截面积（圆形截面：π*d²/4）
    cross_section_area_ = math.pi * (fiber_diameter_ ** 2) / 4
    # 创建桁架截面
    mdb.models[model_name_].TrussSection(
        name='Section-fiber',
        material=truss_material_,
        area=cross_section_area_
    )
    # 获取装配体（原a = mdb.models[model_name].rootAssembly，此处仅声明，后续建模用）
    assembly_ = mdb.models[model_name_].rootAssembly


def abaqus_interaction(model_name_, friction_factor_=0.3):
    """
    创建Abaqus接触属性与通用接触（原ABAQUSinteraction，适配新命名）。
    Args:
        model_name_ (str): 模型名称（原modelName）
        friction_factor_ (float): 摩擦系数（原frictionfactor），默认0.3
    """
    # 创建接触属性（含摩擦）
    mdb.models[model_name_].ContactProperty('contact_with_fric')
    mdb.models[model_name_].interactionProperties['contact_with_fric'].TangentialBehavior(
        formulation=PENALTY,
        directionality=ISOTROPIC,
        slipRateDependency=OFF,
        pressureDependency=OFF,
        temperatureDependency=OFF,
        dependencies=0,
        table=((friction_factor_,),),
        shearStressLimit=None,
        maximumElasticSlip=FRACTION,
        fraction=0.005,
        elasticSlipStiffness=None
    )
    # 创建通用接触
    mdb.models[model_name_].ContactExp(
        name='General_contact_with_fric',
        createStepName='Initial'
    )
    mdb.models[model_name_].interactions['General_contact_with_fric'].includedPairs.setValuesInStep(
        stepName='Initial',
        useAllstar=ON
    )
    mdb.models[model_name_].interactions['General_contact_with_fric'].contactPropertyAssignments.appendInStep(
        stepName='Initial',
        assignments=((GLOBAL, SELF, 'contact_with_fric'),)
    )


def abaqus_nonlinear_spring_connector(model_name_, spring_rl_, spring_k_table_):
    """
    （补充函数）创建非线性弹簧连接器截面（原ABAQUSnonlinearSpringConnector，适配新命名）。
    Args:
        model_name_ (str): 模型名称
        spring_rl_ (float): 弹簧参考长度（原springRL）
        spring_k_table_ (tuple): 弹簧刚度表（原springK），如((200.0,),)
    """
    # 创建连接器截面（非线性弹簧）
    mdb.models[model_name_].ConnectorSection(
        name='ConnSect-nonlinearSpring',
        u1ReferenceLength=spring_rl_,
        translationalType=AXIAL
    )
    elastic_0 = connectorBehavior.ConnectorElasticity(components=(1,), table=spring_k_table_)
    # 定义弹簧行为（轴向刚度）
    mdb.models[model_name_].sections['ConnSect-nonlinearSpring'].setValues(
        behaviorOptions=(elastic_0,), u1ReferenceLength=spring_rl_)
    mdb.models[model_name_].sections['ConnSect-nonlinearSpring'].behaviorOptions[0].ConnectorOptions(
        useBehExtSettings=OFF, extrapolation=LINEAR)

    return

def get_keys_of_rps(i_, row_count_, col_count_, total_rp_count_):
    """
    根据索引i获取参考点（RP）的键（行列标识），适配新的row_count/col_count命名。
    改进：
        1、修复了针对奇数行/列参考点创建失败的风险
    Args:
        i_ (int): 参考点索引
        row_count_ (int): 控制点行数
        col_count_ (int): 控制点列数
        total_rp_count_ (int): 参考点总数
    Returns:
        tuple: 参考点的键（如(1,2)表示第1行第2列）
    """
    global rp_dict
    judgement_list_ = parity_judgement(row_count_, col_count_)

    # 内部控制点（行列范围内）
    if i_ < row_count_ * col_count_:
        # 修正键计算逻辑，确保行列对应正确
        key_ = (i_ % row_count_ + 1, i_ // row_count_ + 1)
    # 边缘参考点
    else:
        left_rp_end_ = int(row_count_ * col_count_ + math.ceil(col_count_ / 2))
        right_rp_end_ = int(total_rp_count_ - col_count_ // 2)
        # 左侧边缘（i=0行）,
        if i_ in range(int(row_count_ * col_count_), left_rp_end_):
            j_ = i_ - row_count_ * col_count_ + 1
            key_ = (0, 2 * j_ - 1)
        # 上下边缘（j=0列或j=col_count+1列）
        elif i_ in range(left_rp_end_, right_rp_end_):
            j_ = int(i_ - left_rp_end_ + 1)
            if judgement_list_[1] == 0:
                key_ = (j_, 0) if j_ % 2 == 1 else (j_, col_count_ + 1)
            else:
                key_ = (j_, col_count_ + 1) if j_ % 2 == 1 else (j_, 0)
        # 右侧边缘（i=row_count+1行）
        else:
            j_ = int(i_ - right_rp_end_ + 1)
            key_ = (row_count_ + 1, 2 * j_)

    return key_


def sum_values_by_keys(dict_, target_row_col_, cond_="row"):
    """
    按行/列分组求和字典值（适配新命名，避免关键字dict）。
    Args:
        dict_ (dict): 待求和的字典（键为行列标识，值为数值）
        target_row_col_ (int): 目标行/列数（原row/column）
        cond_ (str): 分组条件（"row"按行，"column"按列）
    Returns:
        tuple: 奇数行/列的和、偶数行/列的和
    """
    odd_sum_ = 0.0
    even_sum_ = 0.0
    sum_flag_odd = False  # 标记是否已初始化奇数和
    sum_flag_even = False  # 标记是否已初始化偶数和

    if cond_ == "row":
        for key_, value_ in dict_.items():
            # 排除边缘行（0行和target_row_col_+1行）
            if key_[1] not in (target_row_col_ + 1, 0):
                if key_[1] % 2 == 1:
                    odd_sum_ = value_ if not sum_flag_odd else odd_sum_ + value_
                    sum_flag_odd = True
                else:
                    even_sum_ = value_ if not sum_flag_even else even_sum_ + value_
                    sum_flag_even = True
    elif cond_ == "column":
        for key_, value_ in dict_.items():
            # 排除边缘列（0列和target_row_col_+1列）
            if key_[0] not in (target_row_col_ + 1, 0):
                if key_[0] % 2 == 1:
                    odd_sum_ = value_ if not sum_flag_odd else odd_sum_ + value_
                    sum_flag_odd = True
                else:
                    even_sum_ = value_ if not sum_flag_even else even_sum_ + value_
                    sum_flag_even = True
    return odd_sum_, even_sum_


def find_keys_by_sum(dict_, target_values_):
    """
    根据目标值列表查找字典中对应的键（适配新命名）。
    Args:
        dict_ (dict): 待查找的字典
        target_values_ (tuple): 目标值列表
    Returns:
        list: 匹配值的键列表
    """
    matching_keys_ = []
    for key_, value_ in dict_.items():
        if value_ in target_values_:
            matching_keys_.append(key_)
    return matching_keys_


def find_key_by_value(target_value_, dict_):
    """
    根据目标值查找字典中对应的键（适配新命名，避免关键字dictionary）。
    Args:
        target_value_ (any): 目标值
        dict_ (dict): 待查找的字典
    Returns:
        any: 匹配值的键（无匹配则返回None）
    """
    for key_, value_ in dict_.items():
        if value_ == target_value_:
            return key_
    return None


def combine_dicts(dict1_, dict2_):
    """
    合并两个字典（保留dict2的所有键值，添加dict1中dict2没有的值）。
    Args:
        dict1_ (dict): 待合并的字典1
        dict2_ (dict): 待合并的字典2（优先保留）
    Returns:
        dict: 合并后的新字典
    """
    new_dict_ = dict2_.copy()
    for key_, value_ in dict1_.items():
        if value_ not in dict2_.values():
            new_dict_[key_] = value_
    return new_dict_


def abaqus_create_model(model_name_, fibered_result_, row_count_, col_count_, scale_factor):
    """
    核心建模函数：根据新的纤维化结果字典创建Abaqus纱线模型（原ABAQUScreatemodel）。
    自动适配nf=1（单纤维）和nf>1（多纤维）的数据格式，无需手动处理坐标重组。

    Args:
        model_name_ (str): 模型名称
        fibered_result_ (dict): 新纤维化函数输出的结果字典，含：
            - fibered_bundle_knot: 纤维化束结端坐标（list[tuple[3]]）
            - fibered_yarn_connection: 纤维化连接端坐标（list[tuple[3]]）
            - control_end_coords: 控制点坐标（list[tuple[3]]）
            - virtual_fiber_count: 单根纱线的虚拟纤维数（int）
        row_count_ (int): 控制点行数（原nrow）
        col_count_ (int): 控制点列数（原ncolumn）

    Returns:
        tuple: (MeshEdges_yarn, MeshInstances_yarn, rp_dict)
            - MeshEdges_yarn: 所有纱线单元的边集合
            - MeshInstances_yarn: 所有纱线实例的集合
            - rp_dict: 参考点字典（键为行列标识，值为参考点对象）
    """
    global rp_dict
    global rp_dict_copy

    # 1. 从纤维化结果字典提取核心数据
    fibered_bundle_ = fibered_result_["fibered_bundle_knot"]  # 纤维化束结端
    fibered_conn_ = fibered_result_["fibered_yarn_connection"]  # 纤维化连接端
    control_coords_ = fibered_result_["control_end_coords"]  # 控制点坐标
    virtual_fiber_count_ = fibered_result_["virtual_fiber_count"]  # 单根纱线的虚拟纤维数
    total_yarn_count_ = len(control_coords_)  # 总纱线数 = 控制点数量

    # 2. 重组points_array
    # 场景1：nf=1（单纤维，每根纱线1个点）
    if virtual_fiber_count_ == 1:
        points_array_ = np.array([fibered_bundle_, fibered_conn_])  # shape: (2, 总纱线数, 3)
    # 场景2：nf>1（多纤维，每根纱线含virtual_fiber_count_个点）
    else:
        # 将扁平的纤维坐标按纱线分组（每组含virtual_fiber_count_个纤维点）
        bundle_arr_ = np.array(fibered_bundle_).reshape(total_yarn_count_, virtual_fiber_count_, 3)
        conn_arr_ = np.array(fibered_conn_).reshape(total_yarn_count_, virtual_fiber_count_, 3)

        points_array_ = np.array([bundle_arr_, conn_arr_])          # shape: (2, 总纱线数, 纤维数, 3)

    # 3. 初始化装配体与临时变量
    assembly_ = mdb.models[model_name_].rootAssembly
    mesh_edges_yarn = None  # 存储所有纱线的边
    mesh_instances_yarn = ()  # 存储所有纱线的实例
    vertices0 = ()  # 存储所有纱线的束结端顶点（用于施加固定约束）
    init_flag = False  # 标记是否初始化边/实例集合

    # 4. 循环创建每根纱线的Part、实例与截面
    for yarn_idx_ in range(points_array_.shape[1]):  # 遍历每根纱线（points_array_.shape[1] = 总纱线数）
        yarn_connect_vert = None  # 存储当前纱线的连接端顶点（用于耦合）
        other_fiber_vertices = ()  # 存储当前纱线的所有纤维连接端顶点
        fiber_init_flag = False  # 标记是否初始化当前纱线的纤维顶点

        # 遍历当前纱线的每根纤维（nf>1时需循环，nf=1时循环1次）
        fiber_loop_count_ = points_array_.shape[2] if points_array_.shape[2] != 3 else 1
        for fiber_idx_ in range(fiber_loop_count_):
            # 定义Part、集合、实例的名称（避免重复）
            part_name_ = f"yarn_{yarn_idx_}_{fiber_idx_}"
            set_name_ = f"set_yarn_{yarn_idx_}_{fiber_idx_}"
            instance_name_ = f"instance_yarn_{yarn_idx_}_{fiber_idx_}"

            # 4.1 创建纱线Part（可变形体，3D）
            yarn_part_ = mdb.models[model_name_].Part(
                name=part_name_,
                dimensionality=THREE_D,
                type=DEFORMABLE_BODY
            )

            # 4.2 绘制纱线线段（束结端→连接端）
            if virtual_fiber_count_ == 1:
                # nf=1：直接取当前纱线的束结端和连接端
                start_point_ = points_array_[0][yarn_idx_]
                end_point_ = points_array_[1][yarn_idx_]
            else:
                # nf>1：取当前纱线当前纤维的束结端和连接端
                start_point_ = points_array_[0][yarn_idx_][fiber_idx_]
                end_point_ = points_array_[1][yarn_idx_][fiber_idx_]

            yarn_part_.WirePolyLine(
                points=(start_point_, end_point_),
                mergeType=SEPARATE,
                meshable=ON
            )

            # 4.3 创建纤维边集合（用于分配截面）
            fiber_edge_set_ = yarn_part_.Set(
                edges=yarn_part_.edges,
                name=set_name_
            )

            # 4.4 分配桁架截面（使用提前创建的"Section-fiber"）
            yarn_part_.SectionAssignment(
                region=fiber_edge_set_,
                sectionName='Section-fiber'
            )

            # 4.5 创建Part实例并添加到装配体
            yarn_instance_ = assembly_.Instance(
                name=instance_name_,
                part=yarn_part_,
                dependent=OFF
            )

            # 4.6 收集全局边、实例与顶点（用于后续约束）
            if not init_flag:
                mesh_edges_yarn = yarn_instance_.edges
                mesh_instances_yarn = (yarn_instance_,)
                vertices0 = (yarn_instance_.vertices[0],)  # 束结端顶点（固定约束用）
                init_flag = True
            else:
                mesh_edges_yarn += yarn_instance_.edges
                mesh_instances_yarn += (yarn_instance_,)
                vertices0 += (yarn_instance_.vertices[0],)

            # 4.7 收集当前纱线的连接端顶点（用于耦合）
            if not fiber_init_flag:
                yarn_connect_vert = yarn_instance_.vertices[1]  # 当前纱线的第一个纤维连接端顶点
                fiber_init_flag = True
            else:
                other_fiber_vertices += (yarn_instance_.vertices[1],)  # 其他纤维连接端顶点

        # 5. 耦合当前纱线的所有纤维连接端顶点
        if virtual_fiber_count_ > 1:
            # 创建耦合用的集合（主点：纤维连接端顶点集合；从点：第一个纤维连接端顶点）
            main_vert_array_ = VertexArray(other_fiber_vertices)
            slave_vert_array_ = VertexArray((yarn_connect_vert,))

            main_set_ = assembly_.Set(
                vertices=main_vert_array_,
                name=f"set_coupling_main_{yarn_idx_}"
            )
            slave_set_ = assembly_.Set(
                vertices=slave_vert_array_,
                name=f"set_coupling_slave_{yarn_idx_}"
            )

            # 创建运动耦合约束（6个自由度全耦合）
            mdb.models[model_name_].Coupling(
                name=f"constraint_coupling_{yarn_idx_}",
                controlPoint=main_set_,
                surface=slave_set_,
                influenceRadius=WHOLE_SURFACE,
                couplingType=KINEMATIC,
                alpha=0.0,
                localCsys=None,
                u1=ON, u2=ON, u3=ON,
                ur1=ON, ur2=ON, ur3=ON
            )

        # 6. 创建参考点（RP）与弹簧
        # 6.1 在当前纱线的控制点位置创建参考点
        assembly_.ReferencePoint(point=control_coords_[yarn_idx_])
        rp_all_ = assembly_.referencePoints
        rp_current_ = rp_all_[rp_all_.keys()[0]]  # 获取最新创建的参考点（末尾元素）
        rp_set_ = assembly_.Set(
            referencePoints=(rp_current_,),
            name=f"set_rp_{yarn_idx_}"
        )

        # 6.2 为参考点添加点质量（显式动力学必需）
        mdb.models[model_name_].rootAssembly.engineeringFeatures.PointMassInertia(
            name=f"inertia_rp_{yarn_idx_}",
            region=rp_set_,
            mass=yarn_density_v * scale_factor,  # 质量单位：tonne
            alpha=0.0,
            composite=0.0
        )
        mdb.models[model_name_].rootAssembly.engineeringFeatures.inertias[f"inertia_rp_{yarn_idx_}"].setValues(
            i11=1e-9*scale_factor, i22=1e-9*scale_factor, i33=1e-9*scale_factor
        )

        # 6.3 绘制弹簧线段（纤维连接端→参考点）
        assembly_.WirePolyLine(
            points=((yarn_connect_vert, rp_current_),),
            mergeType=IMPRINT,
            meshable=OFF
        )

        # 6.4 为弹簧线段分配非线性弹簧截面
        spring_edge_set_ = assembly_.Set(
            edges=EdgeArray((assembly_.edges[0],)),  # 最新创建的线段（末尾元素）
            name=f"set_spring_wire_{yarn_idx_}"
        )
        assembly_.SectionAssignment(
            sectionName='ConnSect-nonlinearSpring',
            region=spring_edge_set_
        )

        # 6.5 记录参考点到全局字典（键为行列标识）
        print(points_array_)
        total_rp_count_ = points_array_.shape[1]  # 参考点总数 = 纱线数
        rp_key_ = get_keys_of_rps(yarn_idx_, row_count_, col_count_, total_rp_count_)
        rp_dict[rp_key_] = rp_current_

    # 7. 为所有纱线的束结端施加固定约束（Encastre）
    vertices0_array_ = VertexArray(vertices0)
    fixed_set_ = assembly_.Set(
        vertices=vertices0_array_,
        name="set_fixed_bundle"
    )
    mdb.models[model_name_].EncastreBC(
        name='bc_fixed_bundle',
        createStepName='Initial',
        region=fixed_set_,
        localCsys=None
    )

    # 8. 复制参考点字典（供后续边界条件使用）
    rp_dict_copy = copy.copy(rp_dict)

    return mesh_edges_yarn, mesh_instances_yarn, rp_dict


def abaqus_mesh(model_name_, mesh_edges_, mesh_instances_, fiber_diameter_, length_of_element_):
    """
    在Abaqus中为纱线划分网格（原ABAQUSmesh，适配新命名规范）。
    基于纤维直径计算种子尺寸，使用T3D2单元（显式动力学桁架单元）。

    Args:
        model_name_ (str): 模型名称
        mesh_edges_: 所有纱线的边集合（来自abaqus_create_model的返回值）
        mesh_instances_: 所有纱线的实例集合（来自abaqus_create_model的返回值）
        fiber_diameter_ (float): 单根纤维直径（单位：mm），用于计算网格种子尺寸
    """
    # 获取装配体
    assembly_ = mdb.models[model_name_].rootAssembly

    # 创建所有纱线边的集合（用于网格划分）
    assembly_.Set(edges=mesh_edges_, name='set_yarn_to_mesh')
    yarn_mesh_set_ = assembly_.sets['set_yarn_to_mesh']

    # 按纤维直径的2.5倍设置边种子尺寸（原逻辑：5*d_fiber/2）

    seed_size_ = 2.5 * fiber_diameter_ if length_of_element_ == -1.0 else length_of_element_
    assembly_.seedEdgeBySize(
        edges=mesh_edges_,
        size=seed_size_,
        deviationFactor=0.05,
        constraint=FINER
    )

    # 定义单元类型（显式动力学桁架单元T3D2）
    elem_type_yarn_ = mesh.ElemType(elemCode=T3D2, elemLibrary=EXPLICIT)

    # 为纱线边分配单元类型
    assembly_.setElementType(
        regions=yarn_mesh_set_,
        elemTypes=(elem_type_yarn_,)
    )

    # 生成网格
    assembly_.generateMesh(regions=mesh_instances_)


def abaqus_bc_initial(model_name_, rp_dict_):
    """
    为所有参考点（RP）创建初始位移边界条件（原ABAQUS_BCInitial，适配新命名）。
    初始状态下所有参考点的位移自由度设为"SET"（后续分析步中修改）。

    Args:
        model_name_ (str): 模型名称
        rp_dict_ (dict): 参考点字典（键为行列标识，值为参考点对象，来自abaqus_create_model）
    """
    assembly_ = mdb.models[model_name_].rootAssembly

    # 遍历所有参考点，创建边界条件
    for rp_key_, rp_value_ in rp_dict_.items():
        # 生成唯一的集合名称和边界条件名称（用行列标识区分）
        set_name_ = f"set_rp_{int(rp_key_[0])}_{int(rp_key_[1])}"
        bc_name_ = f"bc_rp_{int(rp_key_[0])}_{int(rp_key_[1])}"

        # 创建参考点的集合
        rp_set_ = assembly_.Set(referencePoints=(rp_value_, ), name=set_name_)

        # 创建位移边界条件（U1/U2/U3设为SET，转角自由度UNSET）
        mdb.models[model_name_].DisplacementBC(
            name=bc_name_,
            createStepName='Initial',
            region=rp_set_,
            u1=SET, u2=SET, u3=SET,
            ur1=UNSET, ur2=UNSET, ur3=UNSET,
            amplitude=UNSET,
            distributionType=UNIFORM,
            fieldName='',
            localCsys=None
        )
    return


def abaqus_step(model_name_, step_count_, space_, row_count_, col_count_,
                rp_dict_, rp_dict_copy_, time_period_=20, min_time_increment=0.00001):
    """
    创建显式动力学分析步，更新参考点边界条件并迭代更新参考点字典（原ABAQUSstep）。
    适配新命名规范，移除全局变量依赖，通过参数传入关键数据。

    Args:
        model_name_ (str): 模型名称
        step_count_ (int): 总分析步数
        space_ (float): 参考点位移步长（单位：mm）
        row_count_ (int): 控制点行数（原nrow）
        col_count_ (int): 控制点列数（原ncolumn）
        rp_dict_ (dict): 参考点字典（当前状态，需在分析步中更新）
        rp_dict_copy_ (dict): 参考点初始字典（用于追溯初始键）
        time_period_ (float): 每个分析步的时间周期，默认20
    """
    # 初始化模型对象
    model_ = mdb.models[model_name_]
    mdb.models[model_name].SmoothStepAmplitude(name=Amplitude,
                                               timeSpan=STEP, data=((0.0, 0.0), (time_period_, 1.0)))
    # 创建初始边界条件（所有参考点U1/U2/U3设为SET）
    abaqus_bc_initial(model_name_, rp_dict_)

    # 遍历所有分析步，创建步骤并更新边界条件
    for step_idx_ in range(step_count_):
        # 计算当前步骤的i（组号）和j（组内序号，1-4循环）
        i_ = int(step_idx_ // 4 + 1)
        j_ = int(step_idx_ % 4 + 1)
        step_name_ = f"step_{i_}_{j_}"

        # 确定前一步骤名称
        if i_ == 1 and j_ == 1:
            previous_step_ = "Initial"
        else:
            if j_ == 1:
                previous_step_ = f"step_{i_ - 1}_4"
            else:
                previous_step_ = f"step_{i_}_{j_ - 1}"
        if min_time_increment == -1:
            model.ExplicitDynamicsStep(name=step_name_, previous=previous_step_, timePeriod=time_period_, improvedDtMethod=ON)
        else:
            # 创建显式动力学分析步（含质量缩放）
            model_.ExplicitDynamicsStep(
                name=step_name_,
                previous=previous_step_,
                timePeriod=time_period_,
                 massScaling=(
                     (SEMI_AUTOMATIC, MODEL, AT_BEGINNING,
                      1.0, min_time_increment, BELOW_MIN, 0, 0, 0.0, 0.0, 0, None),
                ),
                improvedDtMethod=ON
            )

        # # 设置场输出（应力、应变、位移等）
        # model_.FieldOutputRequest(
        #     name=f"f_output_{i_}_{j_}",
        #     createStepName=step_name_,
        #     variables=('S', 'E', 'U', 'RF', 'CF', 'CSTRESS'),
        #     numIntervals=10
        # )

        # 获取奇偶性判断结果（用于确定位移方向）
        judgement_list_ = parity_judgement(row_count_, col_count_)
        update_dict_ = {}  # 存储当前步更新后的参考点键值对

        # 奇数子步（j=1,3）：沿Y方向移动
        if j_ % 2 == 1:
            factor1_ = -1 if j_ == 1 else 1  # j=1负方向，j=3正方向
            factor3_ = -1 if judgement_list_[1] == 0 else 1  # 基于列奇偶性的方向因子

            for rp_key_, rp_value_ in rp_dict_.items():
                # 追溯初始键（用于定位边界条件）
                initial_key_ = find_key_by_value(rp_value_, rp_dict_copy_)
                bc_name_ = f"bc_rp_{int(initial_key_[0])}_{int(initial_key_[1])}"

                # 非边缘行的参考点：沿Y方向移动
                if rp_key_[0] not in (0, row_count_ + 1):
                    factor2_ = 1 if rp_key_[0] % 2 == 1 else -1  # 基于行奇偶性的方向因子
                    # 更新边界条件（U2设为位移值，U1/U3固定为0）
                    model_.boundaryConditions[bc_name_].setValuesInStep(
                        stepName=step_name_,
                        u1=0,
                        u2=factor1_ * factor2_ * factor3_ * space_,
                        u3=0,
                        amplitude=Amplitude
                    )
                    # 更新参考点键（Y方向移动后）
                    new_key_ = (rp_key_[0], rp_key_[1] + factor1_ * factor2_ * factor3_ * 1)
                    update_dict_[new_key_] = rp_value_

                # 边缘行的参考点：固定不动
                else:
                    model_.boundaryConditions[bc_name_].setValuesInStep(
                        stepName=step_name_,
                        u1=0, u2=0, u3=0,
                        amplitude=Amplitude
                    )
                    update_dict_[rp_key_] = rp_value_

        # 偶数子步（j=2,4）：沿X方向移动
        else:
            factor1_ = 1 if j_ == 2 else -1  # j=2正方向，j=4负方向

            for rp_key_, rp_value_ in rp_dict_.items():
                # 追溯初始键（用于定位边界条件）
                initial_key_ = find_key_by_value(rp_value_, rp_dict_copy_)
                bc_name_ = f"bc_rp_{int(initial_key_[0])}_{int(initial_key_[1])}"

                # 非边缘列的参考点：沿X方向移动
                if rp_key_[1] not in (0, col_count_ + 1):
                    factor2_ = 1 if rp_key_[1] % 2 == 1 else -1  # 基于列奇偶性的方向因子
                    # 更新边界条件（U1设为位移值，U2/U3固定为0）
                    model_.boundaryConditions[bc_name_].setValuesInStep(
                        stepName=step_name_,
                        u1=factor1_ * factor2_ * space_,
                        u2=0,
                        u3=0,
                        amplitude=Amplitude
                    )
                    # 更新参考点键（X方向移动后）
                    new_key_ = (rp_key_[0] + factor1_ * factor2_ * 1, rp_key_[1])
                    update_dict_[new_key_] = rp_value_

                # 边缘列的参考点：固定不动
                else:
                    model_.boundaryConditions[bc_name_].setValuesInStep(
                        stepName=step_name_,
                        u1=0, u2=0, u3=0,
                        amplitude=Amplitude
                    )
                    update_dict_[rp_key_] = rp_value_

        # 更新参考点字典（用于下一步迭代）
        rp_dict_.clear()
        rp_dict_.update(update_dict_)

    return rp_dict_  # 返回最终更新的参考点字典


if __name__ == "__main__":

    if min_time_increment != -1:
        k_scale_factor, density_scale_factor = optimize(d_fiber / 2, youngs_module, length_of_element_=2.5 * d_fiber,
                                                    target_time_increment_=min_time_increment)
    else:
        k_scale_factor, density_scale_factor = 1, 1
    # 调用纤维化函数，获取所有建模所需点信息
    fibered_result = generate_fibered_yarn_all_points(
        row_count_=row_count,
        col_count_=col_count,
        control_node_spacing_=control_node_spacing,
        fiber_diameter_=d_fiber,
        yarn_length_=yarn_length,
        virtual_fiber_count_=virtual_fiber_count,
        height_=height,
        yarn_interval_factor_=yarn_interval_factor,
        fiber_interval_factor_=fiber_interval_factor,
        plot_hex_pack=False,  # 可视化纤维密排（可选）
        plot_translate=False,  # 可视化纤维平移（可选）
        plot_final=True  # 可视化最终结果（可选）
    )

    # 打印关键信息（供用户验证）
    print("=" * 50)
    print("Fibered Yarn Result Summary:")
    print(f"Virtual Fiber Count: {fibered_result['virtual_fiber_count']}")
    print(f"Fiber Diameter (mm): {fibered_result['d_fiber']:.6f}")
    print(f"Tow Diameter (mm): {fibered_result['d_yarn']:.6f}")
    print(f"Control Points Count: {len(fibered_result['control_end_coords'])}")
    print(f"Fibered Bundle Knots Count: {len(fibered_result['fibered_bundle_knot'])}")
    print("=" * 50)


    # 1. 计算材料密度
    # compute_density = yarn_density_v * 1000000000
    compute_density = yarn_density_v
    compute_spring_k = spring_k * k_scale_factor
    compute_youngs_module = youngs_module
    compute_yarn_elastic = (compute_youngs_module, poissons_ratio)  # 弹性参数元组（供后续材料赋值）
    # 2. 调用适配后的工具函数
    abaqus_material(
        model_name_=model_name,  # model_name为全局变量，如"BraidedSimulia"
        elastic_params_=compute_yarn_elastic,  # yarn_elastic = (youngs_module, poissons_ratio)
        density_=compute_density
    )
    abaqus_section(
        model_name_=model_name,
        fiber_diameter_=fibered_result["d_fiber"]  # 从字典取单纤维直径
    )
    abaqus_nonlinear_spring_connector(
        model_name_=model_name,
        spring_rl_=spring_rl,  # 原springRL参数
        spring_k_table_=((compute_spring_k,),)  # 原springK参数，注意是元组格式
    )
    abaqus_interaction(
        model_name_=model_name,
        friction_factor_=friction_factor  # 原friction_factor参数
    )

    # 3. 调用核心建模函数（关键：传入fibered_result字典+行列数）
    mesh_edges, mesh_instances, rp_dict = abaqus_create_model(
        model_name_=model_name,
        fibered_result_=fibered_result,
        row_count_=row_count,
        col_count_=col_count,
        scale_factor=density_scale_factor
    )

    # 4. 划分网格
    abaqus_mesh(
        model_name_=model_name,
        mesh_edges_=mesh_edges,
        mesh_instances_=mesh_instances,
        fiber_diameter_=fibered_result["d_fiber"],  # 从纤维化结果取纤维直径
        length_of_element_=length_of_element
    )

    # 5. 创建分析步（假设16步，步长space=control_node_spacing）
    final_rp_dict = abaqus_step(
        model_name_=model_name,
        step_count_=analysis_steps,  # 原analysis_steps参数
        space_=control_node_spacing,  # 原control_node_spacing参数
        row_count_=row_count,  # 行数
        col_count_=col_count,  # 列数
        rp_dict_=rp_dict,  # 当前参考点字典
        rp_dict_copy_=rp_dict_copy,  # 初始参考点字典
        time_period_=time_period  # 每步时间周期
    )

    # 6.生成任务
    job_name = f'Job-r{int(row_count)}c{int(col_count)}-step{int(analysis_steps)}'

    # 创建Abaqus分析任务（参数名与前文变量对应，保持一致性）
    mdb.Job(
        name=job_name,
        model=model_name,  # 与前文定义的模型名称变量对应
        description='',
        type=ANALYSIS,
        atTime=None,
        waitMinutes=0,
        waitHours=0,
        queue=None,
        memory=90,
        memoryUnits=PERCENTAGE,
        explicitPrecision=SINGLE,
        nodalOutputPrecision=SINGLE,
        echoPrint=OFF,
        modelPrint=OFF,
        contactPrint=OFF,
        historyPrint=OFF,
        userSubroutine='',
        scratch='',
        resultsFormat=ODB,
        numDomains=int(cpus),  # 与前文CPU核心数变量对应
        activateLoadBalancing=False,
        numThreadsPerMpiProcess=1,
        multiprocessingMode=DEFAULT,
        numCpus=int(cpus)  # 与前文CPU核心数变量对应
    )

    # 视图显示设置（保持功能不变，格式更规范）
    # 渲染梁剖面
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(renderBeamProfiles=ON)
    # 显示边界条件、连接器、弹簧单元
    session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(
        renderBeamProfiles=ON,  # 渲染梁剖面
        bcDisplay=ON,  # 显示边界条件
        connectorDisplay=ON,  # 显示连接器
        springElements=ON  # 显示弹簧单元
    )