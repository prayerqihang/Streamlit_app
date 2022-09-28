import os

# 项目根目录路径
BASE_PATH = os.path.dirname(os.path.dirname(__file__))

# 数据文件目录
DATA_PATH_STATIC = os.path.join(
    BASE_PATH, 'db_file', 'data_analysis', 'static_data'
)
DATA_PATH_PASSENGER = os.path.join(
    BASE_PATH, 'db_file', 'data_analysis', 'passenger_data'
)
DATA_PATH_PARKING = os.path.join(
    BASE_PATH, 'db_file', 'machine_learning', 'parking_occupancy_data'
)

# 自定义模块路径
LIB_PATH = os.path.join(
    BASE_PATH, 'lib'
)

# 图像文件目录
IMAGE_PATH_HOME = os.path.join(
    BASE_PATH, 'images', 'home'
)
IMAGE_PATH_PARKING = os.path.join(
    BASE_PATH, 'images', 'machine_learning'
)

# 样式文件路径
STYLE_PATH = os.path.join(
    BASE_PATH, 'style'
)
