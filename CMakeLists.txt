cmake_minimum_required(VERSION 3.8)
project(MyProject)

# 使用C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置Eigen库路径
set(EIGEN3_INCLUDE_DIRS "D:/lib/eigen")

# 手动指定 Qt5 的安装路径
# set(CMAKE_PREFIX_PATH "D:/lib/Qt/5.15.0/msvc2019_64")  # 请根据你系统中 Qt 的路径进行设置

# 查找 Qt5 库
find_package(Qt5 COMPONENTS Widgets REQUIRED)
# 为 Qt5 处理 UI（moc/uic/rcc 等）
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)


# 收集所有源文件
file(GLOB SOURCES "src/*")

# 添加可执行文件
add_executable(MyExecutable ${SOURCES})

# 包含Eigen头文件
target_include_directories(MyExecutable PUBLIC ${EIGEN3_INCLUDE_DIRS})

# 链接 Qt5 Widgets 库
target_link_libraries(MyExecutable Qt5::Widgets)



