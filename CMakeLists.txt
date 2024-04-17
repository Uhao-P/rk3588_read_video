cmake_minimum_required(VERSION 3.0.0)
project(rga_demo)

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(
    include_libs  
    /home/linaro/librga/build/build_linux/install/include/
    /home/linaro/mpp/osal/inc/
    /home/linaro/mpp/utils/
    /home/linaro/mpp/inc/
    /home/linaro/mpp/mpp/base/inc/
    /usr/include/rga/
   # /usr/include/linux/
)

include_directories(${include_libs})

add_library(mpp_utils STATIC
    /home/linaro/mpp/utils/mpp_enc_roi_utils.c
    /home/linaro/mpp/utils/mpi_enc_utils.c
    /home/linaro/mpp/utils/mpi_dec_utils.c
    /home/linaro/mpp/utils/mpp_opt.c
    /home/linaro/mpp/utils/utils.c
    /home/linaro/mpp/utils/iniparser.c
    /home/linaro/mpp/utils/dictionary.c
    /home/linaro/mpp/utils/camera_source.c
    )


# 添加动态链接库
set(
    dynamic_libs  pthread
    /home/linaro/librga/build/build_linux/librga.so
    /home/linaro/mpp/build/linux/aarch64/mpp/librockchip_mpp.so
    /home/linaro/mpp/build/linux/aarch64/mpp/legacy/librockchip_vpu.so
)





# cam_v4l2_opencv  v4l2直接读摄像头，将读出的数据直接丢入opencv
add_executable(cam_v4l2_opencv
    src/cam_v4l2_opencv.cpp
)
target_link_libraries(cam_v4l2_opencv
    ${include_libs}
    ${dynamic_libs}
    ${OpenCV_LIBS}
    mpp_utils
)


# cam_v4l2_rga  v4l2直接读摄像头，将读出的数据直接丢入rga
add_executable(cam_v4l2_rga
    src/cam_v4l2_rga.cpp
)
target_link_libraries(cam_v4l2_rga
    ${include_libs}
    ${dynamic_libs}
    ${OpenCV_LIBS}
    mpp_utils
)


# cam_opencv  opencv直接读摄像头
add_executable(cam_opencv
    src/cam_opencv.cpp
)
target_link_libraries(cam_opencv
    ${include_libs}
    ${dynamic_libs}
    ${OpenCV_LIBS}
    mpp_utils
)


# cam_mpp_opencv  mpp读摄像头，将读出来的数据直接丢入opencv
add_executable(cam_mpp_opencv
    src/cam_mpp_opencv.cpp
)
target_link_libraries(cam_mpp_opencv
    ${include_libs}
    ${dynamic_libs}
    ${OpenCV_LIBS}
    mpp_utils
)


# cam_mpp_rga  mpp读摄像头，将读出来的数据直接丢入rga
add_executable(cam_mpp_rga
    src/cam_mpp_rga.cpp
)
target_link_libraries(cam_mpp_rga
    ${include_libs}
    ${dynamic_libs}
    ${OpenCV_LIBS}
    mpp_utils
)


# file_opencv  opencv直接读视频文件
add_executable(file_opencv
    src/file_opencv.cpp
)
target_link_libraries(file_opencv
    ${include_libs}
    ${dynamic_libs}
    ${OpenCV_LIBS}
    mpp_utils
)


# file_mpp_rga  mpp读视频文件，将读出来的数据直接丢入rga
add_executable(file_mpp_rga
    src/file_mpp_rga.cpp
)
target_link_libraries(file_mpp_rga
    ${include_libs}
    ${dynamic_libs}
    ${OpenCV_LIBS}
    mpp_utils
)


# resize_demo  对比opencv与rga进行resize比较
add_executable(resize_demo
    src/resize_demo.cpp
)
target_link_libraries(resize_demo
    ${include_libs}
    ${dynamic_libs}
    ${OpenCV_LIBS}
    mpp_utils
)