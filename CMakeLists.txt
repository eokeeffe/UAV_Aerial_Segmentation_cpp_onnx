cmake_minimum_required(VERSION 3.5)
project( Aerial_Segmentation_ONNX )

# OpenCV project requirements
find_package( OpenCV REQUIRED )
if(NOT OPENCV_FOUND)
    message("Couldn't find OpenCV")
else()
    include_directories( ${OpenCV_INCLUDE_DIRS} )
endif()

find_package(onnxruntime)
if(NOT ONNXRUNTIME_FOUND)
    message("Couldn't find onnxruntime")
else()
    include_directories(${onnxruntime_INCLUDE_DIRS}/onnxruntime/include)
endif()

include_directories("${PROJECT_SOURCE_DIR}/include/")

# create the executable
add_executable( aerial_segment src/main.cpp src/ort_session_handler.cpp)
target_link_libraries( aerial_segment ${OpenCV_LIBS} ${onnxruntime_LIBRARY})
