cmake_minimum_required(VERSION 2.8.3)
project(Task)
set(CMAKE_BUILD_TYPE "release")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message("No appropriate opencv package found.")
endif()

find_package(Eigen3 REQUIRED)
if (NOT Eigen3_FOUND)
    message("No appropriate libeigen3 package found.")
endif()

find_package(Ceres REQUIRED)
if (NOT Ceres_FOUND)
    message("No appropriate ceres package found.")
endif()

include_directories(
    .
    ./include
    /usr/local/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
)

link_directories(
    /usr/local/lib
)

aux_source_directory(
    src/  SRC
)

add_executable(Task testSimplex.cc
    ${SRC}
)

target_link_libraries(Task
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
)