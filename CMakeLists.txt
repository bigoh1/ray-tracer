cmake_minimum_required(VERSION 3.19)
project(ray_tracer)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH "$HOME/Projects/Libraries/pngwriter-build")
list(APPEND CMAKE_PREFIX_PATH "$HOME/homebrew/Cellar/libpng/1.6.37")
list(APPEND CMAKE_PREFIX_PATH "$HOME/homebrew/Cellar/freetype/2.12.0")
list(APPEND CMAKE_PREFIX_PATH "$HOME/homebrew/Cellar/eigen/3.4.0_1")

find_package(PNGwriter 0.7.0 REQUIRED)
find_package(Eigen3 3.4.0 REQUIRED)

add_executable(ray_tracer main.cpp)

target_link_libraries(ray_tracer PRIVATE PNGwriter::PNGwriter Eigen3::Eigen)

