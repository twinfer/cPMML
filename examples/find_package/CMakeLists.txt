# The smallest thing a consumer can do: find the installed package, link it,
# and catch one of its exceptions by type. CI builds this against a fresh
# `cmake --install` so that a break in the export or in the exported symbol
# visibility fails a build rather than a user's.
cmake_minimum_required(VERSION 3.14)
project(cpmml_find_package_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(cPMML 2.0 REQUIRED)

add_executable(find_package_example main.cc)
target_link_libraries(find_package_example PRIVATE cPMML::cPMML)
