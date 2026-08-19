cmake_minimum_required(VERSION 3.20)
project(step4_deposition_zone_mpi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(
    TAUDEM_SRC_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/taudem/src"
    CACHE PATH
    "Path to the TauDEM source subset used by Step 4"
)

find_package(MPI REQUIRED)

find_package(GDAL CONFIG QUIET)
if(NOT GDAL_FOUND)
    find_package(GDAL REQUIRED)
endif()

add_executable(step4_deposition_zone_mpi
    step4_deposition_zone_mpi.cpp
    "${TAUDEM_SRC_DIR}/commonLib.cpp"
    "${TAUDEM_SRC_DIR}/tiffIO.cpp"
)

target_include_directories(step4_deposition_zone_mpi PRIVATE
    "${TAUDEM_SRC_DIR}"
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(step4_deposition_zone_mpi PRIVATE
    MPI::MPI_CXX
)

if(TARGET GDAL::GDAL)
    target_link_libraries(step4_deposition_zone_mpi PRIVATE GDAL::GDAL)
else()
    target_include_directories(step4_deposition_zone_mpi PRIVATE ${GDAL_INCLUDE_DIR})
    target_link_libraries(step4_deposition_zone_mpi PRIVATE ${GDAL_LIBRARY})
endif()

if(MSVC)
    target_compile_definitions(step4_deposition_zone_mpi PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
