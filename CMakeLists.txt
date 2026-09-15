cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_COMPILER_WORKS 1)
project(firm3dpp)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    add_definitions(-DUSE_CUDA)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=native")
    set(CUDA_TARGETS  src/firm3dpp/cuda_kernel.cu src/firm3dpp/cuda_wrapper.cpp)
    message(STATUS "CUDA found. GPU bindings will be compiled.")
else()
    set(CUDA_TARGETS)
    message(STATUS "CUDA not found. GPU bindings will be skipped.")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(Python_FIND_STRATEGY LOCATION)
find_package(Python 3 COMPONENTS Interpreter Development.Module)

message(status "Python executable is ${Python_EXECUTABLE}")
message(status "Python Development Module found value is ${Python_Development.Module_FOUND}")
message(status "Python header dirs are ${Python_INCLUDE_DIRS}")
message(status "Python library dirs are ${Python_LIBRARY_DIRS}")
message(status "Python version is ${Python_VERSION}")

# Find NumPy manually since the NumPy component might not be available
execute_process(COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
                OUTPUT_VARIABLE Python_NumPy_INCLUDE_DIRS
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE NUMPY_IMPORT_RESULT)
if(NUMPY_IMPORT_RESULT EQUAL 0)
    message(status "NumPy include dirs are ${Python_NumPy_INCLUDE_DIRS}")
else()
    message(WARNING "NumPy not found during build. This is expected in the build environment.")
    # Set a default path that will be overridden at runtime
    set(Python_NumPy_INCLUDE_DIRS "${Python_INCLUDE_DIRS}")
endif()

# Find pybind11
execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                OUTPUT_VARIABLE pybind11_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)
message("Result of python -m pybind11 --cmakedir: ${pybind11_DIR}")
find_package(pybind11 REQUIRED PATHS pybind11_DIR)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

# For XSIMD usage
if(DEFINED ENV{NO_XSIMD})
    set(NO_XSIMD ON)
endif()
set(_HAS_AUTO_PTR_ETC 1)
configure_file(config.h.in config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

include(CheckCXXCompilerFlag)
IF(DEFINED ENV{CI})
    if (APPLE)
        set(CMAKE_CXX_FLAGS "-O3")
    else()
        message(STATUS "CI environment detected. Set compilation flags targetting Westmere microarch.")
        set(CMAKE_CXX_FLAGS "-O3 -march=westmere")
    endif()
elseif(DEFINED ENV{CONDA_BUILD})
    message(STATUS "conda build environment detected. Let conda set compilation flags accordingly.")
    # set(CMAKE_CXX_FLAGS "-O3 -march=ivybridge -mfma -ffp-contract=fast")
else()
    # Temporary fix till clang 15 is default on macs with Apple silicon
    message(STATUS "Local build detected. Set compilation flags accordingly (march=native).")
    unset(COMPILER_SUPPORTS_MARCH_NATIVE CACHE)
    CHECK_CXX_COMPILER_FLAG(-march=native COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        set(CMAKE_CXX_FLAGS "-O3 -march=native -ffp-contract=fast")
    elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "arm64")
        set(CMAKE_CXX_FLAGS "-O3 -mcpu=apple-a14 -ffp-contract=fast")
    else()
        set(CMAKE_CXX_FLAGS "-O3 -ffp-contract=fast")
    endif()
endif()
# Ensure all code used from Eigen does not have LGPL license:
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_MPL2_ONLY")
message(STATUS "CMAKE_CXX_FLAGS are ${CMAKE_CXX_FLAGS}")

if(DEFINED ENV{CONDA_PREFIX})
    include_directories($ENV{CONDA_PREFIX}/include)
    link_directories("$ENV{CONDA_PREFIX}/lib")
    message(STATUS "Conda prefix is $ENV{CONDA_PREFIX}")
endif()

find_package(OpenMP)

# Find MPI (optional, for parallel interpolation)
find_package(MPI QUIET)
if(MPI_FOUND AND MPI_CXX_FOUND)
    # Cray PE wrappers (cc/CC) absorb MPI include paths internally, so FindMPI
    # leaves MPI_CXX_INCLUDE_DIRS empty. Recover the path from CRAY_MPICH_DIR.
    if(NOT MPI_CXX_INCLUDE_DIRS AND DEFINED ENV{CRAY_MPICH_DIR})
        set(MPI_CXX_INCLUDE_DIRS "$ENV{CRAY_MPICH_DIR}/include")
        set_property(TARGET MPI::MPI_CXX PROPERTY
            INTERFACE_INCLUDE_DIRECTORIES "$ENV{CRAY_MPICH_DIR}/include")
        message(STATUS "Cray MPI: recovered include dirs from CRAY_MPICH_DIR")
    endif()
    message(STATUS "MPI found: ${MPI_VERSION}")
    message(STATUS "MPI CXX compiler: ${MPI_CXX_COMPILER}")
    message(STATUS "MPI CXX include dirs: ${MPI_CXX_INCLUDE_DIRS}")
    message(STATUS "MPI CXX libraries: ${MPI_CXX_LIBRARIES}")
    set(USE_MPI TRUE)
    add_definitions(-DUSE_MPI)
else()
    message(STATUS "MPI not found. MPI parallelization will be disabled.")
    message(STATUS "To enable MPI, install MPI development libraries and ensure MPI_CXX_COMPILER is set.")
    set(USE_MPI FALSE)
endif()

set(BOOST_MINIMUM_VERSION 1.76.0)
set(_boost_user_include_dir "${Boost_INCLUDE_DIR}")

# Preferred route: an install that ships BoostConfig.cmake.
find_package(Boost ${BOOST_MINIMUM_VERSION} QUIET)
if(Boost_FOUND AND NOT Boost_INCLUDE_DIRS)
    message(STATUS "find_package(Boost) provided no include dirs. Searching for headers instead.")
    set(Boost_FOUND FALSE)
elseif(Boost_FOUND)
    message(STATUS "Boost located by find_package.")
endif()

# CMake deprecated FindBoost in 3.30 (CMP0167) and is retiring it, after which
# the call above only succeeds in CONFIG mode against an install that ships
# BoostConfig.cmake. Header-only installs (conda, pip, some system packages)
# ship no such config, and where FindBoost still runs it may see nothing but a
# too-old system Boost. Either way the download below would be triggered on
# every fresh configure, so locate the headers directly instead.
if(NOT Boost_FOUND)
    unset(Boost_INCLUDE_DIR CACHE)
    set(_boost_hints)
    if(_boost_user_include_dir)
        list(APPEND _boost_hints "${_boost_user_include_dir}")
    endif()
    foreach(_boost_root "${BOOST_ROOT}" "${BOOSTROOT}" "$ENV{BOOST_ROOT}"
                        "$ENV{BOOSTROOT}" "$ENV{CONDA_PREFIX}")
        if(_boost_root)
            list(APPEND _boost_hints "${_boost_root}/include" "${_boost_root}")
        endif()
    endforeach()
    if(Python_EXECUTABLE)
        get_filename_component(_python_prefix "${Python_EXECUTABLE}" DIRECTORY)
        get_filename_component(_python_prefix "${_python_prefix}" DIRECTORY)
        list(APPEND _boost_hints "${_python_prefix}/include")
    endif()

    find_path(Boost_INCLUDE_DIR
        NAMES boost/version.hpp
        HINTS ${_boost_hints}
        PATHS /usr/local/include /usr/include /opt/local/include /opt/homebrew/include
        DOC "Directory holding the Boost header tree (the parent of boost/)")

    set(_boost_version_string)
    if(Boost_INCLUDE_DIR AND EXISTS "${Boost_INCLUDE_DIR}/boost/version.hpp")
        file(STRINGS "${Boost_INCLUDE_DIR}/boost/version.hpp" _boost_version_line
             REGEX "^#define[ \t]+BOOST_VERSION[ \t]+[0-9]+")
        if(_boost_version_line MATCHES "BOOST_VERSION[ \t]+([0-9]+)")
            # BOOST_VERSION is encoded as major * 100000 + minor * 100 + patch.
            math(EXPR _boost_major "${CMAKE_MATCH_1} / 100000")
            math(EXPR _boost_minor "${CMAKE_MATCH_1} / 100 % 1000")
            math(EXPR _boost_patch "${CMAKE_MATCH_1} % 100")
            set(_boost_version_string "${_boost_major}.${_boost_minor}.${_boost_patch}")
        endif()
    endif()

    # Minimal installs sometimes ship a partial tree, so check that the headers
    # firm3d actually includes are present before trusting the discovery.
    set(_boost_missing_headers)
    foreach(_header boost/format.hpp boost/math/tools/roots.hpp boost/numeric/odeint.hpp)
        if(NOT EXISTS "${Boost_INCLUDE_DIR}/${_header}")
            list(APPEND _boost_missing_headers "${_header}")
        endif()
    endforeach()

    if(NOT Boost_INCLUDE_DIR OR NOT EXISTS "${Boost_INCLUDE_DIR}/boost/version.hpp")
        message(STATUS "No Boost headers found in BOOST_ROOT, CMAKE_PREFIX_PATH, the Python prefix or the system paths.")
    elseif(NOT _boost_version_string)
        message(STATUS "Could not read a Boost version from ${Boost_INCLUDE_DIR}/boost/version.hpp.")
    elseif(_boost_version_string VERSION_LESS ${BOOST_MINIMUM_VERSION})
        message(STATUS "Boost ${_boost_version_string} in ${Boost_INCLUDE_DIR} is older than the required ${BOOST_MINIMUM_VERSION}.")
    elseif(_boost_missing_headers)
        message(STATUS "Boost headers in ${Boost_INCLUDE_DIR} are incomplete, missing ${_boost_missing_headers}.")
    else()
        set(Boost_FOUND TRUE)
        set(Boost_VERSION_STRING ${_boost_version_string})
        set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIR})
        message(STATUS "Boost located by header search (find_package found no usable package).")
    endif()
endif()

if(Boost_FOUND)
    message(STATUS "Boost version is ${Boost_VERSION_STRING}")
    message(STATUS "Boost include dirs are ${Boost_INCLUDE_DIRS}")
else()
    message(STATUS "Downloading and installing boost.")
    # For some external project macros
    include(ExternalProject)

    # Download boost from git and build the headers
    set(boost_DIR ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/boost)
    set(boost_target boost)

    ExternalProject_Add(
        ${boost_target}
        PREFIX ${boost_DIR}
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG boost-1.84.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        GIT_SUBMODULES tools/build tools/boost_install libs/config libs/numeric
            libs/math libs/type_traits libs/predef libs/assert libs/static_assert
            libs/throw_exception libs/core libs/serialization libs/preprocessor libs/mpl
            libs/utility libs/typeof libs/array libs/units libs/integer libs/fusion
            libs/range libs/iterator libs/concept_check libs/detail libs/function_types
            libs/lexical_cast libs/container libs/move libs/smart_ptr libs/multi_array
            libs/functional libs/function libs/type_index libs/container_hash libs/bind
            libs/format libs/optional
        CONFIGURE_COMMAND ./bootstrap.sh --prefix=<PREFIX>
        BUILD_COMMAND ./b2 headers --prefix=${boost_DIR}
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND ""
    )

    set(Boost_INCLUDE_DIRS ${boost_DIR}/src/${boost_target})
    message(STATUS "Boost include dirs are ${Boost_INCLUDE_DIRS}")
endif()

find_package(GSL)
if(GSL_FOUND)
    add_definitions(-DUSE_GSL)
    message(STATUS "GSL version is ${GSL_VERSION}")
    message(STATUS "GSL include dirs are ${GSL_INCLUDE_DIR}")
    set(GSL_TARGETS src/firm3dpp/symplectic.cpp)
    message(STATUS "GSL targets are ${GSL_TARGETS}")
    message(STATUS "GSL libraries are ${GSL_LIBRARIES}")
else()
    set(GSL_TARGETS)
endif()

set(XTENSOR_USE_OPENMP 0)
set(XTENSOR_USE_TBB 0)

pybind11_add_module(${PROJECT_NAME}
    src/firm3dpp/python.cpp
    src/firm3dpp/python_tracing.cpp
    src/firm3dpp/python_interpolant.cpp
    src/firm3dpp/regular_grid_interpolant_3d_py.cpp
    src/firm3dpp/tracing.cpp
    src/firm3dpp/python_boozermagneticfield.cpp
    src/firm3dpp/boozerradialinterpolant.cpp
    ${CUDA_TARGETS}
    ${GSL_TARGETS}
)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)

# nlohmann-json is provided via the Python nlohmann_json package.
# setup.py finds the include path and passes it to CMake.
if(nlohmann_json_INCLUDE_DIR)
    target_include_directories(${PROJECT_NAME} PRIVATE "thirdparty/xtensor/include" "thirdparty/xtensor-python/include" "thirdparty/xsimd/include" "thirdparty/xtl/include" ${Python_NumPy_INCLUDE_DIRS} "src/firm3dpp/" ${nlohmann_json_INCLUDE_DIR})
else()
    target_include_directories(${PROJECT_NAME} PRIVATE "thirdparty/xtensor/include" "thirdparty/xtensor-python/include" "thirdparty/xsimd/include" "thirdparty/xtl/include" ${Python_NumPy_INCLUDE_DIRS} "src/firm3dpp/")
endif()

if(NOT Boost_FOUND)
    add_dependencies(${PROJECT_NAME} ${boost_target})
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-I${Boost_INCLUDE_DIRS}")

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link MPI if found
if(USE_MPI AND MPI_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX)
    message(STATUS "Linking MPI libraries for parallel interpolation support")
endif()

IF (GSL_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${GSL_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${GSL_LIBRARIES})
endif()
