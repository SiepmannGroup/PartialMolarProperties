# CMake project file for MCCCS Topmon

INCLUDE(CMakeForceCompiler)

cmake_minimum_required(VERSION 2.6)

project(MCCCS Fortran)

# make sure that the default is a RELEASE
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE CACHE STRING "Choose the type of build, options are: None DEBUG RELEASE." FORCE)

  # default installation
  get_filename_component(default_prefix "." ABSOLUTE)
  set(CMAKE_INSTALL_PREFIX "${default_prefix}" CACHE STRING "Choose the installation directory; by default it installs in the current directory." FORCE)
endif()

# Check and warn if cache generated on a different host is being reused
if(CMAKE_HOST_UNIX)
  execute_process(COMMAND hostname OUTPUT_VARIABLE TMP_HOSTNAME OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(DEFINED BUILD_HOSTNAME AND NOT "${BUILD_HOSTNAME}" STREQUAL "${TMP_HOSTNAME}")
    message(WARNING "
            The CMake cache, probably generated on a different host (${BUILD_HOSTNAME}),
            is being reused! This could lead to inconsitencies; therefore, it is
            recommended to regenerate the cache!")
  endif()
  set(BUILD_HOSTNAME "${TMP_HOSTNAME}" CACHE INTERNAL "Hostname of the machine where the cache was generated." FORCE)
endif()

# Get the current working branch
execute_process(COMMAND git rev-parse --abbrev-ref HEAD WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} OUTPUT_VARIABLE BRANCH OUTPUT_STRIP_TRAILING_WHITESPACE)

# Get the latest abbreviated commit hash of the working branch
execute_process(COMMAND git log -1 --format=%h WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} OUTPUT_VARIABLE COMMIT_HASH OUTPUT_STRIP_TRAILING_WHITESPACE)

SET(SHARED_LIBS_DEFAULT ON)
IF(CMAKE_GENERATOR MATCHES "Visual Studio")
    SET(SHARED_LIBS_DEFAULT OFF)
ENDIF()
option(BUILD_SHARED_LIBS "Enable shared libraries (can be problematic with MPI, Windows)." ${SHARED_LIBS_DEFAULT})

if(UNIX AND NOT APPLE)
    option(PREFER_STATIC_LIBS "When finding libraries prefer \".a\" static archives (NOTE: this is enabled only for UNIX (excluding APPLE) platforms but it might not always work!" OFF)
    mark_as_advanced(PREFER_STATIC_LIBS)
    if(PREFER_STATIC_LIBS)
      SET(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
      if(BUILD_SHARED_LIBS)
        message(WARNING "Static libraries requested, the libraries will also be build static (BUILD_SHARED_LIBS=OFF).")
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "Enable shared libraries (can be problematic with MPI, Windows)." FORCE)
      endif()
    endif()
endif()

if(WIN32)
  if (CMAKE_CL_64)
    set(ARCH_IS_64 "TRUE")
  else()
    set(ARCH_IS_64 "FALSE")
  endif()
else()
  get_property(ARCH_IS_64 GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS SET)
endif()

option(USE_MPI "Build a parallel version using message passing interface." ON)
option(USE_OPENMP "Build a parallel version using threads (shared memory). Cannot combine with USE_MPI yet." OFF)

option(DOUBLE_PRECISION "Turn on to use double precision (real*8) floating numbers" ON)
if(DOUBLE_PRECISION)
  add_definitions(-D__DOUBLE_PRECISION__)
endif()

option(USE_OWN "Use own implementation of certain math functions instead of system-provided ones" ON)
if(USE_OWN)
  add_definitions(-D__USEOWN__)
endif()

option(JOB_FARMING "C++ main program manages job distribution and calls topmon Fortran code" OFF)
if(JOB_FARMING)
  enable_language(C CXX)
  include(FortranCInterface)
  FortranCInterface_HEADER(topmon_fortran_interface.h MACRO_NAMESPACE "topmon_" SYMBOLS sim_system:setup_mpi topmon_main:monola)
  FortranCInterface_VERIFY(CXX)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $ENV{CXX_FLAGS}" CACHE STRING "Flags used by the compiler during all build types." FORCE)

  # CFLAGS depend on the compiler
  get_filename_component(CXX_COMPILER_NAME ${CMAKE_CXX_COMPILER} NAME)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # g++
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mtune=native -march=native -ffast-math -fassociative-math -freciprocal-math -fno-signed-zeros -fno-trapping-math -funroll-loops")
    set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -fbounds-check -Wall -Wextra -D__DEBUG__")
    if(USE_OPENMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # icpc
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -xHost -ipo -prec-div")
    set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -traceback -check all -warn unused -warn declarations -warn truncated_source -warn uncalled -warn ignore_loc -stand f08 -D__DEBUG__")
    if(USE_OPENMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
    # pgcpp/pgc++
    set(CMAKE_CXX_FLAGS_RELEASE "-fast -Mipa=fast,inline -O4 -Msmartalloc -Mfprelaxed -Mstack_arrays")
    set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -traceback -Mbounds -Mchkfpstk -Mchkptr -Mchkstk -D__DEBUG__")
    if(USE_OPENMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mp=allcores")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "XL")
    # xlc++ from IBM
    set(CMAKE_CXX_FLAGS_RELEASE "-O5 -qprefetch=aggressive -qunroll=yes")
    set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g9 -qstrict -qcheck -qextchk -qflttrap -D__DEBUG__")
    if(USE_OPENMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qsmp=omp")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "SunPro")
    # (sun)CC from Oracle
    set(CMAKE_CXX_FLAGS_RELEASE "-fast -xipo=2 -xipo_archive=readonly -xprefetch=yes -xprefetch_level=3 -xprefetch_auto_type=indirect_array_access -xvector=simd -m64 -u")
    set(CMAKE_CXX_FLAGS_DEBUG   "-g -traceback -u -C -xcheck=%all -xcommonchk=yes -w4 -Xlist -D__DEBUG__")
    if(USE_OPENMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xopenmp")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MIPSpro")
    # CC from SGI for Origin; Open64 (initially known as Pro64)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -IPA -mips4 -r12000 -OPT:IEEE_arithmetic=3") #Ofast[=ipxx]
    set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -DEBUG:verbose_runtime:subscript_check=ON:div_check=3:fullwarn")
    if(USE_OPENMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mp")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "PathScale")
    # pathCC (derived from MIPSPro)
    set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=auto -inline -funsafe-math-optimizations -OPT:IEEE_arithmetic=3")
    set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g2 -ffortran-bounds-check -trapuv -Wall -fullwarn -D__DEBUG__")
    if(USE_OPENMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mp")
    endif()
  else()
    message("CMAKE_CXX_COMPILER full path: " ${CMAKE_CXX_COMPILER})
    message("C++ compiler: " ${CXX_COMPILER_NAME})
    message("No optimized C++ compiler flags are known, we just try -O2...")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2")
    set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g")
  endif()

  if(NOT "${CXX_COMPILER_NAME}" STREQUAL "${SAVE_CXX_COMPILER_NAME}")
    set(SAVE_CXX_COMPILER_NAME "${CXX_COMPILER_NAME}" CACHE INTERNAL "Compiler name when the cache was generated." FORCE)

    set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} CACHE STRING "Flags used by the compiler during RELEASE builds." FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} CACHE STRING "Flags used by the compiler during DEBUG builds." FORCE)
  else()
    # use cache values
    unset(CMAKE_CXX_FLAGS_RELEASE)
    unset(CMAKE_CXX_FLAGS_DEBUG)
  endif()

  set(C_INCLUDE_DIRS)
  set(CXX_LINK_FLAGS)
endif()

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} $ENV{FC_FLAGS}" CACHE STRING "Flags for Fortran compiler." FORCE)

# FFLAGS depend on the compiler
get_filename_component(Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)

if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  # gfortran (forked from g95 in January 2003; called g77 before gcc v4.0)
  set(CMAKE_Fortran_FLAGS_RELEASE "-ffree-form -ffree-line-length-none -cpp -O3 -mtune=native -march=native -ffast-math -fassociative-math -freciprocal-math -fno-signed-zeros -fno-trapping-math -funroll-loops")
  set(CMAKE_Fortran_FLAGS_DEBUG   "-ffree-form -ffree-line-length-none -cpp -O0 -g -fbacktrace -fbounds-check -Wall -Wextra -Wcharacter-truncation -Wunderflow -D__DEBUG__")
  if(USE_OPENMP)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fopenmp")
  endif()
  if(DOUBLE_PRECISION)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fdefault-real-8 -fdefault-double-8")
  endif()
  add_definitions(-D__TRADITIONAL_CPP__)
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "G95")
  # g95, also based on GCC (GNU Compiler Collection)
  set(CMAKE_Fortran_FLAGS_RELEASE "-ffree-form -ffree-line-length-huge -cpp -O3 -mtune=native -ffast-math -fno-trapping-math -funroll-loops")
  set(CMAKE_Fortran_FLAGS_DEBUG   "-ffree-form -ffree-line-length-huge -cpp -O0 -g -ftrace=full -fbounds-check -Wall -Wextra -D__DEBUG__")
  if(USE_OPENMP)
    message(WARNING "g95 does not support OpenMP.")
    set(USE_OPENMP OFF CACHE BOOL "Build a parallel version using threads (shared memory). Cannot combine with USE_MPI yet." FORCE)
  endif()
  if(DOUBLE_PRECISION)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -r8")
  endif()
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  # ifort
  set(CMAKE_Fortran_FLAGS_RELEASE "-free -fpp -O3 -xHost -ipo -prec-div")# -opt-streaming-stores always -no-unroll-aggressive -scalar-rep -Zp16 -nolib-inline") #-tpp2 for SGI Altix ia64 architecture
  set(CMAKE_Fortran_FLAGS_DEBUG   "-free -fpp -O0 -g -traceback -check all -warn unused -warn declarations -warn truncated_source -warn uncalled -warn ignore_loc -stand f08 -D__DEBUG__")
  if(USE_OPENMP)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -openmp")
  endif()
  if(DOUBLE_PRECISION)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -real-size 64")
  endif()
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "PGI")
  # pgf77, pgfortran/pgf90/pgf95
  set(CMAKE_Fortran_FLAGS_RELEASE "-Mfree -Mpreprocess -fast -Mipa=fast,inline -O4 -Msmartalloc -Mfprelaxed -Mstack_arrays")
  set(CMAKE_Fortran_FLAGS_DEBUG   "-Mfree -Mpreprocess -O0 -g -traceback -Mbounds -Mchkfpstk -Mchkptr -Mchkstk -D__DEBUG__")
  if(USE_OPENMP)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -mp=allcores")
  endif()
  if(DOUBLE_PRECISION)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -r8")
  endif()
  add_definitions(-D__TRADITIONAL_CPP__)
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "XL")
  # xlf/xlf90/xlf95/xlf2003/xlf2008 from IBM
  set(CMAKE_Fortran_FLAGS_RELEASE "-g -qarch=qp -qtune=qp -qfree=f90 -WF,-O3 -qnoipa -qsmp=omp:noauto -qsimd=auto -qhot=level=2 -qprefetch -qunroll=yes -qalias=nostd")
  set(CMAKE_Fortran_FLAGS_DEBUG   "-qfree=f90 -WF,-qfpp -O0 -g9 -qalias=nostd -qstrict -qcheck -qextchk -qflttrap -WF,-D__DEBUG__")
  if(USE_OPENMP)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -qsmp=omp")
  endif()
  if(DOUBLE_PRECISION)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -qautodbl=dbl4")
  endif()
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "SunPro")
  # (sun)f77, (sun)f90/f95 from Oracle
  set(CMAKE_Fortran_FLAGS_RELEASE "-free -fpp -fast -xipo=2 -xipo_archive=readonly -xprefetch=yes -xprefetch_level=3 -xprefetch_auto_type=indirect_array_access -xvector=simd -m64 -u")
  set(CMAKE_Fortran_FLAGS_DEBUG   "-free -fpp -g -traceback -u -C -xcheck=%all -xcommonchk=yes -w4 -Xlist -D__DEBUG__")
  if(USE_OPENMP)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -xopenmp")
  endif()
  if(DOUBLE_PRECISION)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -r8const -xtypemap=real:64,double:64,integer:32")
  endif()
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "MIPSpro")
  # f77, f90 from SGI for Origin; Open64 (initially known as Pro64)
  set(CMAKE_Fortran_FLAGS_RELEASE "-freeform -cpp -O3 -IPA -mips4 -r12000 -OPT:IEEE_arithmetic=3") #Ofast[=ipxx]
  set(CMAKE_Fortran_FLAGS_DEBUG   "-freeform -cpp -O0 -g -DEBUG:verbose_runtime:subscript_check=ON:div_check=3:fullwarn")
  if(USE_OPENMP)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -mp")
  endif()
  if(DOUBLE_PRECISION)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -r8")
  endif()
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "PathScale")
  # pathf90/pathf95 (derived from MIPSPro)
  set(CMAKE_Fortran_FLAGS_RELEASE "-freeform -cpp -Ofast -march=auto -inline -funsafe-math-optimizations -OPT:IEEE_arithmetic=3")
  set(CMAKE_Fortran_FLAGS_DEBUG   "-freeform -cpp -O0 -g2 -ffortran-bounds-check -trapuv -Wall -fullwarn -D__DEBUG__")
  if(USE_OPENMP)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -mp")
  endif()
  if(DOUBLE_PRECISION)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -r8")
  endif()
  add_definitions(-D__TRADITIONAL_CPP__)
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "NAG")
  # f95, nagfor
  set(CMAKE_Fortran_FLAGS_RELEASE "-free -fpp -kind=byte -O4 -Orounding -Ounsafe")
  set(CMAKE_Fortran_FLAGS_DEBUG   "-free -fpp -kind=byte -O0 -g -gline -C=all -C=undefined -color")
  if(USE_OPENMP)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -openmp")
  endif()
  if(DOUBLE_PRECISION)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -double")
  endif()
else()
  message("CMAKE_Fortran_COMPILER full path: " ${CMAKE_Fortran_COMPILER})
  message("Fortran compiler: " ${Fortran_COMPILER_NAME})
  message("No optimized Fortran compiler flags are known, we just try -O2...")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O2")
  set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -g")
endif()

if(NOT "${Fortran_COMPILER_NAME}" STREQUAL "${SAVE_Fortran_COMPILER_NAME}")
  set(SAVE_Fortran_COMPILER_NAME "${Fortran_COMPILER_NAME}" CACHE INTERNAL "Compiler name when the cache was generated." FORCE)

  set(CMAKE_Fortran_FLAGS_RELEASE ${CMAKE_Fortran_FLAGS_RELEASE} CACHE STRING "Flags used by the compiler during RELEASE builds." FORCE)
  set(CMAKE_Fortran_FLAGS_DEBUG ${CMAKE_Fortran_FLAGS_DEBUG} CACHE STRING "Flags used by the compiler during DEBUG builds." FORCE)
else()
  # use cache values
  unset(CMAKE_Fortran_FLAGS_RELEASE)
  unset(CMAKE_Fortran_FLAGS_DEBUG)
endif()

# Set Profiler
set(PROFILER CACHE STRING "Enable a compiler wrapper. Options: mpi, tau, scalasca.")
if(PROFILER STREQUAL "mpi")
  set(COMPILER_WRAPPER "mpif90")
elseif(PROFILER STREQUAL "tau")
  set(COMPILER_WRAPPER "tau_f90.sh")
elseif(PROFILER STREQUAL "scalasca")
  set(COMPILER_WRAPPER "skin_${Fortran_COMPILER_NAME}.sh")
  execute_process(COMMAND echo -e "#!/bin/sh\nskin ${Fortran_COMPILER_NAME} \"$@\"" OUTPUT_FILE ${COMPILER_WRAPPER})
  execute_process(COMMAND chmod +x "${COMPILER_WRAPPER}")
  message("
            NOTE: Please add the absolute path of the current
            directory, where ${COMPILER_WRAPPER} was just generated,
            to the environment variable PATH before running make.
")
endif()

# Set the compiler name to the wrapper, if necessary
if(DEFINED COMPILER_WRAPPER)
  CMAKE_FORCE_Fortran_COMPILER(${COMPILER_WRAPPER} ${CMAKE_Fortran_COMPILER_ID})
endif()

set(EXTRA_LIBRARIES)

if(APPLE)
   find_library(ACCELERATE_FRAMEWORK Accelerate)
   list(APPEND EXTRA_LIBRARIES ${ACCELERATE_FRAMEWORK})
endif()

set(Fortran_INCLUDE_DIRS)
set(Fortran_LINK_FLAGS)

if(USE_MPI)
  # fix the problem with Intel/PGI compilers not being able to find 64-bit libraries
  if(${ARCH_IS_64} STREQUAL "TRUE" AND (CMAKE_Fortran_COMPILER_ID STREQUAL "Intel" OR CMAKE_Fortran_COMPILER_ID STREQUAL "PGI"))
    list(APPEND CMAKE_LIBRARY_PATH "/usr/lib64")
  endif()

  find_package(MPI) # REQUIRED)
  if(MPI_FOUND OR MPI_Fortran_FOUND)
    add_definitions(-D__MPI__)

    if(JOB_FARMING)
      list(APPEND EXTRA_LIBRARIES ${MPI_LIBRARIES})
      list(APPEND C_INCLUDE_DIRS ${MPI_INCLUDE_PATH})
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
      set(CXX_LINK_FLAGS "${CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    endif()

    list(APPEND EXTRA_LIBRARIES ${MPI_Fortran_LIBRARIES})
    list(APPEND Fortran_INCLUDE_DIRS ${MPI_Fortran_INCLUDE_PATH})
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${MPI_Fortran_COMPILE_FLAGS}")
    set(Fortran_LINK_FLAGS "${Fortran_LINK_FLAGS} ${MPI_Fortran_LINK_FLAGS}")
  else()
    message(FATAL_ERROR "MPI support requested, but no MPI compiler found.")
  endif()
endif()

if(USE_OPENMP)
  add_definitions(-D__OPENMP__)
endif()

file(GLOB_RECURSE Fortran_HEADERS "*.inc")
foreach(_headerFile ${Fortran_HEADERS})
  get_filename_component(_dir ${_headerFile} PATH)
  list(APPEND Fortran_INCLUDE_DIRS ${_dir})
endforeach()

if(Fortran_INCLUDE_DIRS)
  list(REMOVE_DUPLICATES Fortran_INCLUDE_DIRS)
endif()
set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "${Fortran_INCLUDE_DIRS};${PROJECT_BINARY_DIR}")

add_subdirectory(src)
# add_subdirectory(contrib/baip/Fortran)

get_directory_property(PREPROCESSOR_DEFINITIONS COMPILE_DEFINITIONS)
get_target_property(SAVE_Fortran_FLAGS topmon INCLUDE_DIRECTORIES)
configure_file(${PROJECT_SOURCE_DIR}/src/defines.h.in ${PROJECT_BINARY_DIR}/defines.h @ONLY)
