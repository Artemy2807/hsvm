cmake_minimum_required(VERSION 3.6)
# Имя проекта
project(train-svm)
# Настройки CMake
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Флаги компиляции
if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=c++14 -Wall -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4 /WX-)
    if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.11)
        add_compile_options(/permissive-)
    endif()
endif()
# Если ОП Linux
if(UNIX AND NOT APPLE)
    # Ищем файлы бибоитеки OpenCV
    find_package(OpenCV)
    # Подключаем файлы бибоитеки OpenCV
    include_directories(${OpenCV_INCLUDES})
# Если ОП Windows
elseif(WIN32)
    # Устаналиваем пути до библиотеки OpenCV
    # ВАЖНО укажите путь до вашей папки с собранной библиотекой OpenCV ВАЖНО
    set(OpenCV_DIR "C:/Program Files/opencv/build")
    set(OpenCV_INCLUDES ${OpenCV_DIR}/include)
    # Ищем файлы бибоитеки OpenCV
    find_package(OpenCV REQUIRED)
    # Подключаем файлы бибоитеки OpenCV
    include_directories(${OpenCV_INCLUDES})
endif()
# Подключаем заголовочные файлы
include_directories(getopt)
# Подключаем файлы .c и .cpp
aux_source_directory(getopt SOURCES)
set(SOURCES ${SOURCES}
    main.cpp
)
add_executable(train-svm ${SOURCES})
# Подключаем файлы .lib в программу
target_link_libraries(train-svm ${OpenCV_LIBS})


