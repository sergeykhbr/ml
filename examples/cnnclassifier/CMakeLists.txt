cmake_minimum_required(VERSION 3.10)
project(cnnclsr DESCRIPTION "Convolutional Network (CNN) classifier example")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if(UNIX)
    # Qt5 installed as a package
else()
    set(CMAKE_PREFIX_PATH "$ENV{QT_PATH}//")
endif()
find_package(Qt5 COMPONENTS Widgets REQUIRED)


add_executable(cnnclsr)

file(
    GLOB_RECURSE _cnnclsr_src
    LIST_DIRECTORIES false
    "../common/*.cpp"
    "../common/*.h"
    "src/*.cpp"
    "src/*.h"
    ${CMAKE_CURRENT_SOURCE_DIR}/cnnclsr.qrc
    ${CMAKE_CURRENT_SOURCE_DIR}/exportmap.def
)

target_sources(cnnclsr PRIVATE
	${_cnnclsr_src}
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/../" FILES ${_cnnclsr_src})

target_include_directories(cnnclsr PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_PREFIX_PATH}include
    ${CMAKE_PREFIX_PATH}include/QtGui
)

target_link_libraries(cnnclsr Qt5::Core Qt5::Gui Qt5::Widgets)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT cnnclsr)

if(UNIX)
else()
    # Find the windeployqt executable relative to the Qt5 Widgets library location
    get_target_property(_qt5_widgets_loc Qt5::Widgets LOCATION)
    get_filename_component(_qt5_bin_dir "${_qt5_widgets_loc}" DIRECTORY)
    
    # Run windeployqt automatically after building the executable
    add_custom_command(TARGET cnnclsr POST_BUILD
        COMMAND "${_qt5_bin_dir}/windeployqt.exe" "$<TARGET_FILE:cnnclsr>"
        COMMENT "Deploying Qt DLLs for cnnclsr..."
    )
endif()


