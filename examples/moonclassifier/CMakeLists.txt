cmake_minimum_required(VERSION 3.10)
project(moonclsr DESCRIPTION "Moon classifier example")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if(UNIX)
    # Qt5 installed as a package
else()
    set(CMAKE_PREFIX_PATH "$ENV{QT_PATH}//")
endif()
find_package(Qt5 COMPONENTS Widgets REQUIRED)


add_executable(moonclsr)

file(
    GLOB_RECURSE _moonclsr_src
    LIST_DIRECTORIES false
    "../common/*.cpp"
    "../common/*.h"
    "src/*.cpp"
    "src/*.h"
    ${CMAKE_CURRENT_SOURCE_DIR}/moonclsr.qrc
    ${CMAKE_CURRENT_SOURCE_DIR}/exportmap.def
)

target_sources(moonclsr PRIVATE
	${_moonclsr_src}
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/../" FILES ${_moonclsr_src})

target_include_directories(moonclsr PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_PREFIX_PATH}include
    ${CMAKE_PREFIX_PATH}include/QtGui
)

target_link_libraries(moonclsr Qt5::Core Qt5::Gui Qt5::Widgets)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT moonclsr)

if(UNIX)
else()
#  add_custom_command(TARGET libguim1 
#          POST_BUILD
#          COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/post_build_event.bat "\"${SIM_BINARY_DIRECTORY}\""
#          )
endif()


