#src/CMakeLists.txt

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

#set up shared library by hand:
ADD_LIBRARY(MSP_toroidal_membrane SHARED MSP_toroidal_membrane.cc)
DEAL_II_SETUP_TARGET(MSP_toroidal_membrane)

#set up executable with autopilot macro:
SET(TARGET "main")
SET(TARGET_SRC main.cc)
DEAL_II_INVOKE_AUTOPILOT()
TARGET_LINK_LIBRARIES(${TARGET} MSP_toroidal_membrane)
