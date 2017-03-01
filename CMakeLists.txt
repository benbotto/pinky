cmake_minimum_required(VERSION 2.6)
project(pinky)

# All source files.
set(SRC
  "./src/pinky.cpp"
  "./src/neuron/Neuron.cpp"
  "./src/neuron/InputNeuron.cpp"
  "./src/neuron/BiasNeuron.cpp"
  "./src/neuron/OutputNeuron.cpp")

# Build the pinky executable.
add_executable(pinky ${SRC})

# Release build by default.
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -Wall -pedantic -std=c++11")

