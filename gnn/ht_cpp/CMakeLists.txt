cmake_minimum_required(VERSION 3.4...3.18)
project(ht)
add_subdirectory(pybind11)
pybind11_add_module(ht main.cpp)