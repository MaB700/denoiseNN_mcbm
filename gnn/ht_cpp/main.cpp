#include <pybind11/pybind11.h>

namespace py = pybind11;

float some_fn(float arg1, float arg2) {
  return arg1 + arg2;
}

PYBIND11_MODULE(ht_cpp, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring

  m.def("some_fn", &some_fn, "A function which adds two numbers");
}