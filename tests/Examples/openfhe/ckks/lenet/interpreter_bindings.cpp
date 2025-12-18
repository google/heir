// A shim around the HEIR OpenFHE interpreter to provide pybind11 bindings.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tests/Examples/openfhe/ckks/lenet/interpreter_shim.h"

namespace py = pybind11;

PYBIND11_MODULE(lenet_interpreter, m) {
  m.def("lenet_interpreter", &lenet_interpreter,
        py::call_guard<py::gil_scoped_release>());
}
