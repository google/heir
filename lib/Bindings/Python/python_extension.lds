/* Linker version script for the python extension modules (Linux): only export
   the module init functions. Keep in sync with python_extension.exp. */
{
  global:
    PyInit_*;
  local:
    *;
};
