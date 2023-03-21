#pragma once
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <numpy/ndarraytypes.h>

PyArrayObject *fill_holes(PyArrayObject *image, unsigned long fill_value, float hole_area);
