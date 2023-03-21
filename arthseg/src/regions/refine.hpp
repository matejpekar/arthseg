#pragma once
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <numpy/ndarraytypes.h>

PyArrayObject *refine_regions(PyArrayObject *image, PyObject *body_labels, float min_area);
