#pragma once
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <numpy/ndarraytypes.h>

PyArrayObject *remove_dirt(PyArrayObject *image, bool keep, size_t max_distance, float min_area);
