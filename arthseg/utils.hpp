#pragma once
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <numpy/arrayobject.h>

inline constexpr int drow[] = { 1, 0, -1, 0, 1, 1, -1, -1 };
inline constexpr int dcol[] = { 0, 1, 0, -1, 1, -1, 1, -1 };

inline bool is_outside(const PyArrayObject *image, npy_intp row, npy_intp col)
{
    return row < 0 || col < 0 || row >= PyArray_DIM(image, 0) || col >= PyArray_DIM(image, 1);
}

inline unsigned long PyArray_At(PyArrayObject *image, npy_intp row, npy_intp col)
{
    return PyLong_AsUnsignedLong(PyArray_GETITEM(image, (char *) PyArray_GETPTR2(image, row, col)));
}

inline void PyArray_Set(PyArrayObject *image, npy_intp row, npy_intp col, unsigned long value)
{
    PyArray_SETITEM(image, (char *) PyArray_GETPTR2(image, row, col), Py_BuildValue("B", value));
}
