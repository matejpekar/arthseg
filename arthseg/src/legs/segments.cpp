#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <numpy/arrayobject.h>

#include "legs.hpp"
#include "shortest_path.hpp"

void leg_segments(PyArrayObject *image, PyObject *labels, PyObject *body_labels, PyObject *alternative_labels, const std::vector<Point> &points)
{
    const size_t size = PyList_Size(labels);
    if (size == 0) {
        return;
    }

    const auto start = find_leg_start(image, body_labels, alternative_labels, points);
    if (start.empty()) {
        return;
    }

    const auto sorted = shortest_path(image, points, start);
    const auto partion = (float) sorted.back().cost / size;
    size_t label = 0;
    for (const auto &node : sorted) {
        PyArray_SETITEM(image, (char *) PyArray_GETPTR2(image, node.row, node.col), PyList_GetItem(labels, label));
        if (node.cost >= partion * (label + 1) && label < size - 1) {
            label++;
        }
    }
}
