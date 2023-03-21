#include <algorithm>
#include <limits>

#include "connected_components.hpp"
#include "remove_dirt.hpp"
#include "utils.hpp"

static size_t min_distance(const std::vector<Point> &left, const std::vector<Point> &right);

PyArrayObject *remove_dirt(PyArrayObject *image, bool keep, size_t max_distance, float min_area)
{
    import_array();
    PyArrayObject *mask = (PyArrayObject *) PyArray_EMPTY(PyArray_NDIM(image), PyArray_DIMS(image), NPY_UINT8, 0);
    PyArrayObject *output = (PyArrayObject *) PyArray_Empty(PyArray_NDIM(image), PyArray_DIMS(image), PyArray_DTYPE(image), 0);
    if (mask == NULL || output == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }

    for (npy_intp row = 0; row < PyArray_DIM(image, 0); row++) {
        for (npy_intp col = 0; col < PyArray_DIM(image, 1); col++) {
            const auto value = PyLong_AsUnsignedLong(PyArray_GETITEM(image, (char *) PyArray_GETPTR2(image, row, col)));
            PyArray_SETITEM(mask, (char *) PyArray_GETPTR2(mask, row, col), Py_BuildValue("B", value != 0));
        }
    }

    if (PyArray_CopyInto(output, image)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to copy image");
        return NULL;
    }

    const auto components = connected_components_with_edge(mask, CONNECTIVITY_4);
    if (components.size() < 2) {
        return output;
    }

    const auto largest = std::max_element(components.begin(), components.end(), [](const auto &left, const auto &right) {
        return left.size() < right.size();
    });

    for (auto it = components.begin(); it != components.end(); it++) {
        if (it == largest) {
            continue;
        }
        if (!keep || it->size() < min_area * largest->size() || min_distance(largest->edge, it->edge) > max_distance) {
            for (const auto &node : it->nodes) {
                PyArray_Set(output, node.row, node.col, 0);
            }
        }
    }

    return output;
}

static size_t min_distance(const std::vector<Point> &left, const std::vector<Point> &right)
{
    size_t distance = std::numeric_limits<size_t>::max();

    for (const auto &point1 : left) {
        for (const auto &point2 : right) {
            distance = std::min(distance, (size_t) Point::distance(point1, point2));
        }
    }
    return distance;
}
