#include <algorithm>
#include <iterator>

#include "legs.hpp"
#include "utils.hpp"

bool is_edge(PyArrayObject *image, PyObject *labels, const Point &point)
{
    for (size_t i = 0; i < CONNECTIVITY_4; i++) {
        const auto row = point.row + drow[i];
        const auto col = point.col + dcol[i];
        if (!is_outside(image, row, col) && PySet_Contains(labels, PyArray_GETITEM(image, (char *) PyArray_GETPTR2(image, row, col)))) {
            return true;
        }
    }
    return false;
}

std::vector<Point> find_leg_start(PyArrayObject *image, PyObject *body_labels, PyObject *alternative_labels, const std::vector<Point> &points)
{
    std::vector<Point> start;
    std::copy_if(points.begin(), points.end(), std::back_inserter(start), [&](const Point &point) { return is_edge(image, body_labels, point); });
    if (start.empty()) {
        std::copy_if(points.begin(), points.end(), std::back_inserter(start), [&](const Point &point) { return is_edge(image, alternative_labels, point); });
    }
    return start;
}
