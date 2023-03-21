#pragma once
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <numpy/ndarraytypes.h>

#include "types.hpp"

struct Node : Point
{
    size_t cost;
    Node(size_t row, size_t col, size_t cost) : Point(row, col), cost(cost) {}
    bool operator<(const Node &other) const { return cost > other.cost; }
};

std::vector<Node> shortest_path(PyArrayObject *image, const std::vector<Point> &points, const std::vector<Point> &start);
