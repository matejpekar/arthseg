#pragma once
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <numpy/ndarraytypes.h>
#include <vector>

#include "types.hpp"

struct Component
{
    size_t label;
    std::vector<Point> nodes;

    Component() = default;
    Component(size_t label) : label(label) {}
    Component(size_t label, const std::vector<Point> &points) : label(label), nodes(points) {}
    bool empty() const { return nodes.empty(); }
    size_t size() const { return nodes.size(); }
};

struct ComponentWithEdge : Component
{
    std::vector<Point> edge;
    using Component::Component;
};

std::vector<Component> connected_components(PyArrayObject *image, Connectivity connectivity = CONNECTIVITY_8);
std::vector<ComponentWithEdge> connected_components_with_edge(PyArrayObject *image, Connectivity connectivity = CONNECTIVITY_8, Connectivity edge_connectivity = CONNECTIVITY_4);
