#include "connected_components.hpp"
#include "utils.hpp"

static void dfs(PyArrayObject *image, Matrix<bool> &marker, ComponentWithEdge &component, Connectivity connectivity, Connectivity edge_connectivity);
static bool is_edge(PyArrayObject *image, const Point &point, Connectivity edge_connectivity);

std::vector<ComponentWithEdge> connected_components_with_edge(PyArrayObject *image, Connectivity connectivity, Connectivity edge_connectivity)
{
    const auto rows = PyArray_DIM(image, 0);
    const auto cols = PyArray_DIM(image, 1);
    Matrix<bool> marker(rows, cols);
    std::vector<ComponentWithEdge> components;

    for (npy_intp row = 0; row < rows; row++) {
        for (npy_intp col = 0; col < cols; col++) {
            if (PyArray_At(image, row, col) != 0 && !marker.at(row, col)) {
                const auto point = Point(row, col);
                components.emplace_back(PyArray_At(image, row, col), std::vector<Point>({ point }));
                components.back().edge.push_back(point);
                marker.at(point) = true;
                dfs(image, marker, components.back(), connectivity, edge_connectivity);
            }
        }
    }

    return components;
}

static void dfs(PyArrayObject *image, Matrix<bool> &marker, ComponentWithEdge &component, Connectivity connectivity, Connectivity edge_connectivity)
{
    for (size_t i = 0; i < component.size(); i++) {
        const auto point = component.nodes[i];
        for (size_t j = 0; j < connectivity; j++) {
            const auto row = point.row + drow[j];
            const auto col = point.col + dcol[j];
            if (!is_outside(image, row, col) && !marker.at(row, col) && PyArray_At(image, row, col) == component.label) {
                marker.at(row, col) = true;
                component.nodes.emplace_back(row, col);

                if (is_edge(image, component.nodes.back(), edge_connectivity)) {
                    component.edge.emplace_back(row, col);
                }
            }
        }
    }
}

static bool is_edge(PyArrayObject *image, const Point &point, Connectivity edge_connectivity)
{
    for (size_t i = 0; i < edge_connectivity; i++) {
        const auto row = point.row + drow[i];
        const auto col = point.col + dcol[i];
        if (!is_outside(image, row, col) && PyArray_At(image, row, col) != PyArray_At(image, point.row, point.col)) {
            return true;
        }
    }
    return false;
}
