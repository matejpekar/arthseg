#include <algorithm>
#include <map>

#include "connected_components.hpp"
#include "refine.hpp"
#include "utils.hpp"

static const void attach(PyArrayObject *image,
        const ComponentWithEdge &component,
        Matrix<ComponentWithEdge *> &marker);

PyArrayObject *refine_regions(PyArrayObject *image, PyObject *body_labels, float min_area)
{
    import_array();
    PyArrayObject *output = (PyArrayObject *) PyArray_Empty(PyArray_NDIM(image), PyArray_DIMS(image), PyArray_DTYPE(image), 0);
    if (output == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }
    if (PyArray_CopyInto(output, image)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to copy image");
        return NULL;
    }

    auto components = connected_components_with_edge(image, CONNECTIVITY_4);

    Matrix<ComponentWithEdge *> marker(PyArray_DIM(image, 0), PyArray_DIM(image, 1));
    std::map<size_t, const ComponentWithEdge *> max_components;
    size_t body_area = 0;
    for (auto &component : components) {
        max_components.try_emplace(component.label, &component);
        if (max_components[component.label]->size() < component.size()) {
            max_components[component.label] = &component;
        }
        if (PySet_Contains(body_labels, PyLong_FromLong(component.label))) {
            body_area += component.size();
        }

        for (const auto &pixel : component.edge) {
            marker.at(pixel) = &component;
        }
    }

    for (const auto &component : components) {
        if (!component.empty() &&
                &component != max_components[component.label] &&
                (PySet_Contains(body_labels, PyLong_FromLong(component.label)) || component.size() < body_area * min_area)) {
            attach(output, component, marker);
        }
    }

    return output;
}

static const void attach(PyArrayObject *image,
        const ComponentWithEdge &component,
        Matrix<ComponentWithEdge *> &marker)
{
    std::map<ComponentWithEdge *, size_t> neighbours;

    for (const auto &edge : component.edge) {
        for (size_t i = 0; i < CONNECTIVITY_4; i++) {
            const auto row = edge.row + drow[i];
            const auto col = edge.col + dcol[i];

            if (!is_outside(image, row, col) &&
                    PyArray_At(image, row, col) != component.label &&
                    marker.at(row, col) != nullptr) {
                neighbours.try_emplace(marker.at(row, col), 0);
                neighbours[marker.at(row, col)]++;
            }
        }
    }

    if (neighbours.empty()) { return; }

    const auto [max_neighbour, _] = *std::max_element(neighbours.begin(),
            neighbours.end(),
            [](const auto &l, const auto &r) { return l.second < r.second; });

    for (const auto &node : component.nodes) {
        PyArray_Set(image, node.row, node.col, max_neighbour->label);
    }

    for (const auto &edge : component.edge) {
        marker.at(edge) = max_neighbour;
    }

    max_neighbour->nodes.insert(max_neighbour->nodes.end(), component.nodes.begin(), component.nodes.end());
    max_neighbour->edge.insert(max_neighbour->edge.end(), component.edge.begin(), component.edge.end());
}
