#include "legs.hpp"
#include "shortest_path.hpp"
#include "utils.hpp"

struct FloodComponent
{
    std::vector<Point> points;
    size_t max_distance;
    size_t min_distance;
    FloodComponent(const Point &point, size_t distance) : points({ point }), max_distance(distance), min_distance(distance) {}
    size_t length() const { return max_distance - min_distance; }
    void add(const Node &node)
    {
        points.push_back(node);
        max_distance = std::max(max_distance, node.cost);
        min_distance = std::min(min_distance, node.cost);
    }
    void add(const FloodComponent &component)
    {
        for (const Point &point : component.points) {
            points.push_back(point);
        }
        max_distance = std::max(max_distance, component.max_distance);
        min_distance = std::min(min_distance, component.min_distance);
    }
};

std::vector<std::vector<Point>> split_leg(PyArrayObject *image, PyObject *body_labels, PyObject *alternative_labels, const Component &component)
{
    const std::vector<Point> start = find_leg_start(image, body_labels, alternative_labels, component.nodes);

    // leg not connected to body
    if (start.empty()) {
        return { component.nodes };
    }

    Matrix<size_t> group_labels(PyArray_DIM(image, 0), PyArray_DIM(image, 1));
    Matrix<bool> marker(PyArray_DIM(image, 0), PyArray_DIM(image, 1));
    std::vector<FloodComponent> groups;

    const auto sorted = shortest_path(image, component.nodes, start);
    for (const auto &point : sorted) {
        marker.at(point) = true;
    }

    // minimal length of leg
    const size_t min_length = (sorted.back().cost - sorted.front().cost) / 5;

    size_t label = 1;
    for (auto it = sorted.rbegin(); it != sorted.rend(); it++) {
        const auto &node = *it;
        if (group_labels.at(node) == 0) {
            group_labels.at(node) = label;
            groups.emplace_back(node, node.cost);
            label++;
        }

        auto &group = groups[group_labels.at(node) - 1];
        for (size_t i = 0; i < CONNECTIVITY_4; i++) {
            const auto row = node.row + drow[i];
            const auto col = node.col + dcol[i];
            if (is_outside(image, row, col) || !marker.at(row, col) || group_labels.at(row, col) == group_labels.at(node)) {
                continue;
            }

            auto &other_group = groups[group_labels.at(row, col) - 1];
            if (group_labels.at(row, col) == 0) {
                group_labels.at(row, col) = group_labels.at(node);
                group.add({ row, col, node.cost });
            } else if (group.length() < min_length || other_group.length() < min_length) {
                for (const Point &point : other_group.points) {
                    group_labels.at(point) = group_labels.at(node);
                    group.points.push_back(point);
                }
                group.max_distance = std::max(group.max_distance, other_group.max_distance);
                group.min_distance = std::min(group.min_distance, other_group.min_distance);
                other_group.points.clear();
            }
        }
    }

    std::vector<std::vector<Point>> result;
    for (const auto &group : groups) {
        if (!group.points.empty()) {
            result.emplace_back(std::move(group.points));
        }
    }
    return result;
}
