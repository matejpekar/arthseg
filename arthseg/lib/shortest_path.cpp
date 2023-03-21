#include <limits>
#include <queue>

#include "shortest_path.hpp"
#include "utils.hpp"

// approximate distance between two points
static constexpr int line = 10, diagonal = 14;

std::vector<Node> shortest_path(PyArrayObject *image, const std::vector<Point> &points, const std::vector<Point> &start)
{
    const size_t rows = PyArray_DIM(image, 0);
    const size_t cols = PyArray_DIM(image, 1);
    Matrix<bool> marker(rows, cols);
    Matrix<size_t> distance(rows, cols, std::numeric_limits<size_t>::max());

    for (const auto &point : points) {
        marker.at(point) = true;
    }

    std::vector<Node> nodes;
    std::priority_queue<Node> queue;
    for (const auto &point : start) {
        queue.emplace(point.row, point.col, 0);
        distance.at(point) = 0;
    }

    while (!queue.empty()) {
        const auto node = queue.top();
        queue.pop();

        if (!marker.at(node)) {
            continue;
        }

        marker.at(node) = false;
        nodes.push_back(node);
        for (size_t i = 0; i < CONNECTIVITY_8; i++) {
            const size_t row = node.row + drow[i];
            const size_t col = node.col + dcol[i];
            const auto cost = node.cost + (i < 4 ? line : diagonal);

            if (row < rows && col < cols && marker.at(row, col) && cost < distance.at(row, col)) {
                distance.at(row, col) = cost;
                queue.emplace(row, col, cost);
            }
        }
    }

    return nodes;
}
