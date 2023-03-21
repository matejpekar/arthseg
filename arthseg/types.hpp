#pragma once

#include <tuple>
#include <vector>

using Connectivity = enum Connectivity {
    CONNECTIVITY_4 = 4,
    CONNECTIVITY_8 = 8
};

struct Point
{
    size_t row, col;
    Point(size_t row, size_t col) : row(row), col(col) {}
    static float distance(const Point &a, const Point &b)
    {
        const auto dx = abs((int) a.col - (int) b.col);
        const auto dy = abs((int) a.row - (int) b.row);
        return dx > dy ? (0.41 * dy + 0.941246 * dx) : (0.41 * dx + 0.941246 * dy);
    }
};

template <typename T>
class Matrix
{
  private:
    std::vector<T> data;

  public:
    const size_t rows, cols;

    Matrix(const size_t rows, const size_t cols) : data(rows * cols), rows(rows), cols(cols) {}
    Matrix(const size_t rows, const size_t cols, T initial_value) : data(rows * cols, initial_value), rows(rows), cols(cols) {}
    typename std::vector<T>::reference at(const size_t row, const size_t col) { return data[row * cols + col]; }
    typename std::vector<T>::reference at(const Point &point) { return data[point.row * cols + point.col]; }
    typename std::vector<T>::const_reference at(const size_t row, const size_t col) const { return data[row * cols + col]; }
    typename std::vector<T>::const_reference at(const Point &point) const { return data[point.row * cols + point.col]; }
};
