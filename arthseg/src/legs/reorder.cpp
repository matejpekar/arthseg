#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <algorithm>
#include <numpy/arrayobject.h>

#include "legs.hpp"
#include "moments.hpp"

using LegWithHeight = std::vector<std::pair<std::vector<Point>, size_t>>;
using LegPair = std::pair<std::vector<Point>, std::vector<Point>>;

static std::vector<LegPair> make_pairs(LegWithHeight &left, LegWithHeight &right, size_t size);
static bool is_pair(LegWithHeight &left_it, LegWithHeight &right_it, LegWithHeight::iterator left, LegWithHeight::iterator right);

void reored_legs(PyArrayObject *image, PyObject *body_labels, PyObject *pair_labels, PyObject *alternative_labels, const std::vector<std::vector<Point>> &legs, const std::vector<Point> &body)
{
    LegWithHeight left, right;

    const auto body_moments = Moments(body);
    const Point intersection_x(0, body_moments.radius / cos(body_moments.angle));

    for (const auto &leg : legs) {
        const auto start = find_leg_start(image, body_labels, alternative_labels, leg);
        if (start.empty()) {
            continue;
        }

        const auto leg_start = Moments::get_centroid(start);
        const auto centroid = Moments::get_centroid(leg);

        if (body_moments.half_axis(centroid) < 0) {
            left.push_back({ std::move(leg), Point::distance(intersection_x, body_moments.orthogonal_projection(leg_start)) });
        } else {
            right.push_back({ std::move(leg), Point::distance(intersection_x, body_moments.orthogonal_projection(leg_start)) });
        }
    }

    std::sort(left.begin(), left.end(), [](const auto &a, const auto &b) { return a.second < b.second; });
    std::sort(right.begin(), right.end(), [](const auto &a, const auto &b) { return a.second < b.second; });

    size_t index = 0;
    for (const auto &[left, right] : make_pairs(left, right, PyList_Size(pair_labels))) {
        auto *left_label = PyTuple_GetItem(PyList_GetItem(pair_labels, index), 0);
        auto *right_label = PyTuple_GetItem(PyList_GetItem(pair_labels, index), 1);

        for (const auto &point : left) {
            PyArray_SETITEM(image, (char *) PyArray_GETPTR2(image, point.row, point.col), left_label);
        }
        for (const auto &point : right) {
            PyArray_SETITEM(image, (char *) PyArray_GETPTR2(image, point.row, point.col), right_label);
        }
        index++;
    }
}

static std::vector<LegPair> make_pairs(LegWithHeight &left, LegWithHeight &right, size_t size)
{
    std::vector<LegPair> pairs;

    auto l = left.begin();
    auto r = right.begin();

    bool left_full = left.size() >= size;
    bool right_full = right.size() >= size;

    while ((l != left.end() || r != right.end()) && pairs.size() < size) {
        left_full |= pairs.size() + left.size() - (l - left.begin()) == size;
        right_full |= pairs.size() + right.size() - (r - right.begin()) == size;
        if (l == left.end()) {
            pairs.emplace_back(std::vector<Point>(), std::move(r->first));
            r++;
        } else if (r == right.end()) {
            pairs.emplace_back(std::move(l->first), std::vector<Point>());
            l++;
        } else {
            if ((left_full && right_full) || is_pair(left, right, l, r)) {
                pairs.emplace_back(std::move(l->first), std::move(r->first));
                l++;
                r++;
            } else if (left_full || (l->second < r->second && !right_full)) {
                pairs.emplace_back(std::move(l->first), std::vector<Point>());
                l++;
            } else {
                pairs.emplace_back(std::vector<Point>(), std::move(r->first));
                r++;
            }
        }
    }

    return pairs;
}

static bool is_pair(LegWithHeight &left, LegWithHeight &right, LegWithHeight::iterator left_it, LegWithHeight::iterator right_it)
{
    const int current_distance = abs((int) left_it->second - (int) right_it->second);

    if ((left_it + 1) != left.end() && abs((int) (left_it + 1)->second - (int) right_it->second) < current_distance) {
        return false;
    }

    if ((right_it + 1) != right.end() && abs((int) left_it->second - (int) (right_it + 1)->second) < current_distance) {
        return false;
    }

    return true;
}
