#include <vector>

#include "types.hpp"

class Moments
{
  public:
    double angle;
    double radius;

    Moments(const std::vector<Point> &points)
    {
        const auto centroid = get_centroid(points);
        long int central_moment_11 = 0;
        long int central_moment_20 = 0;
        long int central_moment_02 = 0;

        for (const auto &[row, col] : points) {
            central_moment_11 += (row - centroid.row) * (col - centroid.col);
            central_moment_20 += (row - centroid.row) * (row - centroid.row);
            central_moment_02 += (col - centroid.col) * (col - centroid.col);
        }

        angle = -0.5 * atan2(2 * central_moment_11, central_moment_20 - central_moment_02);
        radius = centroid.row * sin(angle) + centroid.col * cos(angle);
    }

    Point orthogonal_projection(const Point &point) const
    {
        double a1 = radius * cos(angle);
        double b1 = radius * sin(angle);
        double c1 = -a1 * a1 - b1 * b1;

        double a2 = b1;
        double b2 = -a1;
        double c2 = -a2 * point.col - b2 * point.row;

        size_t x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1);
        size_t y = (c1 * a2 - c2 * a1) / (a1 * b2 - a2 * b1);

        return { y, x };
    }

    int half_axis(const Point &point) const
    /**
     * @param point Point to calculate the half axis for
     * @return negative if point is on the left side of the centroid, positive otherwise
     */
    {
        return point.row * sin(angle) + point.col * cos(angle) - radius;
    }

    static Point get_centroid(const std::vector<Point> &points)
    {
        size_t row = 0, col = 0;
        for (const auto &point : points) {
            row += point.row;
            col += point.col;
        }
        return { row /= points.size(), col /= points.size() };
    }
};
