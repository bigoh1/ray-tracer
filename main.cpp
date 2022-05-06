#include <pngwriter.h>
#include <utility>
#include <vector>
#include <cstddef>
#include <eigen3/Eigen/Dense>
#include <memory>
#include <optional>
#include <limits>

using namespace std;

using vec = Eigen::Vector3d;
using cvecref = const Eigen::Vector3d &;
using Nvec = Eigen::VectorXd;

struct Sphere {
    Sphere(vec center, double radius, vec ambience, vec diffuse, vec specular, double shininess, double reflection) : cent{std::move(center)}, R{radius}, amb{std::move(ambience)},
                                                                                       diff{std::move(diffuse)}, spec{std::move(specular)}, shin{shininess}, refl{reflection} {}

    vec cent, amb, diff, spec;
    double R, shin, refl;
};

struct Light {
    Light(vec center, vec ambience, vec diffuse, vec specular) : cent{std::move(center)}, amb{std::move(ambience)},
    diff{std::move(diffuse)}, spec{std::move(specular)} {}
    vec cent, amb, diff, spec;
};

auto linspace(double start, double stop, size_t num) -> Nvec {
    // start + step*(num - 1) = stop
    Nvec res(num);
    double step = (stop - start)/(static_cast<double>(num) - 1);
    for (int i = 0; i < num; ++i) {
        res(i) = start + step * i;
    }
    return res;
}

optional<double> sphere_intersect(cvecref center, double radius, cvecref origin, cvecref direction) {
    double b = 2 * direction.dot(origin - center);
    double c = (origin - center).squaredNorm() - radius * radius;
    double delta = b * b - 4 * c;
    if (delta > 0) {
        double t1 = (-b + sqrt(delta)) / 2;
        double t2 = (-b - sqrt(delta)) / 2;
        if (t1 > 0 && t2 > 0)
            return {min(t1, t2)};
    }
    return nullopt;
}

pair<shared_ptr<Sphere>, double> nearest_intersect(const vector<Sphere> &objects, cvecref origin, cvecref direction) {
    double nearest_dist = numeric_limits<double>::max();
    shared_ptr<Sphere> nearest_obj(nullptr);

    for (auto &obj : objects) {
        auto dist_ = sphere_intersect(obj.cent, obj.R, origin, direction);
        if (!dist_.has_value())
            continue;
        auto dist = dist_.value();

        if (dist < nearest_dist) {
            nearest_dist = dist;
            nearest_obj = make_shared<Sphere>(obj);
        }
    }

    return {nearest_obj, nearest_dist};
}

auto multiply(cvecref a, cvecref b) -> vec {
    vec res;
    for (int i = 0; i < a.size(); ++i) {
        res(i) = a(i) * b(i);
    }
    return res;
}

auto reflect(cvecref v, cvecref normal) -> vec {
    return v - 2 * v.dot(normal) * normal;
}

int main() {
    vector<Sphere> objects = {
            Sphere({0.1, 0.3, 0.25}, 0.1, {0.1, 0, 0}, {0.7, 0, 0}, {1, 1, 1}, 100, .5),
            Sphere({.1, -.3, 0.12}, 0.1, {0.1, 0, 0.1}, {0.7, 0, 0.7}, {1, 1, 1}, 100, .5),
            Sphere({-.3, 0, 0.50}, 0.23, {0, 0.1, 0}, {0, 0.6, 0}, {1, 1, 1}, 100, .5),
            Sphere({0, -9000, 0}, 9000 - 0.7, {0.1, 0.1, 0.1}, {0.6, 0.6, 0.6}, {1, 1, 1}, 100, .5),
    };

    // TODO...
//    vector<Light> light = {
//
//    };

    const int HEIGHT = 2000, WIDTH = 2000;
    const vec camera = {0, 0, 1};
    const auto light = Light({5, 5, 5}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1});
    pngwriter image(WIDTH, HEIGHT, 0.0, "image.png");

    const int MAX_DEPTH = 10;

    double screen_right = 1, screen_top = static_cast<double>(HEIGHT)/WIDTH;
    double screen_left = -screen_right, screen_bottom = -screen_top;

    auto height_pixels = linspace(screen_bottom, screen_top, HEIGHT);
    auto width_pixels = linspace(screen_left, screen_right, WIDTH);

    for (int i = 0; i < height_pixels.size(); ++i) {
        for (int j = 0; j < width_pixels.size(); ++j) {
            double y = height_pixels[i], x = width_pixels[j];
            vec pixel = {x, y, 0};
            vec illumination = {0, 0, 0};
            double reflection = 1;

            auto origin = camera;
            auto direction = (pixel - origin).normalized();

            for (int k = 0; k < MAX_DEPTH; ++k) {
                auto[near_obj, near_dist] = nearest_intersect(objects, origin, direction);
                if (!near_obj)
                    break;
                auto intersection = origin + direction * near_dist;
                auto surface_normal = (intersection - near_obj->cent).normalized();
                auto shifted = intersection + surface_normal * 1e-5;

                auto to_light = (light.cent - shifted).normalized();
                auto dist_to_light = to_light.norm();
                auto [_, min_dist] = nearest_intersect(objects, shifted, to_light);
                bool is_shadowed = min_dist < dist_to_light;

                if (is_shadowed) {
                    break;
                }

                vec color = {0, 0, 0};
                // TODO: if there are multiple light sources, then amb is the sum of them all
                color += multiply(near_obj->amb, light.amb);


                color += to_light.dot(surface_normal) * multiply(light.diff, near_obj->diff);

                auto to_origin = (origin - shifted).normalized();
                auto H = (to_origin + to_light).normalized();
                color += pow(H.dot(surface_normal), near_obj->shin) * multiply(light.spec, near_obj->spec);

                reflection *= near_obj->refl;
                illumination += reflection * color;

                origin = intersection;
                direction = reflect(direction, surface_normal);
            }

            //illumination = illumination.cwiseMin(0).cwiseMax(1);
            image.plot(j + 1, i + 1, illumination(0), illumination(1), illumination(2));
        }

        cout << i+1 << "/" << HEIGHT << endl;
    }

    image.close();

    return 0;
}