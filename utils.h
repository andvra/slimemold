#pragma once

#include <vector>
#include <random>
#include <functional>

namespace Utils {
    constexpr double PI = 3.14159265358979323846;

    struct Files {
        static std::string readAllFile(std::string path);
    };

    struct Math {
        static constexpr float deg2rad(float deg) { return static_cast<float>(PI) * deg / 180.0f; }
        template <class T>
        static T clamp(T min, T max, T v) {
            if (v < min) {
                return min;
            }
            else if (v > max) {
                return max;
            }

            return v;
        }
    };

    struct Random {
        Random();
        template <class T>
        void shuffleVector(std::vector<T>& v);
        float randFloat();
        float randomDirection();
    private:
        std::uniform_real_distribution<float> floatDist;
        std::mt19937_64 engine;
    };

    extern void runThreaded(const std::function<void(int, int)>& fn, int elementIdxStart, int elementIdxEnd);
};

template <class T>
void Utils::Random::shuffleVector(std::vector<T>& v) {
    std::shuffle(v.begin(), v.end(), engine);
}