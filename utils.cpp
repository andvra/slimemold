#include "utils.h"

using namespace Utils;

Random::Random() {
    floatDist = std::uniform_real_distribution<float>(0.0f, 1.0f);
}

float Random::randFloat() {
    return floatDist(engine);
}

float Random::randomDirection() {
    return 2.0f * PI * randFloat();
}