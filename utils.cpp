#include <sstream>
#include <fstream>

#include "utils.h"

using namespace Utils;

std::string Files::readAllFile(std::string path) {
    std::ifstream f(path);
    std::stringstream buffer;
    buffer << f.rdbuf();

    return buffer.str();
}

Random::Random() {
    floatDist = std::uniform_real_distribution<float>(0.0f, 1.0f);
}

float Random::randFloat() {
    return floatDist(engine);
}

float Random::randomDirection() {
    return 2.0f * static_cast<float>(PI) * randFloat();
}