#include <sstream>
#include <fstream>
#include <thread>

#include "utils.h"

std::string Utils::Files::readAllFile(std::string path) {
    std::ifstream f(path);
    std::stringstream buffer;
    buffer << f.rdbuf();

    return buffer.str();
}

Utils::Random::Random() {
    floatDist = std::uniform_real_distribution<float>(0.0f, 1.0f);
}

float Utils::Random::randFloat() {
    return floatDist(engine);
}

float Utils::Random::randomDirection() {
    return 2.0f * static_cast<float>(PI) * randFloat();
}

void Utils::runThreaded(const std::function<void(int, int)>& fn, int elementIdxStart, int elementIdxEnd) {
    const int numThreads = std::thread::hardware_concurrency();
    int totalElements = elementIdxEnd - elementIdxStart;
    int batchSize = totalElements / numThreads;
    std::vector<std::thread> threads;

    for (int idxThread = 0; idxThread < numThreads; idxThread++) {
        auto colStart = idxThread * batchSize;
        auto colEndExclusive = std::min(totalElements, (idxThread + 1) * batchSize);
        threads.push_back(std::thread(fn, colStart, colEndExclusive));
    }

    std::for_each(threads.begin(), threads.end(), [](std::thread& t)
        {
            t.join();
        });
}