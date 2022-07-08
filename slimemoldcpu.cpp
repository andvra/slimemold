#include <cmath>

#include "slimemoldcpu.h"
#include "utils.h"

int xyToSlimeArrayIdx(float x, float y) {
    return static_cast<int>(x) + static_cast<int>(y) * RunConfiguration::Environment::width;
}

int xyToSlimeArrayIdx(int x, int y) {
    return static_cast<int>(x + y * RunConfiguration::Environment::width);
}

bool coordValid(int x, int y) {
    return (x >= 0 && x < RunConfiguration::Environment::width&& y >= 0 && y < RunConfiguration::Environment::height);
}

template <class T>
T clamp(T min, T max, T v) {
    if (v < min) {
        return min;
    }
    else if (v > max) {
        return max;
    }

    return v;
}

SlimeMoldCpu::SlimeMoldCpu() : SlimeMold() {
    const int imgWidth = RunConfiguration::Environment::width;
    const int imgHeight = RunConfiguration::Environment::height;
    dataTrailCurrent = new float[imgWidth * imgHeight]();
    dataTrailNext = new float[imgWidth * imgHeight]();
    squareTaken = new bool[imgWidth * imgHeight];
    agents = initAgents();
}

SlimeMoldCpu::~SlimeMoldCpu() {
    delete[] dataTrailCurrent;
    delete[] dataTrailNext;
    delete[] squareTaken;
}

void SlimeMoldCpu::diffusion() {
    const auto cols = RunConfiguration::Environment::width;
    const auto rows = RunConfiguration::Environment::height;
    const int kernelSize = RunConfiguration::Environment::diffusionKernelSize;

    for (int col = 0; col < cols; col++) {
        for (int row = 0; row < rows; row++) {
            auto idxDest = xyToSlimeArrayIdx(col, row);
            float chemo = 0.0f;
            int numSquares = 0;
            for (int xd = col - kernelSize / 2; xd <= col + kernelSize / 2; xd++) {
                if (xd >= 0 && xd < cols) {
                    for (int yd = row - kernelSize / 2; yd <= row + kernelSize / 2; yd++) {
                        if (yd >= 0 && yd < rows) {
                            auto idxSrc = xyToSlimeArrayIdx(xd, yd);
                            numSquares++;
                            chemo += dataTrailCurrent[idxSrc];
                        }
                    }
                }
            }
            dataTrailNext[idxDest] = chemo / numSquares;
        }
    }
}

void SlimeMoldCpu::decay() {
    const auto numIndices = RunConfiguration::Environment::width * RunConfiguration::Environment::height;
    const auto decay = RunConfiguration::Environment::diffusionDecay;

    for (int i = 0; i < numIndices; i++) {
        dataTrailCurrent[i] = clamp<float>(0.0f, 255.999f, dataTrailCurrent[i] - decay);
    }
}

void SlimeMoldCpu::move() {
    auto moveOrder = getAgentMoveOrder();
    
    for (int i = 0; i < RunConfiguration::Environment::width * RunConfiguration::Environment::height; i++) {
        squareTaken[i] = false;
    }

    for (int i = 0; i < agents.size(); i++) {
        auto& agent = agents[moveOrder[i]];
        auto newX = agent.x + std::cos(agent.direction) * RunConfiguration::Agent::stepSize;
        auto newY = agent.y + std::sin(agent.direction) * RunConfiguration::Agent::stepSize;
        auto newXSquare = static_cast<int>(newX);
        auto newYSquare = static_cast<int>(newY);
        if (newX >= 0
            && newX < RunConfiguration::Environment::width
            && newY >= 0
            && newY < RunConfiguration::Environment::height
            && !squareTaken[newXSquare + newYSquare * RunConfiguration::Environment::width]
            ) {
            agent.x = newX;
            agent.y = newY;
            deposit(newXSquare, newYSquare);
        }
        else {
            agent.direction = random->randomDirection();
        }
    }
}

void SlimeMoldCpu::deposit(int x, int y) {
    auto idx = xyToSlimeArrayIdx(x, y);
    dataTrailCurrent[idx] = clamp<float>(0.0f, 255.999f, dataTrailCurrent[idx] + RunConfiguration::Agent::chemoDeposition);
}

void SlimeMoldCpu::swapBuffers() {
    std::swap(dataTrailCurrent, dataTrailNext);
}

void SlimeMoldCpu::sense() {
    auto rotationAngle = RunConfiguration::Agent::rotationAngle;
    for (int i = 0; i < agents.size(); i++) {
        auto& agent = agents[i];
        auto senseLeft = senseAtRotation(agent, -RunConfiguration::Agent::sensorAngle);
        auto senseForward = senseAtRotation(agent, 0.0f);
        auto senseRight = senseAtRotation(agent, RunConfiguration::Agent::sensorAngle);

        if (senseForward > senseLeft && senseForward > senseRight) {
            // Do nothing
        }
        else if (senseForward < senseLeft && senseForward < senseRight) {
            // Rotate in random direction
            agent.direction += (random->randFloat() > 0.5) ? -rotationAngle : rotationAngle;
        }
        else if (senseLeft < senseRight) {
            agent.direction += rotationAngle;
        }
        else {
            agent.direction -= rotationAngle;
        }
    }
}

float SlimeMoldCpu::senseAtRotation(Agent& agent, float rotationOffset) {
    int x = static_cast<int>(agent.x + RunConfiguration::Agent::sensorOffset * std::cos(agent.direction + rotationOffset));
    int y = static_cast<int>(agent.y + RunConfiguration::Agent::sensorOffset * std::sin(agent.direction + rotationOffset));

    if (coordValid(x, y)) {
        return dataTrailCurrent[xyToSlimeArrayIdx(x, y)];
    }

    return 0;
}

void SlimeMoldCpu::makeRenderImage() {
    int cols = RunConfiguration::Environment::width;
    int rows = RunConfiguration::Environment::height;

    for (auto col = 0; col < cols; col++) {
        for (auto row = 0; row < rows; row++) {
            auto idx = xyToSlimeArrayIdx(col, row);
            dataTrailRender[idx] = static_cast<unsigned char>(dataTrailCurrent[idx]);
        }
    }
}
