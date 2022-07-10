#include <cmath>

#include "slimemoldcpu.h"
#include "utils.h"

int xyToSlimeArrayIdx(float x, float y) {
    return static_cast<int>(x) + static_cast<int>(y) * RunConfiguration::Environment::width;
}

int xyToSlimeArrayIdx(int x, int y) {
    return x + y * RunConfiguration::Environment::width;
}

bool coordValid(int x, int y) {
    return (x >= 0 && x < RunConfiguration::Environment::width&& y >= 0 && y < RunConfiguration::Environment::height);
}

SlimeMoldCpu::SlimeMoldCpu() : SlimeMold() {
    const int numPixels = RunConfiguration::Environment::numPixels();

    dataTrailCurrent = new float[numPixels]();
    dataTrailNext = new float[numPixels]();
    squareTaken = std::vector<bool>(numPixels);
    agents = initAgents();
}

SlimeMoldCpu::~SlimeMoldCpu() {
    delete[] dataTrailCurrent;
    delete[] dataTrailNext;
}

void SlimeMoldCpu::diffusion() {
    const auto cols = RunConfiguration::Environment::width;
    const auto rows = RunConfiguration::Environment::height;
    const int kernelSize = RunConfiguration::Environment::diffusionKernelSize;
    const float diffuseRate = RunConfiguration::Environment::diffusionRatio;

    auto calc = [this, kernelSize, rows, diffuseRate](int colStart, int colEndExclusive) -> void {
        for (int col = colStart; col < colEndExclusive; col++) {
            for (int row = 0; row < rows; row++) {
                auto idxDest = xyToSlimeArrayIdx(col, row);
                float totalChemo;
                int numMeasuredSquares;
                measureChemoAroundPosition(col, row, kernelSize, totalChemo, numMeasuredSquares);
                float blurredVal = totalChemo / (kernelSize * kernelSize + 1);
                float newVal = diffuseRate * blurredVal + (1 - diffuseRate) * dataTrailCurrent[idxDest];
                dataTrailNext[idxDest] = newVal;
            }
        }
    };

    Utils::runThreaded(calc, 0, cols);
}

float SlimeMoldCpu::validChemo(float v) {
    return Utils::Math::clamp<float>(0.0f, RunConfiguration::Agent::maxTotalChemo, v);
}

void SlimeMoldCpu::decay() {
    const auto numIndices = RunConfiguration::Environment::width * RunConfiguration::Environment::height;
    const auto decay = RunConfiguration::Environment::diffusionDecay;

    for (int i = 0; i < numIndices; i++) {
        dataTrailCurrent[i] = validChemo(dataTrailCurrent[i] - decay);
    }
}

void SlimeMoldCpu::move() {
    auto moveOrder = getAgentMoveOrder();
    auto stepSize = RunConfiguration::Agent::stepSize;
    auto width = RunConfiguration::Environment::width;
    auto height = RunConfiguration::Environment::height;
    auto numPixels = RunConfiguration::Environment::numPixels();

    for (int i = 0; i < numPixels; i++) {
        squareTaken[i] = false;
    }

    std::fill(squareTaken.begin(), squareTaken.end(), false);

    for (int i = 0; i < agents.size(); i++) {
        auto& agent = agents[moveOrder[i]];
        auto newX = agent.x + std::cos(agent.direction) * stepSize;
        auto newY = agent.y + std::sin(agent.direction) * stepSize;
        auto newXSquare = static_cast<int>(newX);
        auto newYSquare = static_cast<int>(newY);
        auto trailIdx = newXSquare + newYSquare * width;
        if (newX >= 0
            && newX < width
            && newY >= 0
            && newY < height
            && !squareTaken[trailIdx]
            ) {
            agent.x = newX;
            agent.y = newY;
            deposit(newXSquare, newYSquare);
            squareTaken[trailIdx] = true;
        }
        else {
            agent.direction = random->randomDirection();
        }
    }
}

void SlimeMoldCpu::deposit(int x, int y) {
    auto chemoDeposition = RunConfiguration::Agent::chemoDeposition;
    auto idx = xyToSlimeArrayIdx(x, y);

    dataTrailCurrent[idx] = validChemo(dataTrailCurrent[idx] + chemoDeposition);
}

void SlimeMoldCpu::swapBuffers() {
    std::swap(dataTrailCurrent, dataTrailNext);
}

void SlimeMoldCpu::sense() {
    auto rotationAngle = RunConfiguration::Agent::rotationAngle;
    auto sensorAngle = RunConfiguration::Agent::sensorAngle;

    for (int i = 0; i < agents.size(); i++) {
        auto& agent = agents[i];
        auto senseLeft = senseAtRotation(agent, -sensorAngle);
        auto senseForward = senseAtRotation(agent, 0.0f);
        auto senseRight = senseAtRotation(agent, sensorAngle);

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
    auto x = static_cast<int>(agent.x + RunConfiguration::Agent::sensorOffset * std::cos(agent.direction + rotationOffset));
    auto y = static_cast<int>(agent.y + RunConfiguration::Agent::sensorOffset * std::sin(agent.direction + rotationOffset));
    auto sensorWidth = RunConfiguration::Agent::sensorWidth;

    float totalChemo;
    int numSquares;
    measureChemoAroundPosition(x, y, sensorWidth, totalChemo, numSquares);

    if (numSquares==0) {
        return 0;
    }
    else {
        return totalChemo;
    }
}

void SlimeMoldCpu::measureChemoAroundPosition(int x, int y, int kernelSize, float& totalChemo, int& numMeasuresSquares) {
    totalChemo = 0.0f;
    numMeasuresSquares = 0;

    for (int xd = x - kernelSize / 2; xd <= x + kernelSize / 2; xd++) {
        for (int yd = y - kernelSize / 2; yd <= y + kernelSize / 2; yd++) {
            if (coordValid(xd, yd)) {
                auto idxSrc = xyToSlimeArrayIdx(xd, yd);
                numMeasuresSquares++;
                totalChemo += dataTrailCurrent[idxSrc];
            }
        }
    }
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
