#include "runstatistics.h"

RunStatistics::RunStatistics() {
    lastUpdate = std::chrono::system_clock::now();
    numSteps = 0;
    framesSinceLastUpdate = 0;
    fps = 0.0f;
}

void RunStatistics::update() {
    numSteps++;
    framesSinceLastUpdate++;
    std::chrono::duration<float> timeSinceLastUpdateS = std::chrono::system_clock::now() - lastUpdate;

    if (timeSinceLastUpdateS.count() > updateFpsIntervalS) {
        fps = framesSinceLastUpdate / timeSinceLastUpdateS.count();
        lastUpdate = std::chrono::system_clock::now();
        framesSinceLastUpdate = 0;
    }
}

float RunStatistics::getFps() const {
    return fps;
}

int RunStatistics::getSteps() const {
    return numSteps;
}

std::string RunStatistics::getStatusString() const {
    return std::to_string(numSteps);
    //return "Steps: " + std::to_string(numSteps) + " FPS: " + std::to_string(fps);
}
