#pragma once

#include <string>
#include <chrono>

class RunStatistics {
public:
    RunStatistics();

    void update();

    float getFps() const;

    int getSteps() const;

    // Update this to return a proper report
    std::string getStatusString() const;

private:
    int numSteps;
    const float updateFpsIntervalS = 0.5f;
    int framesSinceLastUpdate;
    std::chrono::time_point<std::chrono::system_clock> lastUpdate;
    float fps;
};