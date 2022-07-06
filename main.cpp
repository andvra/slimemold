#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "slimemold.h"

class RunStatistics {
public:
    RunStatistics()  {
        lastUpdate = std::chrono::system_clock::now();
        numSteps = 0;
        framesSinceLastUpdate = 0;
        fps = 0.0f;
    }

    void update() {
        numSteps++;
        framesSinceLastUpdate++;
        std::chrono::duration<float> timeSinceLastUpdateS = std::chrono::system_clock::now() - lastUpdate;
        if (timeSinceLastUpdateS.count() > updateFpsIntervalS) {
            fps = framesSinceLastUpdate / timeSinceLastUpdateS.count();
            lastUpdate = std::chrono::system_clock::now();
            framesSinceLastUpdate = 0;
        }
    }

    float getFps() const {
        return fps;
    }

    int getSteps() const {
        return numSteps;
    }

    // Update this to return a proper report
    std::string getStatusString() {
        return std::to_string(numSteps);
        //return "Steps: " + std::to_string(numSteps) + " FPS: " + std::to_string(fps);
    }

private:
    int numSteps;
    const float updateFpsIntervalS = 0.5f;
    int framesSinceLastUpdate;
    std::chrono::time_point<std::chrono::system_clock> lastUpdate;
    float fps;
};

int main()
{
    const std::string windowId = "SomeID";
    auto done = false;
    int height = RunConfiguration::Environment::height;
    int width = RunConfiguration::Environment::width;
    RunStatistics stats;
    SlimeMold* slimeMold;
    bool runOpenCl = true;

    if (runOpenCl) {
        slimeMold = new SlimeMoldOpenCl();
    }
    else {
        slimeMold = new SlimeMoldCpu();
    }

    cv::Mat imgTrail = cv::Mat(height, width, CV_8UC1, slimeMold->getDataTrailRender());

    while (!done) {
        slimeMold->run();

        cv::imshow(windowId, imgTrail);

        auto kc = cv::waitKey(10);

        if (kc == 27) {
            done = true;
        }

        stats.update();

        cv::setWindowTitle(windowId, stats.getStatusString());
    }

    delete slimeMold;

    return 0;
}