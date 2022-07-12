#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "slimemoldcpu.h"
#include "slimemoldopencl.h"
#include "runstatistics.h"
#include "videorecorder.h"

int main()
{
    const std::string windowId = "SomeID";
    auto done = false;
    int height = RunConfiguration::Environment::height;
    int width = RunConfiguration::Environment::width;
    RunStatistics stats;
    SlimeMold* slimeMold;

    if (RunConfiguration::Hardware::onlyCpu) {
        slimeMold = new SlimeMoldCpu();
    }
    else {
        slimeMold = new SlimeMoldOpenCl();
    }

    // The image is just a wrapper over the trail data
    cv::Mat imgTrail = cv::Mat(height, width, CV_8UC1, slimeMold->getDataTrailRender());

    cv::namedWindow(windowId);

    //VideoRecorder videoRecorder("running.mp4", 30, RunConfiguration::Environment::width, RunConfiguration::Environment::height, 2);
    //// Save every nth frame
    //int frameSaveFrequency = 25;

    while (!done) {
        slimeMold->run();

        cv::imshow(windowId, imgTrail);

        auto kc = cv::waitKey(1);

        //if (stats.getSteps() % frameSaveFrequency == 0) {
        //    videoRecorder.writeFrame(imgTrail);
        //}

        if (kc == 27) {
            done = true;
        }

        stats.update();

        cv::setWindowTitle(windowId, stats.getStatusString());
    }

    delete slimeMold;

    return 0;
}