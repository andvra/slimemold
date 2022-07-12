#include "videorecorder.h"

VideoRecorder::VideoRecorder(const std::string tFileName, const double tFps, const int tFrameWidth, const int tFrameHeight, const double tVideoLengthS) {
    int fourCC = -1;
    cv::Size frameSize(tFrameWidth, tFrameHeight);

    frameCounter = 0;
    numFramesToRecord = static_cast<int>(tFps * tVideoLengthS);
    videoWriter = cv::VideoWriter(tFileName, cv::CAP_ANY, fourCC, tFps, frameSize, false);
    videoWriter.set(cv::VIDEOWRITER_PROP_QUALITY, 100);
}

VideoRecorder::~VideoRecorder() {
    if (videoWriter.isOpened()) {
        videoWriter.release();
    }
}

bool VideoRecorder::writeFrame(const cv::Mat& frame) {
    if (frameCounter == numFramesToRecord) {
        return true;
    }

    if (!videoWriter.isOpened()) {
        return true;
    }

    videoWriter << frame;

    if (++frameCounter == numFramesToRecord) {
        videoWriter.release();
        return true;
    }
    else {
        return false;
    }
}