#pragma once

#include <string>
#include <opencv2/videoio.hpp>

/// <summary>
/// This class handles video recording, utilizing the OpenCV class VideoWriter. A common use case is turning consecutive window frames into a video. Example use case:
/// 
/// VideoRecorder videoRecorder("output.mp4", 30, windowWidth, windowHeight, 2);
/// cv::Mat img = cv::Mat(windowHeight, windowWidth, CV_8UC1);
/// 
/// while (!done) {
///     ...
///     videoRecorder.writeFrame(img);
///     ...
/// }
/// 
/// </summary>
class VideoRecorder {
public:
    VideoRecorder(const std::string tFileName, const double tFps, const int tFrameWidth, const int tFrameHeight, const double tVideoLengthS);
    ~VideoRecorder();
    // Write a frame to the stream. When all frames are written, the stream is closed. Returns true when the stream is done.
    bool writeFrame(const cv::Mat& frame);
private:
    int frameCounter;
    int numFramesToRecord;
    cv::VideoWriter videoWriter;
};