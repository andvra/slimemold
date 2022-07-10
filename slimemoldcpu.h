#pragma once

#include <vector>

#include "slimemold.h"

class SlimeMoldCpu : public SlimeMold {
public:
    SlimeMoldCpu();
    ~SlimeMoldCpu();
    void diffusion();
    void decay();
    void move();
    void swapBuffers();
    void sense();
    void makeRenderImage();
private:
    float* dataTrailCurrent;
    float* dataTrailNext;
    bool* squareTaken;
    std::vector<Agent> agents;
    float senseAtRotation(Agent& agent, float rotationOffset);
    void measureChemoAroundPosition(int x, int y, int kernelSize, float& totalChemo, int& numMeasuresSquares);
    void deposit(int x, int y);
};

