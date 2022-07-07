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
    void deposit(int x, int y);
    void swapBuffers();
    void sense();
    float senseAtRotation(Agent& agent, float rotationOffset);
    void makeRenderImage();
private:
    float* dataTrailCurrent;
    float* dataTrailNext;
    bool* squareTaken;
    std::vector<Agent> agents;
};

