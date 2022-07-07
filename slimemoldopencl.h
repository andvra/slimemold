#pragma once

#define CL_TARGET_OPENCL_VERSION 220

#include <boost/compute.hpp>

#include "slimemold.h"

namespace compute = boost::compute;

class SlimeMoldOpenCl : public SlimeMold {
public:
    SlimeMoldOpenCl();
    void diffusion();
    void decay() {}
    void move() {}
    void sense() {}
    float senseAtRotation(Agent& agent, float rotationOffset) { return 0.0f; }
    void makeRenderImage();
    void deposit(int x, int y) {}
    void swapBuffers() {}

private:
    void loadKernels();
    compute::context ctx;
    compute::command_queue queue;
    compute::kernel kernelDiffuse;
    compute::vector<float> dataTrailCurrent;
};