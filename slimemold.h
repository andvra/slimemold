#pragma once

#include <vector>
#include <boost/compute.hpp>

namespace compute = boost::compute;

constexpr double PI = 3.14159265358979323846;
constexpr float deg2rad(float deg) { return PI * deg / 180.0f; }

struct RunConfiguration {
    struct Environment {
        static const unsigned int width = 200;
        static const unsigned int height = 200;
        static const unsigned int diffusionKernelSize = 3;
        static constexpr float diffusionDecay = 0.1f;
        static const unsigned int populationSize() { return width * height * populationSizeRatio; }
    private:
        static constexpr float populationSizeRatio = 0.15;
    };
    struct Agent {
        static constexpr float sensorAngle = deg2rad(22.5f);
        static constexpr float rotationAngle = deg2rad(45.0f);
        static const unsigned int sensorOffset = 9;
        static const unsigned int sensorWidth = 1;
        static const unsigned int stepSize = 1; // TODO: What is the correct way to handle when an agent doesn't move outside its square?
        static const unsigned int chemoDeposition = 5;
        static constexpr float pRandomChangeDirection = 0.0f;
    };
};

struct Agent {
    float x;
    float y;
    float direction; // Radians
};

class SlimeMold {
public:
    SlimeMold();
    ~SlimeMold();
    virtual void diffusion() = 0;
    virtual void decay() = 0;
    virtual void move() = 0;
    virtual void sense() = 0;
    virtual float senseAtRotation(Agent& agent, float rotationOffset) = 0;
    virtual void makeRenderImage() = 0;
    virtual void deposit(int x, int y) = 0;
    virtual void swapBuffers() = 0;
    std::vector<Agent> initAgents();
    void run();
    unsigned char* getDataTrailRender();
protected:
    unsigned char* dataTrailRender;
    std::vector<unsigned int> getAgentMoveOrder();
};

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
    compute::context ctx;
    compute::command_queue queue;
    compute::kernel kernelDiffuse;
    compute::vector<float> dataTrailCurrent;
};