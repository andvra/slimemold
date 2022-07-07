#pragma once

#include "utils.h"

struct RunConfiguration {
    struct Hardware {
        static const bool onlyCpu = false;
    };
    struct Environment {
        static const unsigned int width = 1000;
        static const unsigned int height = 1000;
        static const unsigned int diffusionKernelSize = 3;
        static constexpr float diffusionDecay = 0.1f;
        static const unsigned int populationSize() { return static_cast<unsigned int>(width * height * populationSizeRatio); }
    private:
        static constexpr float populationSizeRatio = 0.15f;
    };
    struct Agent {
        static constexpr float sensorAngle = Utils::Math::deg2rad(22.5f);
        static constexpr float rotationAngle = Utils::Math::deg2rad(45.0f);
        static const unsigned int sensorOffset = 9;
        static const unsigned int sensorWidth = 1;
        static const unsigned int stepSize = 1;
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
    Utils::Random* random;
};