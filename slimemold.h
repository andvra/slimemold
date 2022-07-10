#pragma once

#include "utils.h"

enum class AgentInitPattern {
    Random,
    Circle

};
struct RunConfiguration {
    struct Hardware {
        // True = CPU, false = OpenCL
        static const bool onlyCpu = false;
    };
    struct Environment {
        static const int width = 1000;
        static const int height = 1000;
        static const int diffusionKernelSize = 3;
        static constexpr float diffusionDecay = 0.1f;
        static const int populationSize() { return static_cast<unsigned int>(width * height * populationSizeRatio); }
        static const int numPixels() { return width * height; }
        static const AgentInitPattern initPattern = AgentInitPattern::Random;
    private:
        static constexpr float populationSizeRatio = 0.15f;
    };
    struct Agent {
        static constexpr float sensorAngle = Utils::Math::deg2rad(22.5f);
        static constexpr float rotationAngle = Utils::Math::deg2rad(45.0f);
        static const int sensorOffset = 9;
        static const int sensorWidth = 1;
        static const int stepSize = 1;
        static const int chemoDeposition = 5;
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
    virtual void makeRenderImage() = 0;
    virtual void swapBuffers() = 0;
    std::vector<Agent> initAgents();
    void run();
    unsigned char* getDataTrailRender();
protected:
    unsigned char* dataTrailRender;
    // Agent move will be blocked if there's another agent at the desired position. To Avoid bias, we'll
    //  randomize the order of which the agents move every step
    std::vector<int> getAgentMoveOrder();
    Utils::Random* random;
};