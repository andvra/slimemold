#pragma once

#include "utils.h"

/*
To add a custom type :

1. Add BOOST_COMPUTE_ADAPT_STRUCT in the header.This makes sure to pad the structure so it can be used with OpenCL
2. Add the type definition to the top of the kernel source code : compute::type_definition<NameOfCustomType>() + "\n" + <READ_CODE>;
3. Build the kernels as normal
*/

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
        static const int width = 1920;
        static const int height = 1080;
        static const int diffusionKernelSize = 3;
        static constexpr float diffusionDecay = 0.1f;
        // Blending factor when blurring. 0 = no blur (keep current pixel value), 1 = use kernel output
        static constexpr float diffusionRatio = 0.2f;
        static const int populationSize() { return static_cast<int>(width * height * populationSizeRatio); }
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
        static constexpr float maxTotalChemo = 255.999f;
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