#pragma once

#define CL_TARGET_OPENCL_VERSION 220

#include <boost/compute.hpp>

#include "slimemold.h"

namespace compute = boost::compute;

// Since we made our main configuration structure static, we need to make a copy to be able to put it on the device.
//  We could of course force the configuration to be generated, but we'll stick with this for now.
struct RunConfigurationCl {
    RunConfigurationCl() :
        hwOnlyCpu(RunConfiguration::Hardware::onlyCpu),
        envWidth(RunConfiguration::Environment::width),
        envHeight(RunConfiguration::Environment::height),
        envDiffusionKernelSize(RunConfiguration::Environment::diffusionKernelSize),
        envDiffusionDecay(RunConfiguration::Environment::diffusionDecay),
        envDiffusionRatio(RunConfiguration::Environment::diffusionRatio),
        envPopulationSize(RunConfiguration::Environment::populationSize()),
        agentSensorAngle(RunConfiguration::Agent::sensorAngle),
        agentRotationAngle(RunConfiguration::Agent::rotationAngle),
        agentSensorOffset(RunConfiguration::Agent::sensorOffset),
        agentSensorWidth(RunConfiguration::Agent::sensorWidth),
        agentStepSize(RunConfiguration::Agent::stepSize),
        agentChemoDeposition(RunConfiguration::Agent::chemoDeposition),
        agentpRandomChangeDirection(RunConfiguration::Agent::pRandomChangeDirection),
        agentMaxTotalChemo(RunConfiguration::Agent::maxTotalChemo) {}
    // Hardware
    int hwOnlyCpu;
    // Environment
    unsigned int envWidth;
    unsigned int envHeight;
    unsigned int envDiffusionKernelSize;
    float envDiffusionDecay;
    float envDiffusionRatio;
    unsigned int envPopulationSize;
    // Agent
    float agentSensorAngle;
    float agentRotationAngle;
    unsigned int agentSensorOffset;
    unsigned int agentSensorWidth;
    unsigned int agentStepSize;
    unsigned int agentChemoDeposition;
    float agentpRandomChangeDirection;
    float agentMaxTotalChemo;
};

// Make the Agent struct usable in OpenCL. Rename it to AgentCL so we can easily identify it
BOOST_COMPUTE_ADAPT_STRUCT(Agent, Agent, (x, y, direction))
// NB Make sure to list all the members! Otherwise there will be a compile-time error C2338
BOOST_COMPUTE_ADAPT_STRUCT(RunConfigurationCl, RunConfigurationCl, (hwOnlyCpu, envWidth, envHeight, envDiffusionKernelSize, envDiffusionDecay, envDiffusionRatio, envPopulationSize, agentSensorAngle, agentRotationAngle, agentSensorOffset, agentSensorWidth, agentStepSize, agentChemoDeposition, agentpRandomChangeDirection, agentMaxTotalChemo))

class SlimeMoldOpenCl : public SlimeMold {
public:
    SlimeMoldOpenCl();
    void diffusion();
    void decay();
    void move();
    void sense();
    void makeRenderImage();
    void swapBuffers();

private:
    void loadKernels();
    void loadAgents();
    void loadConfig();
    void loadDeviceMemory();
    void loadHostMemory();
    void loadVariables();
    void loadDeviceMemoryTrailMaps();
    // Whether a move is valid or not depends on if another agent is moving into that square.
    //  So we need a function to synchronize movement when we have the desired movements of each agent
    void moveCoordinate();
    void moveDesiredMoves();
    void moveActualMove();
    std::map<std::string, compute::kernel> kernels;
    compute::context ctx;
    compute::command_queue queue;
    std::vector<compute::vector<float>> dDataTrails;
    compute::vector<Agent> dAgents;
    // Desired position of agents
    compute::vector<Agent> dAgentDesired;
    compute::vector<int> dDesiredDestinationIndices;
    compute::vector<float> dNewDirection;
    // User to make random rotation when sensing
    compute::vector<float> dRandomValues;
    // TODO: Added as a vector here, since we know how to work with that. How do we create a custom type compute variable, not using a vector?
    compute::vector<RunConfigurationCl> dConfig;
    std::vector<unsigned char> hTakenMap;
    std::vector<int> hDesiredDestinationIdx;
    std::vector<float> hDataTrailCurrent;
    // New random directions, sent to device
    std::vector<float> hNewDirection;
    int idxDataTrailInUse, idxDataTrailBuffer;
};