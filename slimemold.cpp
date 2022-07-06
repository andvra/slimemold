#include <random>

#include "slimemold.h"

std::uniform_real_distribution<float> floatDist(0.0f, 1.0f);
std::mt19937_64 engine;

float randFloat() {
    return floatDist(engine);
}

unsigned int xyToSlimeArrayIdx(float x, float y) {
    return static_cast<unsigned int>(x) + static_cast<unsigned int>(y) * RunConfiguration::Environment::width;
}

template <class T>
T clamp(T min, T max, T v) {
    if (v < min) {
        return min;
    }
    else if (v > max) {
        return max;
    }

    return v;
}

SlimeMold::SlimeMold() {
    const int imgWidth = RunConfiguration::Environment::width;
    const int imgHeight = RunConfiguration::Environment::height;
    dataTrailRender = new unsigned char[imgWidth * imgHeight];
}

SlimeMold::~SlimeMold() {
    delete[] dataTrailRender;
}

std::vector<Agent> SlimeMold::initAgents() {
    std::vector<Agent> agents(RunConfiguration::Environment::populationSize(), Agent());

    for (auto& agent : agents) {
        agent.direction = 2.0f * PI * randFloat();
        agent.x = RunConfiguration::Environment::width * randFloat();
        agent.y = RunConfiguration::Environment::height * randFloat();
    }

    return agents;
}

void SlimeMold::run() {
    diffusion();
    swapBuffers();
    decay();
    move();
    sense();
    makeRenderImage();
}

std::vector<unsigned int> SlimeMold::getAgentMoveOrder() {
    std::vector<unsigned int> agentMoveOrder(RunConfiguration::Environment::populationSize(), 0);

    for (auto i = 0; i < RunConfiguration::Environment::populationSize(); i++) {
        agentMoveOrder[i] = i;
    }

    // The first agent that moves into a position is the only one that can stay there.
    //  To avoid bias, randomize move order at each step.
    std::shuffle(agentMoveOrder.begin(), agentMoveOrder.end(), engine);

    return agentMoveOrder;
}

unsigned char* SlimeMold::getDataTrailRender() {
    return dataTrailRender;
}

SlimeMoldCpu::SlimeMoldCpu() : SlimeMold() {
    const int imgWidth = RunConfiguration::Environment::width;
    const int imgHeight = RunConfiguration::Environment::height;
    dataTrailCurrent = new float[imgWidth * imgHeight]();
    dataTrailNext = new float[imgWidth * imgHeight]();
    squareTaken = new bool[imgWidth * imgHeight];
    agents = initAgents();
}

SlimeMoldCpu::~SlimeMoldCpu() {
    delete[] dataTrailCurrent;
    delete[] dataTrailNext;
    delete[] squareTaken;
}

void SlimeMoldCpu::diffusion() {
    const auto cols = RunConfiguration::Environment::width;
    const auto rows = RunConfiguration::Environment::height;
    const int kernelSize = RunConfiguration::Environment::diffusionKernelSize;

    for (int col = 0; col < cols; col++) {
        for (int row = 0; row < rows; row++) {
            auto idxDest = xyToSlimeArrayIdx(col, row);
            float chemo = 0.0f;
            int numSquares = 0;
            for (int xd = col - kernelSize / 2; xd <= col + kernelSize / 2; xd++) {
                if (xd >= 0 && xd < cols) {
                    for (int yd = row - kernelSize / 2; yd <= row + kernelSize / 2; yd++) {
                        if (yd >= 0 && yd < rows) {
                            auto idxSrc = xyToSlimeArrayIdx(xd, yd);
                            numSquares++;
                            chemo += dataTrailCurrent[idxSrc];
                        }
                    }
                }
            }
            dataTrailNext[idxDest] = chemo / numSquares;
        }
    }
}

void SlimeMoldCpu::decay() {
    const auto numIndices = RunConfiguration::Environment::width * RunConfiguration::Environment::height;
    const auto decay = RunConfiguration::Environment::diffusionDecay;

    for (int i = 0; i < numIndices; i++) {
        dataTrailCurrent[i] = clamp<float>(0.0f, 255.999f, dataTrailCurrent[i] - decay);
    }
}

void SlimeMoldCpu::move() {
    auto moveOrder = getAgentMoveOrder();

    for (int i = 0; i < RunConfiguration::Environment::width * RunConfiguration::Environment::height; i++) {
        squareTaken[i] = false;
    }

    for (int i = 0; i < agents.size(); i++) {
        auto& agent = agents[moveOrder[i]];
        auto newX = agent.x + std::cos(agent.direction) * RunConfiguration::Agent::stepSize;
        auto newY = agent.y + std::sin(agent.direction) * RunConfiguration::Agent::stepSize;
        auto newXSquare = static_cast<unsigned int>(newX);
        auto newYSquare = static_cast<unsigned int>(newY);
        if (newX >= 0
            && newX < RunConfiguration::Environment::width
            && newY >= 0
            && newY < RunConfiguration::Environment::height
            && !squareTaken[newXSquare + newYSquare * RunConfiguration::Environment::width]
            ) {
            agent.x = newX;
            agent.y = newY;
            deposit(newXSquare, newYSquare);
        }
        else {
            agent.direction = 2.0f * PI * randFloat();
        }
    }
}

void SlimeMoldCpu::deposit(int x, int y) {
    auto idx = xyToSlimeArrayIdx(x, y);
    dataTrailCurrent[idx] = clamp<float>(0.0f, 255.999f, dataTrailCurrent[idx] + RunConfiguration::Agent::chemoDeposition);
}

void SlimeMoldCpu::swapBuffers() {
    std::swap(dataTrailCurrent, dataTrailNext);
}

void SlimeMoldCpu::sense() {
    auto rotationAngle = RunConfiguration::Agent::rotationAngle;
    for (int i = 0; i < agents.size(); i++) {
        auto& agent = agents[i];
        auto senseLeft = senseAtRotation(agent, -RunConfiguration::Agent::sensorAngle);
        auto senseForward = senseAtRotation(agent, 0.0f);
        auto senseRight = senseAtRotation(agent, RunConfiguration::Agent::sensorAngle);

        if (senseForward > senseLeft && senseForward > senseRight) {
            // Do nothing
        }
        else if (senseForward < senseLeft && senseForward < senseRight) {
            // Rotate in random direction
            agent.direction += (randFloat() > 0.5) ? -rotationAngle : rotationAngle;
        }
        else if (senseLeft < senseRight) {
            agent.direction += rotationAngle;
        }
        else {
            agent.direction -= rotationAngle;
        }
    }
}

float SlimeMoldCpu::senseAtRotation(Agent& agent, float rotationOffset) {
    // TODO: Y is 0 at the top - should we invert the angle?
    unsigned int x = static_cast<unsigned int>(agent.x + RunConfiguration::Agent::sensorOffset * std::cos(agent.direction + rotationOffset));
    unsigned int y = static_cast<unsigned int>(agent.y + RunConfiguration::Agent::sensorOffset * std::sin(agent.direction + rotationOffset));

    return dataTrailCurrent[xyToSlimeArrayIdx(x, y)];
}

void SlimeMoldCpu::makeRenderImage() {
    auto cols = RunConfiguration::Environment::width;
    auto rows = RunConfiguration::Environment::height;

    for (auto col = 0; col < cols; col++) {
        for (auto row = 0; row < rows; row++) {
            auto idx = xyToSlimeArrayIdx(col, row);
            dataTrailRender[idx] = static_cast<unsigned char>(dataTrailCurrent[idx]);
        }
    }
}

SlimeMoldOpenCl::SlimeMoldOpenCl() : SlimeMold() {
    compute::device gpu = compute::system::default_device();

    std::cout << "Using device: " << gpu.name() << std::endl;

    ctx = compute::context(gpu);
    queue = compute::command_queue(ctx, gpu);

    const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
        kernel void add(global float* values, global float* results)//, global float* constant)
        {
            size_t index = get_global_id(0);
            results[index] = values[index] + values[index + 1] + values[index + 2];// +*constant;
        }

        kernel void diffuse(global float* values, global float* constantsff)
        {
            size_t index = get_global_id(0);
            values[index] = values[index] + constantsff[0];
            if (values[index] > 255.999) {
                values[index] = values[index] - 255.999;
            }
        }

    );

    compute::program program = compute::program::build_with_source(source, ctx);
    kernelDiffuse = compute::kernel(program, "diffuse");
    auto kernelAdd = compute::kernel(program, "add");

    int width = RunConfiguration::Environment::width;
    int height = RunConfiguration::Environment::height;
    dataTrailCurrent = compute::vector<float>(width * height, ctx);
}

void SlimeMoldOpenCl::diffusion() {

    // TODO: Continue work here.
    //  1. Write a proper diffuse kernel
    //  2. Write kernels for the other steps as well
    //
    //  Keep current/next data trails as variables. Call the diffuse function with these in the proper order,
    //   based on a variable set in the swap() function. See if it's simple to switch to pointers on device?
    //  Otherwise, just keep a bool to say which is current and which is next.

    compute::vector<float> constants(1, ctx);
    constants[0] = rand() % 10;
    kernelDiffuse.set_arg(0, dataTrailCurrent.get_buffer());
    kernelDiffuse.set_arg(1, constants.get_buffer());
    queue.enqueue_1d_range_kernel(kernelDiffuse, 0, 200*200, 0);
}

void SlimeMoldOpenCl::makeRenderImage() {
    int width = RunConfiguration::Environment::width;
    int height = RunConfiguration::Environment::height;
    std::vector<float> dataTrailCurrentHost(width * height);

    compute::copy(dataTrailCurrent.begin(), dataTrailCurrent.end(), dataTrailCurrentHost.begin(), queue);

    for (int i = 0; i < dataTrailCurrentHost.size(); i++) {
        dataTrailRender[i] = static_cast<unsigned char>(dataTrailCurrentHost[i]);
    }
}