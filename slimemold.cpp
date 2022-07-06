#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <boost/compute.hpp>
#include <opencv2/opencv.hpp>

namespace compute = boost::compute;
constexpr double PI = 3.14159265358979323846;
constexpr float deg2rad(float deg) { return PI * deg / 180.0f; }

struct RunConfiguration {
    struct Environment {
        static const unsigned int width = 200;
        static const unsigned int height = 200;
        static const unsigned int diffusionKernelSize = 3;
        static constexpr float diffusionDecay = 0.1f;
        static const unsigned int populationSize() { return width*height*populationSizeRatio; }
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

struct Agent {
    float x;
    float y;
    float direction; // Radians
};

class RunStatistics {
public:
    RunStatistics()  {
        lastUpdate = std::chrono::system_clock::now();
        numSteps = 0;
        framesSinceLastUpdate = 0;
        fps = 0.0f;
    }

    void update() {
        numSteps++;
        framesSinceLastUpdate++;
        std::chrono::duration<float> timeSinceLastUpdateS = std::chrono::system_clock::now() - lastUpdate;
        if (timeSinceLastUpdateS.count() > updateFpsIntervalS) {
            fps = framesSinceLastUpdate / timeSinceLastUpdateS.count();
            lastUpdate = std::chrono::system_clock::now();
            framesSinceLastUpdate = 0;
        }
    }

    float getFps() const {
        return fps;
    }

    int getSteps() const {
        return numSteps;
    }

    // Update this to return a proper report
    std::string getStatusString() {
        return std::to_string(numSteps);
        //return "Steps: " + std::to_string(numSteps) + " FPS: " + std::to_string(fps);
    }

private:
    int numSteps;
    const float updateFpsIntervalS = 0.5f;
    int framesSinceLastUpdate;
    std::chrono::time_point<std::chrono::system_clock> lastUpdate;
    float fps;
};

std::uniform_real_distribution<float> floatDist(0.0f, 1.0f);
std::mt19937_64 engine;

unsigned int xyToSlimeArrayIdx(float x, float y) {
    return static_cast<unsigned int>(x) + static_cast<unsigned int>(y) * RunConfiguration::Environment::width;
}

float randFloat() {
    return floatDist(engine);

}

void updateImage(float* dataTrail, unsigned char* dataTrailRender) {
    auto cols = RunConfiguration::Environment::width;
    auto rows = RunConfiguration::Environment::height;

    for (auto col = 0; col < cols; col++) {
        for (auto row = 0; row < rows; row++) {
            auto idx = xyToSlimeArrayIdx(col, row);
            dataTrailRender[idx] = static_cast<unsigned char>(dataTrail[idx]);
        }
    }
}

std::vector<Agent> initAgents() {
    std::vector<Agent> agents(RunConfiguration::Environment::populationSize(), Agent());

    for (auto& agent : agents) {
        agent.direction = 2.0f * PI * randFloat();
        agent.x = RunConfiguration::Environment::width * randFloat();
        agent.y = RunConfiguration::Environment::height * randFloat();
    }

    return agents;
}

std::vector<unsigned int> getAgentMoveOrder() {
    std::vector<unsigned int> agentMoveOrder(RunConfiguration::Environment::populationSize(), 0);

    for (auto i = 0; i < RunConfiguration::Environment::populationSize(); i++) {
        agentMoveOrder[i] = i;
    }

    // The first agent that moves into a position is the only one that can stay there.
    //  To avoid bias, randomize move order at each step.
    std::shuffle(agentMoveOrder.begin(), agentMoveOrder.end(), engine);

    return agentMoveOrder;
}

// Keep chemo levels below 256
void deposit(float* dataTrail, unsigned int x, unsigned int y) {
    auto idx = xyToSlimeArrayIdx(x, y);
    dataTrail[idx] = clamp<float>(0.0f, 255.999f, dataTrail[idx] + RunConfiguration::Agent::chemoDeposition);
}

void move(std::vector<Agent>& agents, bool* squareTaken, float* dataTrail) {
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
            deposit(dataTrail, newXSquare, newYSquare);
        }
        else {
            agent.direction = 2.0f * PI * randFloat();
        }
    }
}

float senseAtRotation(float* dataTrail, Agent& agent, float rotationOffset) {
    // TODO: Y is 0 at the top - should we invert the angle?
    unsigned int x = static_cast<unsigned int>(agent.x + RunConfiguration::Agent::sensorOffset * std::cos(agent.direction + rotationOffset));
    unsigned int y = static_cast<unsigned int>(agent.y + RunConfiguration::Agent::sensorOffset * std::sin(agent.direction + rotationOffset));

    return dataTrail[xyToSlimeArrayIdx(x, y)];
}

void sense(std::vector<Agent>& agents, float* dataTrail) {
    auto rotationAngle = RunConfiguration::Agent::rotationAngle;
    for (int i = 0; i < agents.size(); i++) {
        auto& agent = agents[i];
        auto senseLeft = senseAtRotation(dataTrail, agent, -RunConfiguration::Agent::sensorAngle);
        auto senseForward = senseAtRotation(dataTrail, agent, 0.0f);
        auto senseRight = senseAtRotation(dataTrail, agent, RunConfiguration::Agent::sensorAngle);

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

void diffusion(float* dataTrailSource, float* dataTrailDest) {
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
                            chemo += dataTrailSource[idxSrc];
                        }
                    }
                }
            }
            dataTrailDest[idxDest] = chemo / numSquares;
        }
    }
}

void decay(float* dataTrail) {
    const auto numIndices = RunConfiguration::Environment::width * RunConfiguration::Environment::height;
    const auto decay = RunConfiguration::Environment::diffusionDecay;

    for (int i = 0; i < numIndices; i++) {
        dataTrail[i] = clamp<float>(0.0f, 255.999f, dataTrail[i] - decay);
    }
}

void swap(float*& f1, float*& f2) {
    float* f3 = f1;
    f1 = f2;
    f2 = f3;
}

int main()
{
    const std::string windowId = "SomeID";

    // get the default compute device
    compute::device gpu = compute::system::default_device();

    std::cout << "Using device: " << gpu.name() << std::endl;

    // create a compute context and command queue
    compute::context ctx(gpu);
    compute::command_queue queue(ctx, gpu);

    // generate random numbers on the host
    std::vector<float> host_vector(1000000);
    std::generate(host_vector.begin(), host_vector.end(), rand);

    // create vector on the device
    compute::vector<float> device_vector(1000000, ctx);

    // copy data to the device
    compute::copy(
        host_vector.begin(), host_vector.end(), device_vector.begin(), queue
    );

    // sort data on the device
    compute::sort(
        device_vector.begin(), device_vector.end(), queue
    );

    // copy data back to the host
    compute::copy(
        device_vector.begin(), device_vector.end(), host_vector.begin(), queue
    );

    auto done = false;

    const int imgWidth = RunConfiguration::Environment::width;
    const int imgHeight = RunConfiguration::Environment::height;
    unsigned char* data = new unsigned char[imgWidth * imgHeight];
    float* dataTrailCurrent = new float[imgWidth * imgHeight]();
    float* dataTrailNext = new float[imgWidth * imgHeight]();
    unsigned char* dataTrailRender = new unsigned char[imgWidth * imgHeight];
    bool* squareTaken = new bool[imgWidth * imgHeight];
    auto img = cv::Mat(imgHeight, imgWidth, CV_8UC1, data);
    auto imgTrail = cv::Mat(imgHeight, imgWidth, CV_8UC1, dataTrailRender);

    auto agents = initAgents();
    RunStatistics stats;

    while (!done) {
        diffusion(dataTrailCurrent, dataTrailNext);

        std::swap(dataTrailCurrent, dataTrailNext);
        decay(dataTrailCurrent);

        move(agents, squareTaken, dataTrailCurrent);
        sense(agents, dataTrailCurrent);
        updateImage(dataTrailCurrent, dataTrailRender);

        cv::imshow(windowId, imgTrail);

        auto kc = cv::waitKey(10);

        if (kc == 27) {
            done = true;
        }

        stats.update();

        cv::setWindowTitle(windowId, stats.getStatusString());
    }

    delete[] data;
    delete[] dataTrailCurrent;
    delete[] dataTrailNext;
    delete[] dataTrailRender;
    delete[] squareTaken;

    return 0;
}