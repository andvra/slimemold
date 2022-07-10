#include <vector>

#include "slimemold.h"
#include "utils.h"

SlimeMold::SlimeMold() {
    const int imgWidth = RunConfiguration::Environment::width;
    const int imgHeight = RunConfiguration::Environment::height;
    dataTrailRender = new unsigned char[imgWidth * imgHeight];
    random = new Utils::Random();
}

SlimeMold::~SlimeMold() {
    delete[] dataTrailRender;
    delete random;
}

std::vector<Agent> SlimeMold::initAgents() {
    std::vector<Agent> agents(RunConfiguration::Environment::populationSize(), Agent());
    auto pattern = RunConfiguration::Environment::initPattern;

    for (auto& agent : agents) {
        switch (pattern) {
        case AgentInitPattern::Random:
            agent.direction = random->randomDirection();
            agent.x = RunConfiguration::Environment::width * random->randFloat();
            agent.y = RunConfiguration::Environment::height * random->randFloat();
            break;
        case AgentInitPattern::Circle:
            float circleBorderWidth = 100.0f;
            float circleRadius = 200.0f;
            float r = circleRadius + circleBorderWidth * random->randFloat();
            agent.direction = random->randomDirection();
            agent.x = RunConfiguration::Environment::width / 2 + r * std::cos(agent.direction);
            agent.y = RunConfiguration::Environment::height / 2 + r * std::sin(agent.direction);
            break;
        }
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

std::vector<int> SlimeMold::getAgentMoveOrder() {
    std::vector<int> agentMoveOrder(RunConfiguration::Environment::populationSize(), 0);

    for (int i = 0; i < RunConfiguration::Environment::populationSize(); i++) {
        agentMoveOrder[i] = i;
    }

    // The first agent that moves into a position is the only one that can stay there.
    //  To avoid bias, randomize move order at each step.
    random->shuffleVector<int>(agentMoveOrder);

    return agentMoveOrder;
}

unsigned char* SlimeMold::getDataTrailRender() {
    return dataTrailRender;
}
