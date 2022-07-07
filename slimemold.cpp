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

    for (auto& agent : agents) {
        agent.direction = random->randomDirection();
        agent.x = RunConfiguration::Environment::width * random->randFloat();
        agent.y = RunConfiguration::Environment::height * random->randFloat();
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
    random->shuffleVector<unsigned int>(agentMoveOrder);

    return agentMoveOrder;
}

unsigned char* SlimeMold::getDataTrailRender() {
    return dataTrailRender;
}
