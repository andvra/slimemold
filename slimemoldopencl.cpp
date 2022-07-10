#include "slimemoldopencl.h"

SlimeMoldOpenCl::SlimeMoldOpenCl() : SlimeMold() {
    compute::device gpu = compute::system::default_device();

    std::cout << "Using device: " << gpu.name() << std::endl;

    ctx = compute::context(gpu);
    queue = compute::command_queue(ctx, gpu);

    loadKernels();
    loadConfig();
    loadVariables();
}

void addCustomTypes(std::string& source) {
    source = compute::type_definition<Agent>() + "\n" + source;
    source = compute::type_definition<RunConfigurationCl>() + "\n" + source;
}

void SlimeMoldOpenCl::loadVariables() {
    loadDeviceMemory();
    loadHostMemory();

    idxDataTrailInUse = 0;
    idxDataTrailBuffer = 1;
}

void SlimeMoldOpenCl::loadHostMemory() {
    int numPixels = RunConfiguration::Environment::numPixels();
    int numAgents = RunConfiguration::Environment::populationSize();

    hTakenMap = std::vector<unsigned char>(numPixels);
    hDesiredDestinationIdx = std::vector<int>(numAgents);
    hNewDirection = std::vector<float>(numAgents);
}

void SlimeMoldOpenCl::loadDeviceMemoryTrailMaps() {
    int numPixels = RunConfiguration::Environment::numPixels();

    for (int i = 0; i < 2; i++) {
        dDataTrails.push_back(compute::vector<float>(numPixels, ctx));
        int idx = dDataTrails.size() - 1;
        compute::fill(dDataTrails[idx].begin(), dDataTrails[idx].end(), 0.0f, queue);
    }
}

void SlimeMoldOpenCl::loadDeviceMemory() {
    int numPixels = RunConfiguration::Environment::numPixels();
    int numAgents = RunConfiguration::Environment::populationSize();

    loadDeviceMemoryTrailMaps();
    
    dDesiredDestinationIndices = compute::vector<int>(numAgents, ctx);
    dNewDirection = compute::vector<float>(numAgents, ctx);
    dAgentDesired = compute::vector<Agent>(numAgents, ctx);
    dRandomValues = compute::vector<float>(numAgents, ctx);

    loadAgents();
}
void SlimeMoldOpenCl::loadConfig() {
    RunConfigurationCl hostConfig;
    std::vector<RunConfigurationCl> hostConfigs(1);

    hostConfigs[0] = hostConfig;

    dConfig = compute::vector<RunConfigurationCl>(1, ctx);

    compute::copy(hostConfigs.begin(), hostConfigs.end(), dConfig.begin(), queue);
}

void SlimeMoldOpenCl::loadAgents() {
    auto cpuAgents = initAgents();
    
    dAgents = compute::vector<Agent>(cpuAgents.size(), ctx);

    compute::copy(cpuAgents.begin(), cpuAgents.end(), dAgents.begin(), queue);
}

void SlimeMoldOpenCl::loadKernels() {
    auto kernelSource = Utils::Files::readAllFile("kernels.cl");
    
    addCustomTypes(kernelSource);

    compute::program program = compute::program::build_with_source(kernelSource, ctx);

    std::vector<std::string> kernelNames = {
        "diffuse",
        "decay",
        "move",
        "desiredMoves",
        "sense"
    };

    for (auto& kernelName : kernelNames) {
        kernels[kernelName] = compute::kernel(program, kernelName);
    }
}

void SlimeMoldOpenCl::diffusion() {

    int numPixels = RunConfiguration::Environment::numPixels();
    compute::kernel& kernelDiffuse = kernels["diffuse"];

    kernelDiffuse.set_arg(0, dConfig.get_buffer());
    kernelDiffuse.set_arg(1, dDataTrails[idxDataTrailInUse].get_buffer());
    kernelDiffuse.set_arg(2, dDataTrails[idxDataTrailBuffer].get_buffer());
    queue.enqueue_1d_range_kernel(kernelDiffuse, 0, numPixels, 0);
    const compute::extents<3> e = { 1,1,1 };
    size_t globalWorkSize[] = { RunConfiguration::Environment::width, RunConfiguration::Environment::height };

    queue.enqueue_nd_range_kernel(kernelDiffuse, 2, nullptr, &globalWorkSize[0], nullptr);

}

void SlimeMoldOpenCl::decay() {
    int numPixels = RunConfiguration::Environment::numPixels();
    compute::kernel& kernelDecay = kernels["decay"];

    kernelDecay.set_arg(0, dConfig.get_buffer());
    kernelDecay.set_arg(1, dDataTrails[idxDataTrailInUse].get_buffer());
    queue.enqueue_1d_range_kernel(kernelDecay, 0, numPixels, 0);
}

void SlimeMoldOpenCl::moveCoordinate() {
    int numAgents = RunConfiguration::Environment::populationSize();
    auto agentMoveOrder = getAgentMoveOrder();

    // Read desired position of all agents
    std::vector<int> hDesiredDestinationIndices(numAgents);
    compute::copy(dDesiredDestinationIndices.begin(), dDesiredDestinationIndices.end(), hDesiredDestinationIndices.begin(), queue);

    std::fill(hTakenMap.begin(), hTakenMap.end(), 0);

    for (int agentIdx = 0; agentIdx < numAgents; agentIdx++) {
        // We want to randomize the order in which we move agents. An agent will be blocked from moving to a square 
        //  if another agent is there already, so randomizing the order is used to avoid bias.
        auto idx = agentMoveOrder[agentIdx];
        int desiredPos = hDesiredDestinationIndices[idx];
        bool isOutOfBounds = (desiredPos == -1);
        if (isOutOfBounds) {
            hNewDirection[idx] = random->randomDirection();
        }
        else if (hTakenMap[desiredPos] == 0) {
            hTakenMap[desiredPos] = 1;
        }
        else {
            // Position already taken
            hNewDirection[idx] = random->randomDirection();
            // Tell the GPU that the agent can not move
            hDesiredDestinationIndices[idx] = -1;
        }
    }

    compute::copy(hNewDirection.begin(), hNewDirection.end(), dNewDirection.begin(), queue);
    compute::copy(hDesiredDestinationIndices.begin(), hDesiredDestinationIndices.end(), dDesiredDestinationIndices.begin(), queue);
}

void SlimeMoldOpenCl::moveDesiredMoves() {
    int numAgents = RunConfiguration::Environment::populationSize();
    compute::kernel& kernelDesiredMove = kernels["desiredMoves"];

    // Calculate desired next position of all agents

    kernelDesiredMove.set_arg(0, dConfig.get_buffer());
    kernelDesiredMove.set_arg(1, dAgents.get_buffer());
    kernelDesiredMove.set_arg(2, dAgentDesired.get_buffer());
    kernelDesiredMove.set_arg(3, dDesiredDestinationIndices.get_buffer());
    queue.enqueue_1d_range_kernel(kernelDesiredMove, 0, numAgents, 0);
}

void SlimeMoldOpenCl::moveActualMove() {
    int numAgents = RunConfiguration::Environment::populationSize();
    compute::kernel& kernelMove = kernels["move"];
    
    kernelMove.set_arg(0, dConfig.get_buffer());
    kernelMove.set_arg(1, dDataTrails[idxDataTrailInUse].get_buffer());
    kernelMove.set_arg(2, dAgents.get_buffer());
    kernelMove.set_arg(3, dAgentDesired.get_buffer());
    kernelMove.set_arg(4, dDesiredDestinationIndices.get_buffer());
    kernelMove.set_arg(5, dNewDirection.get_buffer());

    queue.enqueue_1d_range_kernel(kernelMove, 0, numAgents, 0);
}

void SlimeMoldOpenCl::move() {
    //  1. (GPU) Calculate desired indices. This functions takes a result vector of size numAgents, where the value is set to its desired move idx.
    //      Keep a buffer on the gpu with desired newx/newy as float values. This will make it more efficient, not rounding all values on CPU
    //  2. (CPU) Coordinate movements on CPU: it's made on CPU so we can get good random numbers
    //  3. (GPU) Make actual movement, based on coordination made on CPU

    moveDesiredMoves();
    moveCoordinate();
    moveActualMove();
}

void SlimeMoldOpenCl::makeRenderImage() {
    int numPixels = RunConfiguration::Environment::numPixels();
    std::vector<float> dataTrailCurrentHost(numPixels);

    compute::copy(dDataTrails[idxDataTrailInUse].begin(), dDataTrails[idxDataTrailInUse].end(), dataTrailCurrentHost.begin(), queue);

    for (int i = 0; i < dataTrailCurrentHost.size(); i++) {
        dataTrailRender[i] = static_cast<unsigned char>(Utils::Math::clamp<float>(0.0f, 255.0f, dataTrailCurrentHost[i]));
    }
}

void SlimeMoldOpenCl::sense() {
    int numAgents = RunConfiguration::Environment::populationSize();
    compute::kernel& kernelSense = kernels["sense"];
    std::vector<float> hRandomValues(numAgents);

    for (auto& f : hRandomValues) {
        f = random->randFloat();
    }

    compute::copy(hRandomValues.begin(), hRandomValues.end(), dRandomValues.begin(), queue);

    kernelSense.set_arg(0, dConfig.get_buffer());
    kernelSense.set_arg(1, dDataTrails[idxDataTrailInUse].get_buffer());
    kernelSense.set_arg(2, dAgents.get_buffer());
    kernelSense.set_arg(3, dRandomValues.get_buffer());

    queue.enqueue_1d_range_kernel(kernelSense, 0, numAgents, 0);
}

void SlimeMoldOpenCl::swapBuffers() {
    std::swap(idxDataTrailBuffer, idxDataTrailInUse);
}