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
}

void SlimeMoldOpenCl::loadHostMemory() {
    int numPixels = RunConfiguration::Environment::numPixels();
    int numAgents = RunConfiguration::Environment::populationSize();

    hTakenMap = std::vector<unsigned char>(numPixels);
    hDesiredDestinationIdx = std::vector<int>(numAgents);
    hNewDirection = std::vector<float>(numAgents);
}

void SlimeMoldOpenCl::loadDeviceMemory() {
    int numPixels = RunConfiguration::Environment::numPixels();
    int numAgents = RunConfiguration::Environment::populationSize();

    dDataTrailCurrent = compute::vector<float>(numPixels, ctx);
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
    // TODO: Continue work here.
    //  1. Write a proper diffuse kernel
    //  2. Write kernels for the other steps as well
    //
    //  Keep current/next data trails as variables. Call the diffuse function with these in the proper order,
    //   based on a variable set in the swap() function. See if it's simple to switch to pointers on device?
    //  Otherwise, just keep a bool to say which is current and which is next.

    //compute::kernel& kernelDiffuse = kernels["diffuse"];

    //compute::vector<float> constants(1, ctx);
    //constants[0] = rand() % 10;
    //kernelDiffuse.set_arg(0, dataTrailCurrent.get_buffer());
    //kernelDiffuse.set_arg(1, constants.get_buffer());
    //queue.enqueue_1d_range_kernel(kernelDiffuse, 0, 200 * 200, 0);

    compute::kernel& kernelDiffuse = kernels["diffuse"];
    kernelDiffuse.set_arg(0, dConfig.get_buffer());
    kernelDiffuse.set_arg(1, dDataTrailCurrent.get_buffer());
    queue.enqueue_1d_range_kernel(kernelDiffuse, 0, numPixels, 0);
}

void SlimeMoldOpenCl::move() {

    int numPixels = RunConfiguration::Environment::numPixels();
    int numAgents = RunConfiguration::Environment::populationSize();
    auto agentMoveOrder = getAgentMoveOrder();
    compute::kernel& kernelDesiredMove = kernels["desiredMoves"];
    compute::kernel& kernelMove = kernels["move"];
    compute::kernel& kernelTempRender = kernels["temprender"];
    //  1. (GPU) Calculate desired indices. This functions takes a result vector if size numAgents, where the value is set to its desired move idx.
    //      Keep a buffer on the gpu with desired newx/newy as float values. This will make it more efficient, not rounding all values on CPU
    //  2. (CPU) Create two lists of size numAgents: canMove and newAngle. Copy to GPU
    //  3. (GPU) If canMove = false for agentIdx, update angle to value in newAngle

    // Calculate desired next position of all agents

    kernelDesiredMove.set_arg(0, dConfig.get_buffer());
    kernelDesiredMove.set_arg(1, dAgents.get_buffer());
    kernelDesiredMove.set_arg(2, dAgentDesired.get_buffer());
    kernelDesiredMove.set_arg(3, dDesiredDestinationIndices.get_buffer());
    queue.enqueue_1d_range_kernel(kernelDesiredMove, 0, numAgents, 0);

    // Read desired position of all agents
    std::vector<int> hDesiredDestinationIndices(numAgents);
    compute::copy(dDesiredDestinationIndices.begin(), dDesiredDestinationIndices.end(), hDesiredDestinationIndices.begin(), queue);

    std::fill(hTakenMap.begin(), hTakenMap.end(), 0);

    for (int idx = 0; idx < numAgents; idx++) {
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

    kernelMove.set_arg(0, dConfig.get_buffer());
    kernelMove.set_arg(1, dDataTrailCurrent.get_buffer());
    kernelMove.set_arg(2, dAgents.get_buffer());
    kernelMove.set_arg(3, dAgentDesired.get_buffer());
    kernelMove.set_arg(4, dDesiredDestinationIndices.get_buffer());
    kernelMove.set_arg(5, dNewDirection.get_buffer());

    queue.enqueue_1d_range_kernel(kernelMove, 0, numAgents, 0);
}

void SlimeMoldOpenCl::makeRenderImage() {
    int numPixels = RunConfiguration::Environment::numPixels();
    std::vector<float> dataTrailCurrentHost(numPixels);

    compute::copy(dDataTrailCurrent.begin(), dDataTrailCurrent.end(), dataTrailCurrentHost.begin(), queue);

    for (int i = 0; i < dataTrailCurrentHost.size(); i++) {
        dataTrailRender[i] = static_cast<unsigned char>(dataTrailCurrentHost[i]);
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
    kernelSense.set_arg(1, dDataTrailCurrent.get_buffer());
    kernelSense.set_arg(2, dAgents.get_buffer());
    kernelSense.set_arg(3, dRandomValues.get_buffer());

    queue.enqueue_1d_range_kernel(kernelSense, 0, numAgents, 0);
}