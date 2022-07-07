#include "slimemoldopencl.h"

SlimeMoldOpenCl::SlimeMoldOpenCl() : SlimeMold() {
    compute::device gpu = compute::system::default_device();

    std::cout << "Using device: " << gpu.name() << std::endl;

    ctx = compute::context(gpu);
    queue = compute::command_queue(ctx, gpu);

    loadKernels();
    loadAgents();
    loadConfig();

    int width = RunConfiguration::Environment::width;
    int height = RunConfiguration::Environment::height;
    dataTrailCurrent = compute::vector<float>(width * height, ctx);
}

void addCustomTypes(std::string& source) {
    source = compute::type_definition<Agent>() + "\n" + source;
    source = compute::type_definition<RunConfigurationCl>() + "\n" + source;
}

void SlimeMoldOpenCl::loadConfig() {
    RunConfigurationCl hostConfig;
    std::vector<RunConfigurationCl> hostConfigs(1);

    hostConfigs[0] = hostConfig;

    config = compute::vector<RunConfigurationCl>(1, ctx);

    compute::copy(hostConfigs.begin(), hostConfigs.end(), config.begin(), queue);
}

void SlimeMoldOpenCl::loadAgents() {
    auto cpuAgents = initAgents();
    
    agents = compute::vector<Agent>(cpuAgents.size(), ctx);

    compute::copy(
        cpuAgents.begin(), cpuAgents.end(), agents.begin(), queue
    );
}

void SlimeMoldOpenCl::loadKernels() {
    auto kernelSource = Utils::Files::readAllFile("kernels.cl");
    
    addCustomTypes(kernelSource);

    compute::program program = compute::program::build_with_source(kernelSource, ctx);

    kernels["diffuse"] = compute::kernel(program, "diffuse");
    kernels["clear"] = compute::kernel(program, "clear");
    kernels["move"] = compute::kernel(program, "move");
}

void SlimeMoldOpenCl::diffusion() {

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

}

void SlimeMoldOpenCl::move() {
    int numPixels = RunConfiguration::Environment::width * RunConfiguration::Environment::height;
    int numAgents = RunConfiguration::Environment::populationSize();
    compute::kernel& kernelClear = kernels["clear"];

    kernelClear.set_arg(0, dataTrailCurrent.get_buffer());
    queue.enqueue_1d_range_kernel(kernelClear, 0, numPixels, 0);

    compute::kernel& kernelMove = kernels["move"];

    kernelMove.set_arg(0, config.get_buffer());
    kernelMove.set_arg(1, dataTrailCurrent.get_buffer());
    kernelMove.set_arg(2, agents.get_buffer());

    queue.enqueue_1d_range_kernel(kernelMove, 0, numAgents, 0);
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