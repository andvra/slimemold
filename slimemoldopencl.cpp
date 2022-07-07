#include "slimemoldopencl.h"

SlimeMoldOpenCl::SlimeMoldOpenCl() : SlimeMold() {
    compute::device gpu = compute::system::default_device();

    std::cout << "Using device: " << gpu.name() << std::endl;

    ctx = compute::context(gpu);
    queue = compute::command_queue(ctx, gpu);

    loadKernels();

    int width = RunConfiguration::Environment::width;
    int height = RunConfiguration::Environment::height;
    dataTrailCurrent = compute::vector<float>(width * height, ctx);
}

void SlimeMoldOpenCl::loadKernels() {
    auto kernelSource = Utils::Files::readAllFile("kernels.cl");
    compute::program program = compute::program::build_with_source(kernelSource, ctx);
    kernelDiffuse = compute::kernel(program, "diffuse");
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
    queue.enqueue_1d_range_kernel(kernelDiffuse, 0, 200 * 200, 0);
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