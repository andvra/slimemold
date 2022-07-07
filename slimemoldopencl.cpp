#include "slimemoldopencl.h"

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