Based on  [this](https://uwe-repository.worktribe.com/output/980579) paper


Tech: opencl, boost-compute, opencv (rendering)


## OpenCL

To add a custom type:

1. Add aaa in the header. This makes sure to pad the structure so it can be used with OpenCL
1. Add the type definition to the top of the kernel source code: compute::type_definition<NameOfCustomType>() + "\n" + <READ_CODE>;
1. Build the kernels as normal