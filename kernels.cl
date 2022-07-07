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