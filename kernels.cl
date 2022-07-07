kernel void diffuse(global float* values, global float* constantsff)
{
    size_t idx = get_global_id(0);
    values[idx] = values[idx] + constantsff[0];
    if (values[idx] > 255.999) {
        values[idx] = values[idx] - 255.999;
    }
}

kernel void clear(global float* trailMap)
{
    size_t idx = get_global_id(0);

    trailMap[idx] = 0.0f;
}

kernel void move(global RunConfigurationCl* config, global float* trailMap, global Agent* agents)
{
    size_t idx = get_global_id(0);

    agents[idx].x += 0.1f;
    
    if(agents[idx].x >= config[0].envWidth) {
        agents[idx].x -= config[0].envWidth;
    }

    int x = agents[idx].x;
    int y = agents[idx].y;

    trailMap[x + y*config[0].envWidth] = 255.0f;
}