kernel void diffuse(global RunConfigurationCl* config, global float* trailMapSource, global float* trailMapDestination)
{
    size_t col = get_global_id(0);
    size_t row = get_global_id(1);
    int windowWidth = config[0].envWidth;
    int windowHeight = config[0].envHeight;
    int kernelSize = config[0].envDiffusionKernelSize;
    size_t idxDest = col + row * windowWidth;

    float chemo = 0.0f;
    int numSquares = 0;

    for (int xd = col - kernelSize / 2; xd <= col + kernelSize / 2; xd++) {
        if (xd >= 0 && xd < windowWidth) {
            for (int yd = row - kernelSize / 2; yd <= row + kernelSize / 2; yd++) {
                if (yd >= 0 && yd < windowHeight) {
                    int idxSrc = xd + windowWidth * yd;
                    numSquares++;
                    chemo += trailMapSource[idxSrc];
                }
            }
        }
    }

    //  Why does this look "better" if we use numSquares+1 instead of numSquares?
    trailMapDestination[idxDest] = chemo / (numSquares+1);

}

kernel void decay(global RunConfigurationCl* config, global float* trailMap)
{
    size_t idx = get_global_id(0);
    float decay = config[0].envDiffusionDecay;

    trailMap[idx] = clamp(trailMap[idx] - decay, 0.0f, 255.999f);
}

kernel void desiredMoves(global RunConfigurationCl* config, global Agent* agents, global Agent* agentsNewPos, global int* desiredDestinationIndices)
{
    size_t idx = get_global_id(0);
    int width = config[0].envWidth;
    int height = config[0].envHeight;
    int stepSize = config[0].agentStepSize;

    // Calculate new desired position
    Agent agent = agents[idx];
    float newX = agent.x + cos(agent.direction) * stepSize;
    float newY = agent.y + sin(agent.direction) * stepSize;
    int newXSquare = newX;
    int newYSquare = newY;

    int desiredDestinationIdx = newXSquare + newYSquare * width;

    if(newXSquare>=0 && newXSquare<width && newYSquare>=0 && newYSquare<height) {
        // Store new agent position in a buffer object.
        agentsNewPos[idx].x = newX;
        agentsNewPos[idx].y = newY;
        desiredDestinationIndices[idx] = desiredDestinationIdx;
    }
    else {
        desiredDestinationIndices[idx] = -1;
    }

}

kernel void move(global RunConfigurationCl* config, global float* trailMap, global Agent* agents, global Agent* agentsNewPos, global int* desiredDestinationIndices, global float* newDirections)
{
    size_t idx = get_global_id(0);
    int chemoDeposition = config[0].agentChemoDeposition;

    int desiredDestinationIdx = desiredDestinationIndices[idx];

    // -1 means we could not move
    if(desiredDestinationIdx == -1) {
        agents[idx].direction = newDirections[idx];
    }
    else {
        agents[idx].x = agentsNewPos[idx].x;
        agents[idx].y = agentsNewPos[idx].y;
        int x = agents[idx].x;
        int y = agents[idx].y;
        int width = config[0].envWidth;
        int trailIdx = x + width * y;
        float desiredChemo = trailMap[desiredDestinationIdx] + (float)chemoDeposition;
        trailMap[desiredDestinationIdx] = clamp(desiredChemo, 0.0f, 255.999f);
    }
}

kernel void senseAtRotation(global RunConfigurationCl* config, global float* trailMap, global Agent* agents, int agentIdx, float rotationOffset, float* res)
{
    int sensorOffset = config[0].agentSensorOffset;
    int width = config[0].envWidth;
    int height = config[0].envHeight;

    int x = agents[agentIdx].x + sensorOffset * cos(agents[agentIdx].direction + rotationOffset);
    int y = agents[agentIdx].y + sensorOffset * sin(agents[agentIdx].direction + rotationOffset);

    // Check if position is valid
    if(x>= 0 && x<width && y>=0 && y<height) {
        int idx = x + y * width;
        *res = trailMap[idx];
    }
    else {
        *res = 0.0f;
    }
}

kernel void sense(global RunConfigurationCl* config, global float* trailMap, global Agent* agents, global float* randomValues)
{
    size_t idx = get_global_id(0);
    float sensorAngle = config[0].agentSensorAngle;
    float senseLeft, senseRight, senseForward;
    float rotationAngle = config[0].agentRotationAngle;
    
    senseAtRotation(config, trailMap, agents, idx, -sensorAngle, &senseLeft);
    senseAtRotation(config, trailMap, agents, idx, sensorAngle, &senseRight);
    senseAtRotation(config, trailMap, agents, idx, 0, &senseForward);

    if (senseForward > senseLeft && senseForward > senseRight) {
        // Do nothing
    }
    else if (senseForward < senseLeft && senseForward < senseRight) {
        // Rotate in random direction
        agents[idx].direction += (randomValues[idx] > 0.5) ? -rotationAngle : rotationAngle;
    }
    else if (senseLeft < senseRight) {
        agents[idx].direction += rotationAngle;
    }
    else {
        agents[idx].direction -= rotationAngle;
    }
}