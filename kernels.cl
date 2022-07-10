kernel void validChemo(global RunConfigurationCl* config, float* chemo)
{
    //float maxTotalChemo = config[0].agentMaxTotalChemo;
    //clamp(*chemo, 0.0f, maxTotalChemo);
}

kernel void measureChemoAroundPosition(global RunConfigurationCl* config, float* trailMap, int x, int y, int kernelSize, float* totalChemo, int* numMeasuredSquares) 
{
    int windowWidth = config[0].envWidth;
    int windowHeight = config[0].envHeight;

    *totalChemo = 0.0f;
    *numMeasuredSquares = 0;

    for (int xd = x - kernelSize / 2; xd <= x + kernelSize / 2; xd++) {
        if (xd >= 0 && xd < windowWidth) {
            for (int yd = y - kernelSize / 2; yd <= y + kernelSize / 2; yd++) {
                if (yd >= 0 && yd < windowHeight) {
                    int idxSrc = xd + windowWidth * yd;
                    *numMeasuredSquares = *numMeasuredSquares + 1;
                    *totalChemo += trailMap[idxSrc];
                }
            }
        }
    }
}

kernel void diffuse(global RunConfigurationCl* config, global float* trailMapSource, global float* trailMapDestination)
{
    size_t col = get_global_id(0);
    size_t row = get_global_id(1);
    int windowWidth = config[0].envWidth;
    int windowHeight = config[0].envHeight;
    int kernelSize = config[0].envDiffusionKernelSize;
    size_t idxDest = col + row * windowWidth;
    float diffuseRate = 0.2f;

    float chemo = 0.0f;
    int numSquares = 0;

    measureChemoAroundPosition(config, trailMapSource, col, row, kernelSize, &chemo, &numSquares);

    //  Why does this look "better" if we use numSquares+1 instead of numSquares?

    // TODO: Check with the paper. Is there diffuseRate vs decayRate, are they both there?

    float blurredVal =  chemo / (kernelSize * kernelSize+ 1);
    float newVal = diffuseRate * blurredVal + (1 - diffuseRate) * trailMapSource[idxDest];

    trailMapDestination[idxDest] = newVal;

}

kernel void decay(global RunConfigurationCl* config, global float* trailMap)
{
    size_t idx = get_global_id(0);
    float decay = config[0].envDiffusionDecay;

    float chemo = trailMap[idx] - decay;
    validChemo(config, &chemo);
    trailMap[idx] = chemo;
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

    if(desiredDestinationIdx == -1) {
        // -1 means we could not move. Update agent direction to new random direction, sent from host
        agents[idx].direction = newDirections[idx];
    }
    else {
        // We can move! We've already calculated the move, so just copy it.
        agents[idx].x = agentsNewPos[idx].x;
        agents[idx].y = agentsNewPos[idx].y;
        int x = agents[idx].x;
        int y = agents[idx].y;
        int width = config[0].envWidth;
        int trailIdx = x + width * y;
        float chemo = trailMap[desiredDestinationIdx] + (float)chemoDeposition;
        validChemo(config, &chemo);
        trailMap[desiredDestinationIdx] = chemo;
    }
}

kernel void senseAtRotation(global RunConfigurationCl* config, global float* trailMap, global Agent* agents, int agentIdx, float rotationOffset, float* res)
{
    int sensorOffset = config[0].agentSensorOffset;
    int width = config[0].envWidth;
    int height = config[0].envHeight;
    int sensorWidth = config[0].agentSensorWidth;

    int x = agents[agentIdx].x + sensorOffset * cos(agents[agentIdx].direction + rotationOffset);
    int y = agents[agentIdx].y + sensorOffset * sin(agents[agentIdx].direction + rotationOffset);

    float chemo;
    int numSquares;

    measureChemoAroundPosition(config, trailMap, x, y, sensorWidth, &chemo, &numSquares);

    // Check if position is valid
    int idx = x + y * width;
    *res = chemo;
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