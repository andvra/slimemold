kernel void diffuse(global RunConfigurationCl* config, global float* trailMap)
{
    // TODO: Only used for clearing the image right now
    size_t idx = get_global_id(0);

    trailMap[idx] = clamp(trailMap[idx] - 1.0f, 0.0f, 255.0f);

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
        trailMap[desiredDestinationIdx] = clamp(desiredChemo, 0.0f, 255.0f);
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