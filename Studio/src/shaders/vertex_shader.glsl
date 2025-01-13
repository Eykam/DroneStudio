#version 330 core
// Base attributes (used by all meshes)
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec2 aTexCoord;

// Instance attributes (only used by instanced meshes)
layout(location = 3) in vec4 aInstancePos;
layout(location = 4) in vec4 aInstanceEnd;
layout(location = 5) in vec4 aInstanceColor;

// Uniforms
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform bool uInstancedKeypoints;
uniform bool uInstancedLines;

out vec4 vertexColor;
out vec2 texCoord;

void main()
{
    vec3 worldPos;
    vec4 finalColor;
    
    if (uInstancedKeypoints) {
        vec4 instanceOffset = uModel * aInstancePos;
        vec4 modelPos = uModel * vec4(aPos, 1.0);

        // Add instance offset in world space
        worldPos = modelPos.xyz + instanceOffset.xyz;
        finalColor = aInstanceColor;
        gl_Position = uProjection * uView * vec4(worldPos, 1.0);
        gl_PointSize = 15.0;
    } 
    else if(uInstancedLines){
        vec4 instanceOffset = gl_VertexID == 0 ? aInstancePos : uModel * aInstanceEnd;
        vec4 modelPos = uModel * vec4(aPos, 1.0);

        // Add instance offset in world space
        worldPos =  modelPos.xyz + instanceOffset.xyz;
        finalColor = aInstanceColor;
        gl_Position = uProjection * uView * vec4(worldPos, 1.0);
    }
    else {
        // Normal rendering path
        worldPos = aPos;
        finalColor = vec4(aColor, 1.0);
        gl_Position = uProjection * uView * uModel * vec4(worldPos, 1.0);
    }
    
    vertexColor = finalColor;
    texCoord = aTexCoord;
}