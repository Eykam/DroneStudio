#version 330 core
// Base attributes (used by all meshes)
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec2 aTexCoord;

// Instance attributes (only used by instanced meshes)
layout(location = 3) in vec3 aInstancePos;
layout(location = 4) in vec3 aInstanceColor;

// Uniforms
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform bool uUseInstancing;  // Flag to determine rendering mode

out vec3 vertexColor;
out vec2 texCoord;

void main()
{
    vec3 worldPos;
    vec3 finalColor;
    
    if (uUseInstancing) {
        vec4 instanceOffset = uModel * vec4(aInstancePos, 0.0);
        vec4 modelPos = uModel * vec4(aPos, 1.0);

        // Add instance offset in world space
        worldPos = modelPos.xyz + instanceOffset.xyz;
        finalColor = aInstanceColor;
        gl_Position = uProjection * uView * vec4(worldPos, 1.0);
        gl_PointSize = 5.0;
    } else {
        // Normal rendering path
        worldPos = aPos;
        finalColor = aColor;
        gl_Position = uProjection * uView * uModel * vec4(worldPos, 1.0);
    }
    
    vertexColor = finalColor;
    texCoord = aTexCoord;
}