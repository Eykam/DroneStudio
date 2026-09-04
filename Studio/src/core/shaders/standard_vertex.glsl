#version 450 core

// Base attributes
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec3 aNormal;

// Instance attributes (for instanced rendering)
layout(location = 4) in vec4 aInstancePos;
layout(location = 5) in vec4 aInstanceEnd;
layout(location = 6) in vec4 aInstanceColor;

// Uniforms
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform bool uInstancedKeypoints;
uniform bool uInstancedLines;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec4 VertexColor;

void main() {
    vec3 worldPos;
    vec4 finalColor;

    if (uInstancedKeypoints) {
        vec4 instanceOffset = uModel * aInstancePos;
        vec4 modelPos = uModel * vec4(aPos, 1.0);
        worldPos = modelPos.xyz + instanceOffset.xyz;
        finalColor = aInstanceColor;
        gl_Position = uProjection * uView * vec4(worldPos, 1.0);
        gl_PointSize = 6.0;
        Normal = vec3(0.0, 1.0, 0.0);
    }
    else if (uInstancedLines) {
        vec4 instanceOffset;
        if (gl_VertexID == 0) {
            instanceOffset = aInstancePos;
            worldPos = instanceOffset.xyz;
        } else {
            instanceOffset = uModel * aInstanceEnd;
            vec4 modelPos = uModel * vec4(aPos, 1.0);
            worldPos = modelPos.xyz + instanceOffset.xyz;
        }
        finalColor = aInstanceColor;
        gl_Position = uProjection * uView * vec4(worldPos, 1.0);
        Normal = vec3(0.0, 1.0, 0.0);
    }
    else {
        // Normal rendering path
        vec4 worldPosition = uModel * vec4(aPos, 1.0);
        worldPos = worldPosition.xyz;
        finalColor = vec4(aColor, 1.0);
        gl_Position = uProjection * uView * worldPosition;

        // Transform normal
        mat3 normalMatrix = transpose(inverse(mat3(uModel)));
        Normal = normalize(normalMatrix * aNormal);
    }

    FragPos = worldPos;
    VertexColor = finalColor;
    TexCoord = aTexCoord;
}
