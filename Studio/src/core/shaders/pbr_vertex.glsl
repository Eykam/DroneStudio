#version 450 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec3 aNormal;
layout(location = 4) in vec3 aTangent;
layout(location = 5) in vec3 aBitangent;

// MVP matrices
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

// Outputs to fragment shader
out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec3 VertexColor;
out mat3 TBN;

void main() {
    // Calculate vertex position in world space
    vec4 worldPosition = uModel * vec4(aPos, 1.0);
    FragPos = worldPosition.xyz;

    // Calculate position in clip space
    gl_Position = uProjection * uView * worldPosition;

    // Pass texture coordinates to fragment shader
    TexCoord = aTexCoord;

    // Pass vertex color to fragment shader
    VertexColor = aColor;

    // Transform normal to world space
    mat3 normalMatrix = transpose(inverse(mat3(uModel)));
    Normal = normalize(normalMatrix * aNormal);

    // Calculate TBN matrix for normal mapping
    vec3 T = normalize(normalMatrix * aTangent);
    vec3 B = normalize(normalMatrix * aBitangent);
    vec3 N = normalize(normalMatrix * aNormal);
    TBN = mat3(T, B, N);
}
