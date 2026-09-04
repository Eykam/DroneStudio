#version 450 core
#extension GL_ARB_bindless_texture : require

// Inputs from vertex shader
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in vec4 VertexColor;

// Output
out vec4 FragColor;

// Camera position
uniform vec3 viewPos;

// Material index into SSBOs
uniform uint uMaterialIndex;

// Flag for instanced rendering (no material lookup)
uniform bool uInstancedKeypoints;
uniform bool uInstancedLines;

// Flag for vertex color mode (no material, use vertex colors)
uniform bool uUseVertexColor;

// ============================================================================
// SSBO Definitions (std430 layout, must match Zig structs)
// ============================================================================

const uint TEX_BASE_COLOR = 0;
const uint TEX_NORMAL_MAP = 1;
const uint TEX_SPECULAR = 5;

const uint SLOT_COUNT = 10;

struct MaterialGPU {
    uvec2 texture_handles[SLOT_COUNT];
    uint texture_mask;
    uint material_type;
    uint data_index;
    uint flags;
};

struct PhongDataGPU {
    vec3 ambientColor;
    float shininess;
    vec4 diffuseColor;
    vec3 specularColor;
    float _pad;
};

layout(std430, binding = 0) readonly buffer MaterialsBuffer {
    MaterialGPU materials[];
};

layout(std430, binding = 2) readonly buffer PhongDataBuffer {
    PhongDataGPU phong_data[];
};

// ============================================================================
// Helper functions
// ============================================================================

bool hasTexture(uint mask, uint slot) {
    return (mask & (1u << slot)) != 0u;
}

// ============================================================================
// Main
// ============================================================================

void main() {
    // For instanced rendering or vertex color mode, just use vertex color
    if (uInstancedKeypoints || uInstancedLines || uUseVertexColor) {
        FragColor = VertexColor;
        return;
    }

    // Fetch material from SSBO
    MaterialGPU mat = materials[uMaterialIndex];
    PhongDataGPU phong = phong_data[mat.data_index];

    // Get diffuse color
    vec4 diffuse = phong.diffuseColor;

    if (hasTexture(mat.texture_mask, TEX_BASE_COLOR)) {
        sampler2D diffuseTex = sampler2D(mat.texture_handles[TEX_BASE_COLOR]);
        diffuse *= texture(diffuseTex, TexCoord);
    }

    // Get normal
    vec3 N = normalize(Normal);
    if (hasTexture(mat.texture_mask, TEX_NORMAL_MAP)) {
        // For Phong, we'd need TBN matrix which isn't passed. Use basic normal.
        // Could be extended if needed.
    }

    // Get specular
    vec3 specular = phong.specularColor;
    if (hasTexture(mat.texture_mask, TEX_SPECULAR)) {
        sampler2D specularTex = sampler2D(mat.texture_handles[TEX_SPECULAR]);
        specular *= texture(specularTex, TexCoord).rgb;
    }

    // Simple Phong lighting
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
    vec3 V = normalize(viewPos - FragPos);
    vec3 R = reflect(-lightDir, N);

    // Ambient
    vec3 ambient = phong.ambientColor * diffuse.rgb * 0.5;

    // Diffuse
    float NdotL = max(dot(N, lightDir), 0.0);
    vec3 diff = diffuse.rgb * NdotL;

    // Specular
    float spec = pow(max(dot(V, R), 0.0), phong.shininess);
    vec3 specContrib = specular * spec;

    vec3 color = ambient + diff + specContrib;

    FragColor = vec4(color, diffuse.a);
}
