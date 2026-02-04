#version 460 core
#extension GL_ARB_bindless_texture : require

// Inputs from vertex shader
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in vec3 VertexColor;
in mat3 TBN;

// Output
out vec4 FragColor;

// Camera position
uniform vec3 viewPos;

// Material index into SSBOs
uniform uint uMaterialIndex;

// ============================================================================
// SSBO Definitions (std430 layout, must match Zig structs)
// ============================================================================

const uint TEX_BASE_COLOR = 0;
const uint TEX_NORMAL_MAP = 1;
const uint TEX_METALLIC_ROUGHNESS = 2;
const uint TEX_OCCLUSION = 3;
const uint TEX_EMISSIVE = 4;
const uint TEX_SPECULAR = 5;
const uint SLOT_COUNT = 10;

struct MaterialGPU {
    uvec2 texture_handles[SLOT_COUNT];
    uint texture_mask;
    uint material_type;
    uint data_index;
    uint flags;
};

struct PBRDataGPU {
    vec4 baseColorFactor;
    vec3 emissiveFactor;
    float emissiveStrength;
    vec3 specularColor;
    float specularStrength;
    float metallicFactor;
    float roughnessFactor;
    float alphaCutoff;
    float _pad;
};

layout(std430, binding = 0) readonly buffer MaterialsBuffer {
    MaterialGPU materials[];
};

layout(std430, binding = 1) readonly buffer PBRDataBuffer {
    PBRDataGPU pbr_data[];
};

// ============================================================================
// PBR Functions
// ============================================================================

const float PI = 3.14159265359;

// Normal Distribution Function (GGX/Trowbridge-Reitz)
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return a2 / max(denom, 0.0001);
}

// Geometry Function (Schlick-GGX)
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

// Fresnel (Schlick approximation)
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

bool hasTexture(uint mask, uint slot) {
    return (mask & (1u << slot)) != 0u;
}

// ============================================================================
// Main
// ============================================================================

void main() {
    MaterialGPU mat = materials[uMaterialIndex];
    PBRDataGPU pbr = pbr_data[mat.data_index];

    // Albedo
    vec4 albedo = pbr.baseColorFactor;
    if (hasTexture(mat.texture_mask, TEX_BASE_COLOR)) {
        sampler2D tex = sampler2D(mat.texture_handles[TEX_BASE_COLOR]);
        albedo *= texture(tex, TexCoord);
    }

    // Alpha handling
    uint alphaMode = mat.flags & 0x3u;
    if (alphaMode == 1u && albedo.a < pbr.alphaCutoff) {
        discard;
    }

    // Metallic/Roughness
    float metallic = pbr.metallicFactor;
    float roughness = pbr.roughnessFactor;
    if (hasTexture(mat.texture_mask, TEX_METALLIC_ROUGHNESS)) {
        sampler2D tex = sampler2D(mat.texture_handles[TEX_METALLIC_ROUGHNESS]);
        vec4 mr = texture(tex, TexCoord);
        metallic *= mr.b;
        roughness *= mr.g;
    }
    roughness = clamp(roughness, 0.04, 1.0);

    // Normal
    vec3 N = normalize(Normal);
    if (hasTexture(mat.texture_mask, TEX_NORMAL_MAP)) {
        sampler2D tex = sampler2D(mat.texture_handles[TEX_NORMAL_MAP]);
        vec3 normalMap = texture(tex, TexCoord).rgb * 2.0 - 1.0;
        N = normalize(TBN * normalMap);
    }

    // Occlusion
    float ao = 1.0;
    if (hasTexture(mat.texture_mask, TEX_OCCLUSION)) {
        sampler2D tex = sampler2D(mat.texture_handles[TEX_OCCLUSION]);
        ao = texture(tex, TexCoord).r;
    }

    // Emissive
    vec3 emissive = pbr.emissiveFactor * pbr.emissiveStrength;
    if (hasTexture(mat.texture_mask, TEX_EMISSIVE)) {
        sampler2D tex = sampler2D(mat.texture_handles[TEX_EMISSIVE]);
        emissive *= texture(tex, TexCoord).rgb;
    }

    // View direction
    vec3 V = normalize(viewPos - FragPos);

    // F0 - surface reflection at zero incidence
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo.rgb, metallic);

    // Lighting
    vec3 Lo = vec3(0.0);

    // Single directional light (sun-like)
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
    vec3 lightColor = vec3(1.0, 0.98, 0.95) * 2.5;

    vec3 L = lightDir;
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);

    // Cook-Torrance BRDF
    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
    vec3 specular = numerator / denominator;

    // Energy conservation
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic; // Metals have no diffuse

    Lo += (kD * albedo.rgb / PI + specular) * lightColor * NdotL;

    // Ambient (simple IBL approximation)
    vec3 ambient = vec3(0.5) * albedo.rgb * ao;

    vec3 color = ambient + Lo + emissive;

    // Tone mapping (Reinhard)
    color = color / (color + vec3(1.0));

    FragColor = vec4(color, albedo.a);
}
