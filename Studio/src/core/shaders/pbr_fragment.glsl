#version 330 core

// Inputs from vertex shader
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in vec3 VertexColor;
in mat3 TBN;

// Output
out vec4 FragColor;

// Camera position (for reflection calculations)
uniform vec3 viewPos;

// Material uniforms - PBR Metallic-Roughness
uniform bool useTexture;
uniform vec4 baseColorFactor;
uniform float metallicFactor;
uniform float roughnessFactor;
uniform vec3 emissiveFactor;
uniform float emissiveStrength;
uniform float alphaCutoff;
uniform int alphaModeEnum;
uniform bool doubleSided;
uniform vec3 specularColor;
uniform float specularStrength;

// Texture presence flags
uniform bool hasBaseColorTexture;
uniform bool hasNormalTexture;
uniform bool hasMetallicRoughnessTexture;
uniform bool hasOcclusionTexture;
uniform bool hasEmissiveTexture;
uniform bool hasSpecularTexture;

// Textures
uniform sampler2D baseColorTexture; 

// Lighting
uniform vec3 ambientColor;
uniform float ambientStrength;

// For demonstrations - typically these would be arrays or light buffers in a real engine
uniform vec3 lightPosition;  // Point light position
uniform vec3 lightColor;     // Point light color
uniform float lightIntensity; // Point light intensity



void main() {
    // Get albedo color
    vec4 albedo;
    
    if (useTexture && hasBaseColorTexture) {
        // Use texture if available
        albedo = texture(baseColorTexture, TexCoord) * baseColorFactor;
    } else {
        // Use the base color factor from material
        albedo = baseColorFactor;
    }
    
    FragColor = albedo;
}
