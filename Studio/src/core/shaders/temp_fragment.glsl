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
uniform bool hasBaseColorTexture;
uniform bool hasMetallicRoughnessTexture;
uniform bool hasNormalTexture;
uniform bool hasOcclusionTexture;
uniform bool hasEmissiveTexture;

// Material uniforms - Specular-Glossiness Extension
uniform bool useSpecularGlossiness;
uniform vec4 diffuseFactor;
uniform vec3 specularFactor;
uniform float glossinessFactor;

// Material uniforms - Specular Extension
uniform bool useSpecularExtension;
uniform float specularStrength;
uniform vec3 specularColorFactor;

// Material uniforms - Other
uniform vec3 emissiveFactor;
uniform float emissiveStrength;
uniform float alphaCutoff;
uniform int alphaMode; // 0 = OPAQUE, 1 = MASK, 2 = BLEND

// Textures - now using RGB directly
uniform sampler2D baseColorTexture;  // Main RGB texture (replaces baseColorTextureY)
uniform sampler2D metallicRoughnessTexture;
uniform sampler2D normalTexture;
uniform sampler2D occlusionTexture;
uniform sampler2D emissiveTexture;
uniform sampler2D specularTexture;

// Backward compatibility with your current texturing system
uniform sampler2D yTexture;
uniform sampler2D uvTexture;
uniform sampler2D depthTexture;

// Lighting
uniform vec3 ambientColor;
uniform float ambientStrength;

// For demonstrations - typically these would be arrays or light buffers in a real engine
uniform vec3 lightPosition;  // Point light position
uniform vec3 lightColor;     // Point light color
uniform float lightIntensity; // Point light intensity

// Constants
const float PI = 3.14159265359;

// Function to calculate normal from normal map
vec3 getNormalFromMap() {
    if (!hasNormalTexture) {
        return normalize(Normal);
    }
    
    vec3 tangentNormal = texture(normalTexture, TexCoord).xyz * 2.0 - 1.0;
    return normalize(TBN * tangentNormal);
}

// PBR functions
// Distribution function - GGX/Trowbridge-Reitz
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    
    return a2 / max(denom, 0.001);
}

// Geometry function - Smith's Schlick-GGX
float geometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx1 = geometrySchlickGGX(NdotV, roughness);
    float ggx2 = geometrySchlickGGX(NdotL, roughness);
    
    return ggx1 * ggx2;
}

// Fresnel equation - Schlick approximation
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

void main() {
    // Get albedo color
    vec4 albedo;
    
    if (useTexture && hasBaseColorTexture) {
        // Use direct RGB texture sampling
        albedo = texture(baseColorTexture, TexCoord) * baseColorFactor;
    } else if (useSpecularGlossiness && hasBaseColorTexture) {
        // For specular-glossiness, the diffuse texture is used as the base color
        albedo = texture(baseColorTexture, TexCoord) * diffuseFactor;
    } else if (useTexture) {
        // Backward compatibility with old texture system
        albedo = texture(yTexture, TexCoord) * baseColorFactor;
    } else {
        // Use the base color factor or vertex color
        albedo = vec4(VertexColor, 1.0) * baseColorFactor;
    }
    
    // Apply alpha mode
    if (alphaMode == 1) { // MASK
        if (albedo.a < alphaCutoff) {
            discard;
        }
    }
    
    // Get the normal
    vec3 N = getNormalFromMap();
    
    // View direction
    vec3 V = normalize(viewPos - FragPos);
    
    // Metallic and roughness parameters
    float metallic;
    float roughness;
    
    if (useSpecularGlossiness) {
        // For specular-glossiness workflow, convert to equivalent metallic-roughness
        // This is a simple approximation - a proper conversion would be more complex
        vec3 specular = specularFactor;
        if (hasMetallicRoughnessTexture) {
            specular *= texture(metallicRoughnessTexture, TexCoord).rgb;
        }
        float specularIntensity = max(max(specular.r, specular.g), specular.b);
        
        metallic = specularIntensity;
        roughness = 1.0 - glossinessFactor;
    } else {
        // Standard metallic-roughness workflow
        metallic = metallicFactor;
        roughness = roughnessFactor;
        
        if (hasMetallicRoughnessTexture) {
            // In a metallic-roughness texture, G channel is roughness, B channel is metallic
            vec2 metallicRoughness = texture(metallicRoughnessTexture, TexCoord).bg;
            roughness *= metallicRoughness.r;
            metallic *= metallicRoughness.g;
        }
    }
    
    // Clamp roughness to avoid issues with division by zero
    roughness = max(roughness, 0.05);
    
    // Ambient occlusion
    float ao = 1.0;
    if (hasOcclusionTexture) {
        ao = texture(occlusionTexture, TexCoord).r;
    }
    
    // Emissive component
    vec3 emissive = emissiveFactor;
    if (hasEmissiveTexture) {
        emissive *= texture(emissiveTexture, TexCoord).rgb;
    }
    emissive *= emissiveStrength;
    
    // Calculate reflectance at normal incidence (F0)
    vec3 F0 = vec3(0.04); // Default value for non-metals
    if (useSpecularExtension) {
        // If using the specular extension, directly use the specular color
        F0 = specularColorFactor * specularStrength;
        if (hasMetallicRoughnessTexture) {
            F0 *= texture(specularTexture, TexCoord).rgb;
        }
    } else {
        // Standard PBR F0 calculation with metal workflow
        F0 = mix(F0, albedo.rgb, metallic);
    }
    
    // Reflectance equation
    vec3 Lo = vec3(0.0);
    
    // Calculate per-light radiance (simplified to one point light)
    vec3 L = normalize(lightPosition - FragPos);
    vec3 H = normalize(V + L);
    float distance = length(lightPosition - FragPos);
    float attenuation = 1.0 / (distance * distance);
    vec3 radiance = lightColor * lightIntensity * attenuation;
    
    // Cook-Torrance BRDF
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    
    float NDF = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    
    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.001; // Add small value to prevent divide by zero
    vec3 specularTerm = numerator / denominator;
    
    // For energy conservation, the diffuse and specular light can't
    // be above 1.0 (unless it's an emissive material)
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic; // Multiply by (1 - metallic) to ensure pure metals have no diffuse light
    
    // Add to outgoing radiance Lo
    Lo += (kD * albedo.rgb / PI + specularTerm) * radiance * NdotL;
    
    // Ambient lighting (using IBL would be better but this is simplified)
    vec3 ambient = ambientColor * ambientStrength * albedo.rgb * ao;
    
    // Final color
    vec3 color = ambient + Lo + emissive;
    
    // HDR tonemapping
    color = color / (color + vec3(1.0));
    
    // Gamma correction
    color = pow(color, vec3(1.0/2.2));
    
    // Final output with alpha
    FragColor = vec4(color, albedo.a);
}