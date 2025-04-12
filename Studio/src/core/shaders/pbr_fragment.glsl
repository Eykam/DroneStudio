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


// Textures
uniform sampler2D baseColorTexture;  // Y component


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



void main() {
    // Get albedo color
    vec4 albedo;
    
    if (useTexture) {
        // Fall back to your original YUV texture approach
        albedo = texture(baseColorTexture, TexCoord);
    } else {
        // Use the base color factor or vertex color
        albedo = vec4(0.1, 0.1, 0.1, 0.3);
    }
    
    FragColor = albedo;
}