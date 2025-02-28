#version 330 core

in vec4 vertexColor;
in vec2 texCoord;

out vec4 FragColor;

uniform sampler2D yTexture;
uniform sampler2D uvTexture;
uniform bool useTexture;


void main()
{   
    vec4 rgb;
       
    if (useTexture) {
        float y = texture(yTexture, texCoord).r;
        vec2 uv = texture(uvTexture, texCoord ).rg - vec2(0.5, 0.5);
                
        mat3 conversion = mat3(
            1.0,     1.0,     1.0,
            0.0,    -0.21482, 2.12798,
            1.28033, -0.38059, 0.0
        );
        
        vec3 conv = conversion * vec3(y, uv);
        rgb = vec4(conv, 1.0);
    }
    else {
        rgb = vertexColor;
    }

    FragColor = rgb;
}