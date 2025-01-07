#version 330 core

in vec3 vertexColor;
in vec2 texCoord;

out vec4 FragColor;

uniform sampler2D yTexture;
uniform sampler2D uvTexture;
uniform bool useTexture;


void main()
{   
    vec3 rgb;
       
    if (useTexture) {
        float y = texture(yTexture, texCoord).r;
        vec2 uv = texture(uvTexture, texCoord ).rg - vec2(0.5, 0.5);
                
        rgb = mat3(      1,       1,       1,
                         0, -.21482, 2.12798,
                   1.28033, -.38059,       0) * vec3(y,uv);
        // rgb = vec3(y,y,y);
    }
    else {
        rgb = vertexColor;
    }

    FragColor = vec4(rgb, 1.0);
}