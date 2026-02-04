#version 330 core

in vec2 texCoord;
out vec4 FragColor;
uniform sampler2D viewTexture; // The rendered FBO texture

void main()
{
    FragColor = texture(viewTexture, texCoord);
}