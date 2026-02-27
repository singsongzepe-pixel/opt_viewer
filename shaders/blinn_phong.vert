#version 450

// Defined in binding 0
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;

    vec4 lightPos;
    vec4 lightColor;
    vec4 viewPos;

    vec4 ka;
    vec4 kdiff;
    vec4 kspec;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragPos;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);

    fragColor = inColor;
    mat3 normalMatrix = mat3(transpose(inverse(ubo.model)));
    fragNormal = normalMatrix * inNormal;
    fragPos = vec3(ubo.model * vec4(inPosition, 1.0));
}