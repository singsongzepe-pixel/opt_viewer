#version 450

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


layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

void main() {
    // Output the color received from vertex shader
    vec3 ambientStrength = vec3(0.1);

    // ambient
    vec3 ambient = ubo.ka.rgb * ambientStrength;

    // diffuse
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(ubo.lightPos.xyz - fragPos);    
    vec3 diffuse =  max(0.0, dot(norm, lightDir)) * ubo.kdiff.rgb * ubo.lightColor.rgb;

    // hight light
    vec3 viewDir = normalize(ubo.viewPos.xyz - fragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);

    float p = ubo.kspec.a;
    vec3 specular = pow(max(0.0, dot(norm, halfwayDir)), p) * ubo.kspec.rgb * ubo.lightColor.rgb;
    
    vec3 result = ambient + diffuse + specular;
    result = result / (result + vec3(1.0));
    
    outColor = vec4(result, 1.0);
}