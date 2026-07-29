#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform Push {
    ivec2 dst_origin;
    ivec2 dst_size;
    ivec2 src_origin;
    ivec2 src_size;
    ivec2 overlay_size;
    int premultiplied_alpha;
    float global_alpha;
} pc;

layout(set = 0, binding = 0, rgba8) uniform image2D mainImage;
layout(set = 0, binding = 1) uniform sampler2D overlayTex;

void main()
{
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    if (gid.x >= pc.dst_size.x || gid.y >= pc.dst_size.y)
        return;

    ivec2 dst = pc.dst_origin + gid;
    ivec2 src_pos = pc.src_origin + gid;

    vec4 src = texelFetch(overlayTex, src_pos, 0);
    vec4 dstc = imageLoad(mainImage, dst);
    float alpha = src.a;
    vec3 src_rgb = src.rgb;

    if (pc.premultiplied_alpha != 0 && alpha > 0.0)
        src_rgb = clamp(src_rgb / alpha, 0.0, 1.0);

    alpha *= pc.global_alpha;

    vec3 out_rgb = src_rgb * alpha + dstc.rgb * (1.0 - alpha);
    imageStore(mainImage, dst, vec4(out_rgb, dstc.a));
}
