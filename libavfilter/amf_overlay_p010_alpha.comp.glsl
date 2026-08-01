#version 450
#extension GL_EXT_shader_image_load_formatted : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform Push {
    ivec2 dst_y_origin;
    ivec2 y_size;
    ivec2 src_y_origin;
    ivec2 dst_uv_origin;
    ivec2 uv_size;
    ivec2 src_uv_origin;
    float global_alpha;
    int debug_mode;
    int pad0;
    int pad1;
} pc;

layout(set = 0, binding = 0) uniform readonly image2D mainYInputImage;
layout(set = 0, binding = 1) uniform writeonly image2D mainYOutputImage;
layout(set = 0, binding = 2) uniform readonly image2D overlayYImage;
layout(set = 0, binding = 3) uniform readonly image2D mainUVInputImage;
layout(set = 0, binding = 4) uniform writeonly image2D mainUVOutputImage;
layout(set = 0, binding = 5) uniform readonly image2D overlayUVImage;

void main()
{
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    float alpha = clamp(pc.global_alpha, 0.0, 1.0);

    if (gid.x < pc.y_size.x && gid.y < pc.y_size.y) {
        ivec2 dst = pc.dst_y_origin + gid;
        ivec2 src = pc.src_y_origin + gid;
        vec4 main_y = imageLoad(mainYInputImage, dst);
        vec4 over_y = imageLoad(overlayYImage, src);

        if (pc.debug_mode != 2)
            imageStore(mainYOutputImage, dst,
                       pc.debug_mode == 1 ? over_y : mix(main_y, over_y, alpha));
    }

    if (gid.x < pc.uv_size.x && gid.y < pc.uv_size.y) {
        ivec2 dst = pc.dst_uv_origin + gid;
        ivec2 src = pc.src_uv_origin + gid;
        vec4 main_uv = imageLoad(mainUVInputImage, dst);
        vec4 over_uv = imageLoad(overlayUVImage, src);

        if (pc.debug_mode != 1)
            imageStore(mainUVOutputImage, dst,
                       pc.debug_mode == 2 ? over_uv : mix(main_uv, over_uv, alpha));
    }
}
