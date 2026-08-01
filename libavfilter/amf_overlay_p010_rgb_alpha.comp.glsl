#version 450
#extension GL_EXT_shader_image_load_formatted : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform Push {
    ivec2 dst_y_origin;
    ivec2 y_size;
    ivec2 src_y_origin;
    ivec2 dst_uv_origin;
    ivec2 uv_size;
    ivec2 overlay_size;
    int premultiplied_alpha;
    float global_alpha;
    float kr;
    float kb;
    float y_offset;
    float y_scale;
    float uv_offset;
    float uv_scale;
    int debug_mode;
} pc;

layout(set = 0, binding = 0) uniform readonly image2D mainYInputImage;
layout(set = 0, binding = 1) uniform writeonly image2D mainYOutputImage;
layout(set = 0, binding = 2) uniform readonly image2D mainUVInputImage;
layout(set = 0, binding = 3) uniform writeonly image2D mainUVOutputImage;
layout(set = 0, binding = 4) uniform sampler2D overlayTex;

vec4 sample_overlay_rgba(ivec2 src_pos)
{
    ivec2 clamped_pos = clamp(src_pos, ivec2(0), pc.overlay_size - ivec2(1));
    vec4 src = texelFetch(overlayTex, clamped_pos, 0);
    vec3 src_rgb = clamp(src.rgb, 0.0, 1.0);
    float alpha = clamp(src.a, 0.0, 1.0);

    if (pc.premultiplied_alpha != 0 && alpha > 0.0)
        src_rgb = clamp(src_rgb / alpha, 0.0, 1.0);

    alpha = clamp(alpha * pc.global_alpha, 0.0, 1.0);
    return vec4(src_rgb, alpha);
}

vec3 rgb_to_p010(vec3 rgb)
{
    float kg = 1.0 - pc.kr - pc.kb;
    float y  = pc.kr * rgb.r + kg * rgb.g + pc.kb * rgb.b;
    float cb = (rgb.b - y) / (2.0 * (1.0 - pc.kb));
    float cr = (rgb.r - y) / (2.0 * (1.0 - pc.kr));

    return clamp(vec3(pc.y_offset + pc.y_scale * y,
                      pc.uv_offset + pc.uv_scale * cb,
                      pc.uv_offset + pc.uv_scale * cr), 0.0, 1.0);
}

void main()
{
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);

    if (gid.x < pc.y_size.x && gid.y < pc.y_size.y) {
        ivec2 dst = pc.dst_y_origin + gid;
        ivec2 src_pos = pc.src_y_origin + gid;
        vec4 src = sample_overlay_rgba(src_pos);
        vec4 main_y = imageLoad(mainYInputImage, dst);
        vec3 src_yuv = rgb_to_p010(src.rgb);
        float out_y = src_yuv.x * src.a + main_y.x * (1.0 - src.a);

        if (pc.debug_mode != 2)
            imageStore(mainYOutputImage, dst,
                       vec4(pc.debug_mode == 1 ? src_yuv.x : out_y, 0.0, 0.0, 1.0));
    }

    if (gid.x < pc.uv_size.x && gid.y < pc.uv_size.y) {
        ivec2 dst_uv = pc.dst_uv_origin + gid;
        vec4 main_uv = imageLoad(mainUVInputImage, dst_uv);
        ivec2 uv_luma_origin = dst_uv * 2;
        ivec2 y_begin = max(pc.dst_y_origin, uv_luma_origin);
        ivec2 y_end = min(pc.dst_y_origin + pc.y_size, uv_luma_origin + ivec2(2));
        vec2 premul_uv_sum = vec2(0.0);
        float alpha_sum = 0.0;
        int sample_count = 0;

        for (int y = y_begin.y; y < y_end.y; y++) {
            for (int x = y_begin.x; x < y_end.x; x++) {
                ivec2 src_pos = pc.src_y_origin + ivec2(x, y) - pc.dst_y_origin;
                vec4 src = sample_overlay_rgba(src_pos);
                vec3 src_yuv = rgb_to_p010(src.rgb);
                premul_uv_sum += src_yuv.yz * src.a;
                alpha_sum += src.a;
                sample_count++;
            }
        }

        if (sample_count > 0 && pc.debug_mode != 1) {
            float inv_samples = 1.0 / float(sample_count);
            float avg_alpha = alpha_sum * inv_samples;
            vec2 overlay_uv = premul_uv_sum * inv_samples;
            vec2 out_uv = overlay_uv + main_uv.rg * (1.0 - avg_alpha);

            imageStore(mainUVOutputImage, dst_uv,
                       vec4(pc.debug_mode == 2 ? overlay_uv : out_uv, 0.0, 1.0));
        }
    }
}
