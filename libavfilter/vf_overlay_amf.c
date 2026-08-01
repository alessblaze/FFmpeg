/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Overlay one AMF video on top of another.
 */

#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_amf.h"
#include "libavutil/hwcontext_amf_internal.h"

#include "AMF/components/ComponentCaps.h"
#include "AMF/components/VideoConverter.h"

#include "amf_overlay_compute.h"
#include "avfilter.h"
#include "avfilter_internal.h"
#include "filters.h"
#include "formats.h"
#include "framesync.h"
#include "vf_amf_common.h"

#define MAIN    0
#define OVERLAY 1

typedef struct OverlayAMFContext {
    AMFFilterContext common;

    FFFrameSync fs;
    FFAMFOverlayComputeContext *compute;

    int x_position;
    int y_position;
    int overlay_has_alpha;
    int enable_alpha_blend;
    int async_submit;
    int premultiplied_alpha;
    int alpha_mode;
    int p010_debug_mode;
    float global_alpha;
    enum AMF_SURFACE_FORMAT main_surface_format;
} OverlayAMFContext;

typedef struct OverlayAMFFormatSupport {
    int supported;
    int native;
} OverlayAMFFormatSupport;

static const char *overlay_amf_pix_fmt_name(enum AVPixelFormat fmt)
{
    const char *name = av_get_pix_fmt_name(fmt);
    return name ? name : "unknown";
}

static int overlay_amf_requested_alpha_mode(const OverlayAMFContext *ctx)
{
    if (ctx->alpha_mode != AVALPHA_MODE_UNSPECIFIED)
        return ctx->alpha_mode;

    if (ctx->premultiplied_alpha >= 0)
        return ctx->premultiplied_alpha ? AVALPHA_MODE_PREMULTIPLIED
                                        : AVALPHA_MODE_STRAIGHT;

    return AVALPHA_MODE_UNSPECIFIED;
}

static int overlay_amf_resolve_premultiplied_alpha(AVFilterContext *avctx,
                                                    const AVFrame *overlay)
{
    OverlayAMFContext *ctx = avctx->priv;
    int requested_alpha_mode = overlay_amf_requested_alpha_mode(ctx);

    if (ctx->alpha_mode != AVALPHA_MODE_UNSPECIFIED &&
        ctx->premultiplied_alpha >= 0 &&
        ctx->premultiplied_alpha != (ctx->alpha_mode == AVALPHA_MODE_PREMULTIPLIED)) {
        av_log(avctx, AV_LOG_WARNING,
               "overlay_amf: alpha=%s overrides conflicting premultiplied=%d\n",
               av_alpha_mode_name(ctx->alpha_mode), ctx->premultiplied_alpha);
    }

    if (requested_alpha_mode == AVALPHA_MODE_PREMULTIPLIED)
        return 1;
    if (requested_alpha_mode == AVALPHA_MODE_STRAIGHT)
        return 0;

    if (overlay && overlay->alpha_mode == AVALPHA_MODE_PREMULTIPLIED)
        return 1;

    return 0;
}

static int overlay_amf_is_supported_packed_rgb_format(enum AVPixelFormat fmt)
{
    switch (fmt) {
    case AV_PIX_FMT_BGRA:
    case AV_PIX_FMT_RGBA:
    case AV_PIX_FMT_BGR0:
    case AV_PIX_FMT_RGBAF16:
        return 1;
    default:
        return 0;
    }
}

static int overlay_amf_is_supported_opaque_format(enum AVPixelFormat fmt)
{
    switch (fmt) {
    case AV_PIX_FMT_BGRA:
    case AV_PIX_FMT_RGBA:
    case AV_PIX_FMT_BGR0:
    case AV_PIX_FMT_RGBAF16:
    case AV_PIX_FMT_NV12:
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_P010:
        return 1;
    default:
        return 0;
    }
}

static enum AVPixelFormat overlay_amf_output_sw_format(enum AVPixelFormat fmt)
{
    switch (fmt) {
    case AV_PIX_FMT_BGR0:
        return AV_PIX_FMT_BGRA;
    default:
        return fmt;
    }
}

static int overlay_amf_sw_format_has_alpha(enum AVPixelFormat fmt)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(fmt);
    if (desc && (desc->flags & AV_PIX_FMT_FLAG_ALPHA))
        return 1;
    return 0;
}

static AVAMFDeviceContext *overlay_amf_get_device_ctx(AVHWFramesContext *frames_ctx)
{
    AVHWDeviceContext *device_ctx;

    if (!frames_ctx || !frames_ctx->device_ref)
        return NULL;

    device_ctx = (AVHWDeviceContext *)frames_ctx->device_ref->data;
    if (!device_ctx || device_ctx->type != AV_HWDEVICE_TYPE_AMF)
        return NULL;

    return device_ctx->hwctx;
}

static void overlay_amf_find_format_support(AMFIOCaps *caps,
                                            enum AMF_SURFACE_FORMAT format,
                                            OverlayAMFFormatSupport *support)
{
    int nb_formats;
    int i;

    if (!support)
        return;

    support->supported = 0;
    support->native = 0;

    if (!caps || format == AMF_SURFACE_UNKNOWN)
        return;

    nb_formats = caps->pVtbl->GetNumOfFormats(caps);
    for (i = 0; i < nb_formats; i++) {
        AMF_SURFACE_FORMAT candidate = AMF_SURFACE_UNKNOWN;
        amf_bool native = 0;

        if (caps->pVtbl->GetFormatAt(caps, i, &candidate, &native) != AMF_OK)
            continue;
        if (candidate != format)
            continue;

        support->supported = 1;
        support->native = !!native;
        return;
    }
}

static int overlay_amf_query_converter_support(AVFilterContext *avctx,
                                               AVAMFDeviceContext *device_ctx,
                                               enum AMF_SURFACE_FORMAT input_format,
                                               enum AMF_SURFACE_FORMAT output_format,
                                               OverlayAMFFormatSupport *input_support,
                                               OverlayAMFFormatSupport *output_support)
{
    AMFComponent *converter = NULL;
    AMFCaps *caps = NULL;
    AMFIOCaps *input_caps = NULL;
    AMFIOCaps *output_caps = NULL;
    AMF_RESULT res;
    int err = 0;

    if (input_support) {
        input_support->supported = 0;
        input_support->native = 0;
    }
    if (output_support) {
        output_support->supported = 0;
        output_support->native = 0;
    }

    if (!device_ctx || !device_ctx->factory || !device_ctx->context)
        return AVERROR(EINVAL);

    av_log(avctx, AV_LOG_VERBOSE,
           "overlay_amf: probing AMF VideoConverter caps for %s -> %s\n",
           overlay_amf_pix_fmt_name(av_amf_to_av_format(input_format)),
           overlay_amf_pix_fmt_name(av_amf_to_av_format(output_format)));

    res = device_ctx->factory->pVtbl->CreateComponent(device_ctx->factory,
                                                      device_ctx->context,
                                                      AMFVideoConverter,
                                                      &converter);
    if (res != AMF_OK || !converter) {
        av_log(avctx, AV_LOG_VERBOSE,
               "overlay_amf: CreateComponent(%ls) for caps probe failed with error %d\n",
               AMFVideoConverter, res);
        return AVERROR_EXTERNAL;
    }

    res = converter->pVtbl->GetCaps(converter, &caps);
    if (res != AMF_OK || !caps) {
        av_log(avctx, AV_LOG_VERBOSE,
               "overlay_amf: AMF VideoConverter GetCaps() failed with error %d\n",
               res);
        err = AVERROR_EXTERNAL;
        goto fail;
    }

    res = caps->pVtbl->GetInputCaps(caps, &input_caps);
    if (res != AMF_OK || !input_caps) {
        av_log(avctx, AV_LOG_VERBOSE,
               "overlay_amf: AMF VideoConverter GetInputCaps() failed with error %d\n",
               res);
        err = AVERROR_EXTERNAL;
        goto fail;
    }

    res = caps->pVtbl->GetOutputCaps(caps, &output_caps);
    if (res != AMF_OK || !output_caps) {
        av_log(avctx, AV_LOG_VERBOSE,
               "overlay_amf: AMF VideoConverter GetOutputCaps() failed with error %d\n",
               res);
        err = AVERROR_EXTERNAL;
        goto fail;
    }

    overlay_amf_find_format_support(input_caps, input_format, input_support);
    overlay_amf_find_format_support(output_caps, output_format, output_support);

fail:
    if (output_caps)
        output_caps->pVtbl->Release(output_caps);
    if (input_caps)
        input_caps->pVtbl->Release(input_caps);
    if (caps)
        caps->pVtbl->Release(caps);
    if (converter)
        converter->pVtbl->Release(converter);

    return err;
}

static void overlay_amf_log_conversion_hint(AVFilterContext *avctx,
                                            AVAMFDeviceContext *device_ctx,
                                            enum AVPixelFormat src_fmt,
                                            enum AVPixelFormat dst_fmt,
                                            const char *label)
{
    OverlayAMFFormatSupport input_support;
    OverlayAMFFormatSupport output_support;
    enum AMF_SURFACE_FORMAT input_format = av_av_to_amf_format(src_fmt);
    enum AMF_SURFACE_FORMAT output_format = av_av_to_amf_format(dst_fmt);
    int ret;

    if (src_fmt == dst_fmt)
        return;

    ret = overlay_amf_query_converter_support(avctx, device_ctx,
                                              input_format, output_format,
                                              &input_support, &output_support);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR,
               "overlay_amf: %s input %s is not directly usable here; expected %s. "
               "Unable to query AMF VideoConverter caps, but inserting vpp_amf=format=%s "
               "before overlay_amf is the intended conversion path.\n",
               label, overlay_amf_pix_fmt_name(src_fmt), overlay_amf_pix_fmt_name(dst_fmt),
               overlay_amf_pix_fmt_name(dst_fmt));
        return;
    }

    if (input_support.supported && output_support.supported) {
        av_log(avctx, AV_LOG_ERROR,
               "overlay_amf: %s input %s is not directly usable here; expected %s. "
               "AMF VideoConverter caps advertise %s input support (native=%d) and "
               "%s output support (native=%d). Insert vpp_amf=format=%s before overlay_amf.\n",
               label, overlay_amf_pix_fmt_name(src_fmt), overlay_amf_pix_fmt_name(dst_fmt),
               overlay_amf_pix_fmt_name(src_fmt), input_support.native,
               overlay_amf_pix_fmt_name(dst_fmt), output_support.native,
               overlay_amf_pix_fmt_name(dst_fmt));
    } else {
        av_log(avctx, AV_LOG_ERROR,
               "overlay_amf: %s input %s is not directly usable here; expected %s. "
               "AMF VideoConverter caps do not advertise the %s -> %s conversion on this "
               "device/driver.\n",
               label, overlay_amf_pix_fmt_name(src_fmt), overlay_amf_pix_fmt_name(dst_fmt),
               overlay_amf_pix_fmt_name(src_fmt), overlay_amf_pix_fmt_name(dst_fmt));
    }
}

static int overlay_amf_output_surface(AVFilterContext *avctx, AVFilterLink *outlink,
                                      AVFrame *input_main, AMFSurface **main_surface)
{
    OverlayAMFContext *ctx = avctx->priv;
    AVFrame *out;
    int ret;

    out = amf_amfsurface_to_avframe(avctx, *main_surface);
    if (!out)
        return AVERROR(ENOMEM);

    ret = av_frame_copy_props(out, input_main);
    if (ret < 0) {
        av_frame_free(&out);
        return ret;
    }

    out->hw_frames_ctx = av_buffer_ref(ctx->common.hwframes_out_ref);
    if (!out->hw_frames_ctx) {
        av_frame_free(&out);
        return AVERROR(ENOMEM);
    }

    out->alpha_mode = outlink->alpha_mode;

    *main_surface = NULL;
    av_frame_free(&input_main);
    return ff_filter_frame(outlink, out);
}

static int overlay_amf_blend(FFFrameSync *fs)
{
    int ret;
    AVFilterContext *avctx = fs->parent;
    OverlayAMFContext *ctx = avctx->priv;
    AVFilterLink *outlink = avctx->outputs[0];
    AVFrame *input_main = NULL;
    AVFrame *input_overlay = NULL;
    AMFSurface *main_surface = NULL;
    AMFSurface *overlay_surface = NULL;
    int premultiplied_alpha = 0;

    ret = ff_framesync_dualinput_get(fs, &input_main, &input_overlay);
    if (ret < 0)
        return ret;

    if (!input_main)
        return AVERROR_BUG;

    ret = amf_avframe_to_amfsurface(avctx, input_main, &main_surface);
    if (ret < 0)
        goto fail;

    if (!input_overlay) {
        ret = overlay_amf_output_surface(avctx, outlink, input_main, &main_surface);
        if (ret < 0)
            goto fail;
        if (overlay_surface)
            overlay_surface->pVtbl->Release(overlay_surface);
        return ret;
    }

    ret = amf_avframe_to_amfsurface(avctx, input_overlay, &overlay_surface);
    if (ret < 0)
        goto fail;

    premultiplied_alpha = overlay_amf_resolve_premultiplied_alpha(avctx,
                                                                  input_overlay);

    ret = ff_amf_overlay_compute_run(ctx->compute,
                                     main_surface,
                                     overlay_surface,
                                     input_main->width,
                                     input_main->height,
                                     input_overlay->width,
                                     input_overlay->height,
                                     ctx->x_position,
                                     ctx->y_position,
                                     ctx->overlay_has_alpha,
                                     ctx->enable_alpha_blend,
                                     ctx->async_submit,
                                     premultiplied_alpha,
                                     ctx->global_alpha,
                                     ctx->p010_debug_mode);
    if (ret < 0)
        goto fail;

    if (overlay_surface) {
        overlay_surface->pVtbl->Release(overlay_surface);
        overlay_surface = NULL;
    }

    ret = overlay_amf_output_surface(avctx, outlink, input_main, &main_surface);
    if (ret < 0)
        goto fail;

    return ret;

fail:
    if (main_surface)
        main_surface->pVtbl->Release(main_surface);
    if (overlay_surface)
        overlay_surface->pVtbl->Release(overlay_surface);
    return ret;
}

static int overlay_amf_config_output(AVFilterLink *outlink)
{
    AVFilterContext *avctx = outlink->src;
    OverlayAMFContext *ctx = avctx->priv;
    AVFilterLink *main_inlink = avctx->inputs[MAIN];
    AVFilterLink *overlay_inlink = avctx->inputs[OVERLAY];
    FilterLink *main_inl = ff_filter_link(main_inlink);
    FilterLink *overlay_inl = ff_filter_link(overlay_inlink);
    AVHWFramesContext *main_fc;
    AVHWFramesContext *overlay_fc;
    AVAMFDeviceContext *device_ctx;
    enum AVPixelFormat in_format;
    int main_is_packed_rgb;
    int overlay_is_packed_rgb;
    int main_is_p010;
    int overlay_is_p010;
    int alpha_is_packed_rgb;
    int alpha_is_p010;
    int main_is_opaque_format;
    int overlay_is_opaque_format;
    const char *path_name;
    int ret;

    if (!main_inl->hw_frames_ctx || !overlay_inl->hw_frames_ctx) {
        av_log(avctx, AV_LOG_ERROR, "overlay_amf requires AMF hardware frames on both inputs\n");
        return AVERROR(EINVAL);
    }

    main_fc = (AVHWFramesContext *)main_inl->hw_frames_ctx->data;
    overlay_fc = (AVHWFramesContext *)overlay_inl->hw_frames_ctx->data;
    device_ctx = overlay_amf_get_device_ctx(main_fc);
    main_is_packed_rgb = overlay_amf_is_supported_packed_rgb_format(main_fc->sw_format);
    overlay_is_packed_rgb = overlay_amf_is_supported_packed_rgb_format(overlay_fc->sw_format);
    main_is_p010 = main_fc->sw_format == AV_PIX_FMT_P010;
    overlay_is_p010 = overlay_fc->sw_format == AV_PIX_FMT_P010;
    alpha_is_packed_rgb = main_is_packed_rgb && overlay_is_packed_rgb;
    alpha_is_p010 = main_is_p010 && overlay_is_p010;
    main_is_opaque_format = overlay_amf_is_supported_opaque_format(main_fc->sw_format);
    overlay_is_opaque_format = overlay_amf_is_supported_opaque_format(overlay_fc->sw_format);

    if (ctx->enable_alpha_blend) {
        if (!alpha_is_packed_rgb && !alpha_is_p010) {
            av_log(avctx, AV_LOG_ERROR,
                   "overlay_amf alpha_blend=1 requires matching packed RGB AMF surfaces or matching p010 AMF surfaces; got %s and %s\n",
                   overlay_amf_pix_fmt_name(main_fc->sw_format),
                   overlay_amf_pix_fmt_name(overlay_fc->sw_format));
            if (!main_is_packed_rgb && !main_is_p010)
                overlay_amf_log_conversion_hint(avctx, device_ctx, main_fc->sw_format,
                                                AV_PIX_FMT_RGBA, "main");
            if (!overlay_is_packed_rgb && !overlay_is_p010)
                overlay_amf_log_conversion_hint(avctx, device_ctx, overlay_fc->sw_format,
                                                AV_PIX_FMT_RGBA, "overlay");
            return AVERROR(ENOSYS);
        }
    } else {
        if (!main_is_opaque_format || !overlay_is_opaque_format) {
            av_log(avctx, AV_LOG_ERROR,
                   "overlay_amf opaque copy currently supports matching AMF surface formats among packed RGB, nv12, yuv420p, and p010; got %s and %s\n",
                   overlay_amf_pix_fmt_name(main_fc->sw_format),
                   overlay_amf_pix_fmt_name(overlay_fc->sw_format));
            if (!main_is_opaque_format)
                overlay_amf_log_conversion_hint(avctx, device_ctx, main_fc->sw_format,
                                                AV_PIX_FMT_BGRA, "main");
            if (!overlay_is_opaque_format)
                overlay_amf_log_conversion_hint(avctx, device_ctx, overlay_fc->sw_format,
                                                overlay_amf_output_sw_format(main_fc->sw_format), "overlay");
            return AVERROR(ENOSYS);
        }
    }

    ctx->main_surface_format = av_av_to_amf_format(main_fc->sw_format);
    ctx->overlay_has_alpha = overlay_amf_sw_format_has_alpha(overlay_fc->sw_format);

    if (ctx->main_surface_format != av_av_to_amf_format(overlay_fc->sw_format)) {
        av_log(avctx, AV_LOG_ERROR,
               "overlay_amf requires matching AMF surface formats; got %s and %s\n",
               overlay_amf_pix_fmt_name(main_fc->sw_format),
               overlay_amf_pix_fmt_name(overlay_fc->sw_format));
        overlay_amf_log_conversion_hint(avctx, device_ctx, overlay_fc->sw_format,
                                        main_fc->sw_format, "overlay");
        return AVERROR(EINVAL);
    }

    if (ctx->enable_alpha_blend && alpha_is_packed_rgb &&
        !ctx->overlay_has_alpha) {
        av_log(avctx, AV_LOG_ERROR,
               "overlay_amf alpha_blend=1 requires an alpha-bearing packed RGB overlay surface; got %s over %s\n",
               overlay_amf_pix_fmt_name(overlay_fc->sw_format),
               overlay_amf_pix_fmt_name(main_fc->sw_format));
        overlay_amf_log_conversion_hint(avctx, device_ctx, overlay_fc->sw_format,
                                        AV_PIX_FMT_RGBA, "overlay");
        return AVERROR(ENOSYS);
    }

    ctx->common.width = main_inlink->w;
    ctx->common.height = main_inlink->h;
    ctx->common.format = overlay_amf_output_sw_format(main_fc->sw_format);
    ctx->common.reset_sar = 0;

    path_name = ctx->enable_alpha_blend && alpha_is_p010 ? "P010 global-alpha blend" :
                ctx->enable_alpha_blend ? "alpha blend" : "native opaque copy";
    av_log(avctx, AV_LOG_VERBOSE,
           "overlay_amf: using %s AMF surfaces directly on the %s path; no vpp_amf pre-conversion required\n",
           overlay_amf_pix_fmt_name(main_fc->sw_format), path_name);

    ret = amf_init_filter_config(outlink, &in_format);
    if (ret < 0)
        return ret;

    ret = ff_amf_overlay_compute_init(&ctx->compute, ctx->common.amf_device_ctx, avctx);
    if (ret < 0)
        return ret;

    ret = ff_framesync_init_dualinput(&ctx->fs, avctx);
    if (ret < 0)
        return ret;

    return ff_framesync_configure(&ctx->fs);
}

static int overlay_amf_query_formats(const AVFilterContext *avctx,
                                     AVFilterFormatsConfig **cfg_in,
                                     AVFilterFormatsConfig **cfg_out)
{
    const OverlayAMFContext *ctx = avctx->priv;
    static const enum AVPixelFormat input_pix_fmts[] = {
        AV_PIX_FMT_AMF_SURFACE,
        AV_PIX_FMT_BGRA,
        AV_PIX_FMT_RGBA,
        AV_PIX_FMT_BGR0,
        AV_PIX_FMT_RGBAF16,
        AV_PIX_FMT_NV12,
        AV_PIX_FMT_YUV420P,
        AV_PIX_FMT_P010,
        AV_PIX_FMT_NONE,
    };
    static const enum AVPixelFormat output_pix_fmts[] = {
        AV_PIX_FMT_AMF_SURFACE,
        AV_PIX_FMT_NONE,
    };
    AVFilterFormats *main_and_output_alpha_modes = NULL;
    AVFilterFormats *overlay_alpha_modes = NULL;
    int alpha_mode = overlay_amf_requested_alpha_mode(ctx);
    int ret;

    ret = ff_formats_ref(ff_make_pixel_format_list(input_pix_fmts),
                         &cfg_in[MAIN]->formats);
    if (ret < 0)
        return ret;

    ret = ff_formats_ref(ff_make_pixel_format_list(input_pix_fmts),
                         &cfg_in[OVERLAY]->formats);
    if (ret < 0)
        return ret;

    ret = ff_formats_ref(ff_make_pixel_format_list(output_pix_fmts),
                         &cfg_out[0]->formats);
    if (ret < 0)
        return ret;

    ret = ff_formats_ref(ff_all_color_spaces(), &cfg_in[MAIN]->color_spaces);
    if (ret < 0)
        return ret;
    ret = ff_formats_ref(ff_all_color_spaces(), &cfg_in[OVERLAY]->color_spaces);
    if (ret < 0)
        return ret;
    ret = ff_formats_ref(ff_all_color_spaces(), &cfg_out[0]->color_spaces);
    if (ret < 0)
        return ret;

    ret = ff_formats_ref(ff_all_color_ranges(), &cfg_in[MAIN]->color_ranges);
    if (ret < 0)
        return ret;
    ret = ff_formats_ref(ff_all_color_ranges(), &cfg_in[OVERLAY]->color_ranges);
    if (ret < 0)
        return ret;
    ret = ff_formats_ref(ff_all_color_ranges(), &cfg_out[0]->color_ranges);
    if (ret < 0)
        return ret;

    main_and_output_alpha_modes = ff_all_alpha_modes();
    ret = ff_formats_ref(main_and_output_alpha_modes, &cfg_in[MAIN]->alpha_modes);
    if (ret < 0)
        return ret;

    ret = ff_formats_ref(main_and_output_alpha_modes, &cfg_out[0]->alpha_modes);
    if (ret < 0)
        return ret;

    overlay_alpha_modes = alpha_mode != AVALPHA_MODE_UNSPECIFIED ?
                          ff_make_formats_list_singleton(alpha_mode) :
                          ff_all_alpha_modes();
    return ff_formats_ref(overlay_alpha_modes, &cfg_in[OVERLAY]->alpha_modes);
}

static av_cold int overlay_amf_init(AVFilterContext *avctx)
{
    OverlayAMFContext *ctx = avctx->priv;

    ctx->fs.on_event = overlay_amf_blend;
    return 0;
}

static av_cold void overlay_amf_uninit(AVFilterContext *avctx)
{
    OverlayAMFContext *ctx = avctx->priv;

    ff_amf_overlay_compute_uninit(&ctx->compute);
    ff_framesync_uninit(&ctx->fs);
    amf_filter_uninit(avctx);
}

static int overlay_amf_activate(AVFilterContext *avctx)
{
    OverlayAMFContext *ctx = avctx->priv;

    return ff_framesync_activate(&ctx->fs);
}

#define OFFSET(x) offsetof(OverlayAMFContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption overlay_amf_options[] = {
    { "x", "Overlay x position", OFFSET(x_position), AV_OPT_TYPE_INT, { .i64 = 0 }, INT_MIN, INT_MAX, FLAGS },
    { "y", "Overlay y position", OFFSET(y_position), AV_OPT_TYPE_INT, { .i64 = 0 }, INT_MIN, INT_MAX, FLAGS },
    { "alpha_blend", "Blend overlay alpha/global alpha instead of copying it opaquely", OFFSET(enable_alpha_blend), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
    { "async_submit", "Submit alpha_blend Vulkan work asynchronously when AMF Vulkan sync is available", OFFSET(async_submit), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
    { "alpha", "alpha format", OFFSET(alpha_mode), AV_OPT_TYPE_INT, { .i64 = AVALPHA_MODE_UNSPECIFIED }, 0, AVALPHA_MODE_NB - 1, FLAGS, .unit = "alpha_mode" },
        { "auto",          "", 0, AV_OPT_TYPE_CONST, { .i64 = AVALPHA_MODE_UNSPECIFIED },   .flags = FLAGS, .unit = "alpha_mode" },
        { "unknown",       "", 0, AV_OPT_TYPE_CONST, { .i64 = AVALPHA_MODE_UNSPECIFIED },   .flags = FLAGS, .unit = "alpha_mode" },
        { "straight",      "", 0, AV_OPT_TYPE_CONST, { .i64 = AVALPHA_MODE_STRAIGHT },      .flags = FLAGS, .unit = "alpha_mode" },
        { "premultiplied", "", 0, AV_OPT_TYPE_CONST, { .i64 = AVALPHA_MODE_PREMULTIPLIED }, .flags = FLAGS, .unit = "alpha_mode" },
    { "premultiplied", "Compatibility alias for alpha mode: -1=auto, 0=straight, 1=premultiplied", OFFSET(premultiplied_alpha), AV_OPT_TYPE_INT, { .i64 = -1 }, -1, 1, FLAGS },
    { "p010_debug", "Debug P010 alpha path by writing only one plane", OFFSET(p010_debug_mode), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 2, FLAGS, .unit = "p010_debug" },
        { "off",     "", 0, AV_OPT_TYPE_CONST, { .i64 = 0 }, .flags = FLAGS, .unit = "p010_debug" },
        { "y_only",  "", 0, AV_OPT_TYPE_CONST, { .i64 = 1 }, .flags = FLAGS, .unit = "p010_debug" },
        { "uv_only", "", 0, AV_OPT_TYPE_CONST, { .i64 = 2 }, .flags = FLAGS, .unit = "p010_debug" },
    { "global_alpha", "Scale overlay alpha when alpha_blend=1", OFFSET(global_alpha), AV_OPT_TYPE_FLOAT, { .dbl = 1.0 }, 0.0, 1.0, FLAGS },
    { "eof_action", "Action to take when encountering EOF from secondary input ",
        OFFSET(fs.opt_eof_action), AV_OPT_TYPE_INT, { .i64 = EOF_ACTION_REPEAT },
        EOF_ACTION_REPEAT, EOF_ACTION_PASS, .flags = FLAGS, .unit = "eof_action" },
        { "repeat", "Repeat the previous frame.",   0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_REPEAT }, .flags = FLAGS, .unit = "eof_action" },
        { "endall", "End both streams.",            0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_ENDALL }, .flags = FLAGS, .unit = "eof_action" },
        { "pass",   "Pass through the main input.", 0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_PASS },   .flags = FLAGS, .unit = "eof_action" },
    { "shortest", "Force termination when the shortest input terminates", OFFSET(fs.opt_shortest), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
    { "repeatlast", "Repeat overlay of the last overlay frame", OFFSET(fs.opt_repeatlast), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, FLAGS },
    { NULL },
};

FRAMESYNC_DEFINE_CLASS(overlay_amf, OverlayAMFContext, fs);

static const AVFilterPad overlay_amf_inputs[] = {
    {
        .name = "main",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    {
        .name = "overlay",
        .type = AVMEDIA_TYPE_VIDEO,
    },
};

static const AVFilterPad overlay_amf_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = overlay_amf_config_output,
    },
};

const FFFilter ff_vf_overlay_amf = {
    .p.name         = "overlay_amf",
    .p.description  = NULL_IF_CONFIG_SMALL("Overlay one AMF video on top of another"),
    .p.priv_class   = &overlay_amf_class,
    .p.flags        = AVFILTER_FLAG_HWDEVICE,
    .priv_size      = sizeof(OverlayAMFContext),
    .preinit        = overlay_amf_framesync_preinit,
    .init           = overlay_amf_init,
    .uninit         = overlay_amf_uninit,
    .activate       = overlay_amf_activate,
    FILTER_QUERY_FUNC2(overlay_amf_query_formats),
    FILTER_INPUTS(overlay_amf_inputs),
    FILTER_OUTPUTS(overlay_amf_outputs),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
