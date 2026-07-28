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

#include "amf_overlay_compute.h"
#include "avfilter.h"
#include "avfilter_internal.h"
#include "filters.h"
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
    enum AMF_SURFACE_FORMAT main_surface_format;
} OverlayAMFContext;

static int overlay_amf_is_supported_sw_format(enum AVPixelFormat fmt)
{
    switch (fmt) {
    case AV_PIX_FMT_BGRA:
    case AV_PIX_FMT_RGBA:
    case AV_PIX_FMT_BGR0:
    case AV_PIX_FMT_RGB0:
        return 1;
    default:
        return 0;
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
                                     ctx->enable_alpha_blend);
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
    enum AVPixelFormat in_format;
    enum AMF_SURFACE_FORMAT overlay_surface_format;
    int ret;

    if (!main_inl->hw_frames_ctx || !overlay_inl->hw_frames_ctx) {
        av_log(avctx, AV_LOG_ERROR, "overlay_amf requires AMF hardware frames on both inputs\n");
        return AVERROR(EINVAL);
    }

    main_fc = (AVHWFramesContext *)main_inl->hw_frames_ctx->data;
    overlay_fc = (AVHWFramesContext *)overlay_inl->hw_frames_ctx->data;

    if (!overlay_amf_is_supported_sw_format(main_fc->sw_format) ||
        !overlay_amf_is_supported_sw_format(overlay_fc->sw_format)) {
        av_log(avctx, AV_LOG_ERROR,
               "overlay_amf currently supports packed RGB AMF surfaces only; got %s and %s\n",
               av_get_pix_fmt_name(main_fc->sw_format),
               av_get_pix_fmt_name(overlay_fc->sw_format));
        return AVERROR(ENOSYS);
    }

    ctx->main_surface_format = av_av_to_amf_format(main_fc->sw_format);
    overlay_surface_format = av_av_to_amf_format(overlay_fc->sw_format);
    if (ctx->main_surface_format != overlay_surface_format) {
        av_log(avctx, AV_LOG_ERROR,
               "overlay_amf requires matching AMF surface formats; got %s and %s\n",
               av_get_pix_fmt_name(main_fc->sw_format),
               av_get_pix_fmt_name(overlay_fc->sw_format));
        return AVERROR(EINVAL);
    }

    if (ctx->enable_alpha_blend &&
        (main_fc->sw_format != AV_PIX_FMT_RGBA ||
         overlay_fc->sw_format != AV_PIX_FMT_RGBA)) {
        av_log(avctx, AV_LOG_ERROR,
               "overlay_amf alpha_blend=1 currently requires RGBA AMF surfaces on both inputs; got %s and %s\n",
               av_get_pix_fmt_name(main_fc->sw_format),
               av_get_pix_fmt_name(overlay_fc->sw_format));
        return AVERROR(ENOSYS);
    }

    ctx->overlay_has_alpha = overlay_fc->sw_format == AV_PIX_FMT_BGRA ||
                             overlay_fc->sw_format == AV_PIX_FMT_RGBA;

    ctx->common.width = main_inlink->w;
    ctx->common.height = main_inlink->h;
    switch (ctx->main_surface_format) {
    case AMF_SURFACE_BGRA:
        ctx->common.format = AV_PIX_FMT_BGRA;
        break;
    case AMF_SURFACE_RGBA:
        ctx->common.format = AV_PIX_FMT_RGBA;
        break;
    default:
        ctx->common.format = AV_PIX_FMT_NONE;
        break;
    }
    ctx->common.reset_sar = 0;

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
    { "alpha_blend", "Blend overlay alpha instead of copying it opaquely", OFFSET(enable_alpha_blend), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
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
    FILTER_INPUTS(overlay_amf_inputs),
    FILTER_OUTPUTS(overlay_amf_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_AMF_SURFACE),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
