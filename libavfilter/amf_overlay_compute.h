#ifndef AVFILTER_AMF_OVERLAY_COMPUTE_H
#define AVFILTER_AMF_OVERLAY_COMPUTE_H

typedef struct AVAMFDeviceContext AVAMFDeviceContext;

#ifdef __cplusplus
namespace amf {
class AMFSurface;
}
typedef amf::AMFSurface AMFSurface;
extern "C" {
#else
typedef struct AMFSurface AMFSurface;
#endif

typedef struct FFAMFOverlayComputeContext FFAMFOverlayComputeContext;

int ff_amf_overlay_compute_init(FFAMFOverlayComputeContext **ctx,
                                AVAMFDeviceContext *device_ctx,
                                void *log_ctx);
void ff_amf_overlay_compute_uninit(FFAMFOverlayComputeContext **ctx);
int ff_amf_overlay_compute_run(FFAMFOverlayComputeContext *ctx,
                               AMFSurface *main_surface,
                               AMFSurface *overlay_surface,
                               int main_width,
                               int main_height,
                               int overlay_width,
                               int overlay_height,
                               int x_position,
                               int y_position,
                               int overlay_has_alpha,
                               int enable_alpha_blend,
                               int premultiplied_alpha,
                               float global_alpha);

#ifdef __cplusplus
}
#endif

#endif
