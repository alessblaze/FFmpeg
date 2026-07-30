extern "C" {
#include "libavutil/error.h"
#include "libavutil/log.h"
#include "libavutil/macros.h"
}

#include "amf_overlay_compute.h"
#include "amf_overlay_alpha_comp_spv.h"

#include <AMF/core/Compute.h>
#include <AMF/core/Context.h>
#include <AMF/core/Factory.h>
#include <AMF/core/Plane.h>
#include <AMF/core/Surface.h>
#include <AMF/core/VulkanAMF.h>

#include <vulkan/vulkan.h>

#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

static const unsigned OVERLAY_AMF_MAX_ASYNC_JOBS = 4;
static const unsigned OVERLAY_AMF_MAX_SYNC_RESOURCES = 8;
static const unsigned OVERLAY_AMF_MAX_COPY_PLANES = 4;
static const unsigned OVERLAY_AMF_MAX_COPY_SURFACES = 4;

typedef struct OverlayAlphaPushConstants {
    int32_t dst_origin[2];
    int32_t dst_size[2];
    int32_t src_origin[2];
    int32_t src_size[2];
    int32_t overlay_size[2];
    int32_t premultiplied_alpha;
    float global_alpha;
} OverlayAlphaPushConstants;

typedef struct OverlayAlphaJob {
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
    VkDescriptorPool desc_pool;
    VkDescriptorSet desc_set;
    amf::AMFSurface *main_surface_ref;
    amf::AMFSurface *overlay_surface_ref;
    amf::AMFVulkanSync *tracked_syncs[OVERLAY_AMF_MAX_SYNC_RESOURCES];
    unsigned nb_tracked_syncs;
    int in_flight;
} OverlayAlphaJob;

typedef struct OverlayPendingSync {
    amf::AMFVulkanSync *sync;
    amf::AMFVulkanTimeline *timeline;
    uint64_t signal_value;
    int signaled;
} OverlayPendingSync;

typedef struct OverlayCopyPlane {
    amf::AMFVulkanView *main_view;
    amf::AMFVulkanView *overlay_view;
    VkImageAspectFlags aspect_mask;
    VkOffset3D src_offset;
    VkOffset3D dst_offset;
    VkExtent3D extent;
} OverlayCopyPlane;

typedef struct OverlayCopySurface {
    amf::AMFVulkanSurface *surface;
    VkImageAspectFlags aspect_mask;
    VkImageLayout old_layout;
} OverlayCopySurface;

struct FFAMFOverlayComputeContext {
    amf::AMFContext *context;
    amf::AMFCompute *compute;
    amf::AMFContext1 *context1;
    amf::AMFVulkanDevice *device;
    VkQueue queue;
    uint32_t queue_index;
    uint32_t queue_family_index;
    OverlayAlphaJob alpha_jobs[OVERLAY_AMF_MAX_ASYNC_JOBS];
    unsigned next_alpha_job;
    VkSampler alpha_sampler;
    VkDescriptorSetLayout alpha_desc_set_layout;
    VkPipelineLayout alpha_pipeline_layout;
    VkPipeline alpha_pipeline;
    VkShaderModule alpha_comp_module;

    void (*lock)(void *lock_ctx);
    void (*unlock)(void *lock_ctx);
    void *lock_ctx;
    void *log_ctx;
};

struct AVAMFDeviceContext {
    void *library;
    amf::AMFFactory *factory;
    void *trace_writer;

    int64_t version;
    amf::AMFContext *context;
    amf::AMF_MEMORY_TYPE memory_type;

    void (*lock)(void *lock_ctx);
    void (*unlock)(void *lock_ctx);
    void *lock_ctx;
};

static int amf_result_to_averror(void *log_ctx, const char *what, AMF_RESULT res)
{
    av_log(log_ctx, AV_LOG_ERROR, "%s failed with AMF error %d\n", what, res);
    return AVERROR_EXTERNAL;
}

static int vk_result_to_averror(void *log_ctx, const char *what, VkResult res)
{
    av_log(log_ctx, AV_LOG_ERROR, "%s failed with Vulkan error %d\n", what, (int)res);
    return AVERROR_EXTERNAL;
}

static void reset_alpha_job_tracking(OverlayAlphaJob *job)
{
    job->main_surface_ref = nullptr;
    job->overlay_surface_ref = nullptr;
    job->nb_tracked_syncs = 0;
    job->in_flight = 0;
}

static int wait_alpha_job(FFAMFOverlayComputeContext *ctx, OverlayAlphaJob *job)
{
    VkResult res;

    if (!job->in_flight)
        return 0;

    res = vkWaitForFences(ctx->device->hDevice, 1, &job->fence, VK_TRUE, UINT64_MAX);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkWaitForFences", res);

    for (unsigned i = 0; i < job->nb_tracked_syncs; i++) {
        if (job->tracked_syncs[i] && job->tracked_syncs[i]->hFence == job->fence)
            job->tracked_syncs[i]->hFence = VK_NULL_HANDLE;
        job->tracked_syncs[i] = nullptr;
    }

    if (job->main_surface_ref) {
        job->main_surface_ref->Release();
        job->main_surface_ref = nullptr;
    }
    if (job->overlay_surface_ref) {
        job->overlay_surface_ref->Release();
        job->overlay_surface_ref = nullptr;
    }

    job->nb_tracked_syncs = 0;
    job->in_flight = 0;
    return 0;
}

static void destroy_alpha_job(FFAMFOverlayComputeContext *ctx, OverlayAlphaJob *job)
{
    if (!ctx || !ctx->device || !ctx->device->hDevice)
        return;

    (void)wait_alpha_job(ctx, job);

    if (job->desc_pool)
        vkDestroyDescriptorPool(ctx->device->hDevice, job->desc_pool, nullptr);
    if (job->fence)
        vkDestroyFence(ctx->device->hDevice, job->fence, nullptr);
    if (job->command_pool)
        vkDestroyCommandPool(ctx->device->hDevice, job->command_pool, nullptr);

    job->desc_pool = VK_NULL_HANDLE;
    job->desc_set = VK_NULL_HANDLE;
    job->fence = VK_NULL_HANDLE;
    job->command_pool = VK_NULL_HANDLE;
    job->command_buffer = VK_NULL_HANDLE;
    reset_alpha_job_tracking(job);
}

static void destroy_alpha_pipeline(FFAMFOverlayComputeContext *ctx)
{
    if (!ctx || !ctx->device || !ctx->device->hDevice)
        return;

    if (ctx->alpha_pipeline)
        vkDestroyPipeline(ctx->device->hDevice, ctx->alpha_pipeline, nullptr);

    ctx->alpha_pipeline = VK_NULL_HANDLE;
}

static void destroy_alpha_objects(FFAMFOverlayComputeContext *ctx)
{
    if (!ctx || !ctx->device || !ctx->device->hDevice)
        return;

    destroy_alpha_pipeline(ctx);

    if (ctx->alpha_comp_module)
        vkDestroyShaderModule(ctx->device->hDevice, ctx->alpha_comp_module, nullptr);
    if (ctx->alpha_pipeline_layout)
        vkDestroyPipelineLayout(ctx->device->hDevice, ctx->alpha_pipeline_layout, nullptr);
    if (ctx->alpha_desc_set_layout)
        vkDestroyDescriptorSetLayout(ctx->device->hDevice, ctx->alpha_desc_set_layout, nullptr);
    if (ctx->alpha_sampler)
        vkDestroySampler(ctx->device->hDevice, ctx->alpha_sampler, nullptr);

    ctx->alpha_comp_module = VK_NULL_HANDLE;
    ctx->alpha_pipeline_layout = VK_NULL_HANDLE;
    ctx->alpha_desc_set_layout = VK_NULL_HANDLE;
    ctx->alpha_sampler = VK_NULL_HANDLE;
}

static void log_surface_plane_info(const char *label, amf::AMFSurface *surface, void *log_ctx)
{
    amf::AMFPlane *packed;
    amf::AMFPlane *plane_y;
    amf::AMFPlane *plane_uv;
    void *native = nullptr;
    int plane_count;
    int pixel_size = -1;
    int vpitch = -1;
    int tiled = -1;

    if (!surface) {
        av_log(log_ctx, AV_LOG_VERBOSE, "overlay_amf: %s surface is null\n", label);
        return;
    }

    plane_count = (int)surface->GetPlanesCount();
    packed = surface->GetPlane(amf::AMF_PLANE_PACKED);
    plane_y = surface->GetPlane(amf::AMF_PLANE_Y);
    plane_uv = surface->GetPlane(amf::AMF_PLANE_UV);
    if (packed) {
        native = packed->GetNative();
        pixel_size = packed->GetPixelSizeInBytes();
        vpitch = packed->GetVPitch();
        tiled = packed->IsTiled();
    }

    av_log(log_ctx, AV_LOG_VERBOSE,
           "overlay_amf: %s surface mem=%d format=%d reusable=%d planes=%d packed=%p native=%p pixel=%d pitch=%d vpitch=%d size=%dx%d offset=(%d,%d) tiled=%d\n",
           label,
           (int)surface->GetMemoryType(),
           (int)surface->GetFormat(),
           (int)surface->IsReusable(),
           plane_count,
           packed,
           native,
           pixel_size,
           packed ? packed->GetHPitch() : -1,
           vpitch,
           packed ? packed->GetWidth() : -1,
           packed ? packed->GetHeight() : -1,
           packed ? packed->GetOffsetX() : -1,
           packed ? packed->GetOffsetY() : -1,
           tiled);

    if (plane_y) {
        av_log(log_ctx, AV_LOG_VERBOSE,
               "overlay_amf: %s Y plane native=%p pixel=%d pitch=%d vpitch=%d size=%dx%d offset=(%d,%d) tiled=%d\n",
               label,
               plane_y->GetNative(),
               plane_y->GetPixelSizeInBytes(),
               plane_y->GetHPitch(),
               plane_y->GetVPitch(),
               plane_y->GetWidth(),
               plane_y->GetHeight(),
               plane_y->GetOffsetX(),
               plane_y->GetOffsetY(),
               plane_y->IsTiled());
    }

    if (plane_uv) {
        av_log(log_ctx, AV_LOG_VERBOSE,
               "overlay_amf: %s UV plane native=%p pixel=%d pitch=%d vpitch=%d size=%dx%d offset=(%d,%d) tiled=%d\n",
               label,
               plane_uv->GetNative(),
               plane_uv->GetPixelSizeInBytes(),
               plane_uv->GetHPitch(),
               plane_uv->GetVPitch(),
               plane_uv->GetWidth(),
               plane_uv->GetHeight(),
               plane_uv->GetOffsetX(),
               plane_uv->GetOffsetY(),
               plane_uv->IsTiled());
    }
}

static int interop_or_convert_surface(amf::AMFSurface *surface, amf::AMF_MEMORY_TYPE type,
                                      void *log_ctx, const char *what)
{
    AMF_RESULT res = surface->Interop(type);
    if (res == AMF_OK)
        return 0;

    res = surface->Convert(type);
    if (res != AMF_OK)
        return amf_result_to_averror(log_ctx, what, res);

    return 0;
}

static amf::AMFVulkanView *get_plane_vulkan_view(amf::AMFSurface *surface,
                                                 amf::AMF_PLANE_TYPE plane_type,
                                                 const char *plane_name,
                                                 void *log_ctx)
{
    if (surface->GetDataType() != amf::AMF_DATA_SURFACE) {
        av_log(log_ctx, AV_LOG_ERROR, "AMF overlay expected AMF_DATA_SURFACE, got %d\n",
               (int)surface->GetDataType());
        return nullptr;
    }

    if (surface->GetMemoryType() != amf::AMF_MEMORY_VULKAN) {
        int err = interop_or_convert_surface(surface, amf::AMF_MEMORY_VULKAN,
                                             log_ctx, "AMFSurface::Interop/Convert(VULKAN)");
        if (err < 0)
            return nullptr;
    }

    amf::AMFPlane *plane = surface->GetPlane(plane_type);
    if (!plane) {
        av_log(log_ctx, AV_LOG_ERROR,
               "AMF overlay requires Vulkan view on %s plane\n", plane_name);
        return nullptr;
    }

    auto *view = reinterpret_cast<amf::AMFVulkanView *>(plane->GetNative());
    if (!view || !view->pSurface || !view->hView) {
        av_log(log_ctx, AV_LOG_ERROR,
               "AMF Vulkan %s plane did not expose a native image view\n",
               plane_name);
        return nullptr;
    }

    return view;
}

static amf::AMFVulkanView *get_packed_vulkan_view(amf::AMFSurface *surface, void *log_ctx)
{
    return get_plane_vulkan_view(surface, amf::AMF_PLANE_PACKED, "packed", log_ctx);
}

static int floor_rshift_int(int value, unsigned shift)
{
    if (!shift)
        return value;
    if (value >= 0)
        return value >> shift;
    return -(((-value) + (1 << shift) - 1) >> shift);
}

static int get_copy_layout_from_surface_format(amf::AMF_SURFACE_FORMAT format,
                                               amf::AMF_PLANE_TYPE plane_types[OVERLAY_AMF_MAX_COPY_PLANES],
                                               VkImageAspectFlags plane_aspects[OVERLAY_AMF_MAX_COPY_PLANES],
                                               unsigned plane_shifts_x[OVERLAY_AMF_MAX_COPY_PLANES],
                                               unsigned plane_shifts_y[OVERLAY_AMF_MAX_COPY_PLANES],
                                               unsigned *nb_planes)
{
    switch (format) {
    case amf::AMF_SURFACE_BGRA:
    case amf::AMF_SURFACE_RGBA:
    case amf::AMF_SURFACE_ARGB:
    case amf::AMF_SURFACE_R10G10B10A2:
    case amf::AMF_SURFACE_RGBA_F16:
    case amf::AMF_SURFACE_YUY2:
        plane_types[0] = amf::AMF_PLANE_PACKED;
        plane_aspects[0] = VK_IMAGE_ASPECT_COLOR_BIT;
        plane_shifts_x[0] = 0;
        plane_shifts_y[0] = 0;
        *nb_planes = 1;
        return 0;
    case amf::AMF_SURFACE_NV12:
    case amf::AMF_SURFACE_P010:
        plane_types[0] = amf::AMF_PLANE_Y;
        plane_types[1] = amf::AMF_PLANE_UV;
        plane_aspects[0] = VK_IMAGE_ASPECT_PLANE_0_BIT;
        plane_aspects[1] = VK_IMAGE_ASPECT_PLANE_1_BIT;
        plane_shifts_x[0] = plane_shifts_y[0] = 0;
        plane_shifts_x[1] = plane_shifts_y[1] = 1;
        *nb_planes = 2;
        return 0;
    case amf::AMF_SURFACE_YUV420P:
        plane_types[0] = amf::AMF_PLANE_Y;
        plane_types[1] = amf::AMF_PLANE_U;
        plane_types[2] = amf::AMF_PLANE_V;
        plane_aspects[0] = VK_IMAGE_ASPECT_PLANE_0_BIT;
        plane_aspects[1] = VK_IMAGE_ASPECT_PLANE_1_BIT;
        plane_aspects[2] = VK_IMAGE_ASPECT_PLANE_2_BIT;
        plane_shifts_x[0] = plane_shifts_y[0] = 0;
        plane_shifts_x[1] = plane_shifts_y[1] = 1;
        plane_shifts_x[2] = plane_shifts_y[2] = 1;
        *nb_planes = 3;
        return 0;
    default:
        *nb_planes = 0;
        return AVERROR(ENOSYS);
    }
}

static VkImageAspectFlags resolve_copy_aspect(const amf::AMFVulkanSurface *surface,
                                              VkImageAspectFlags requested_aspect)
{
    VkFormat format;

    if (!surface)
        return requested_aspect;

    if (requested_aspect == VK_IMAGE_ASPECT_COLOR_BIT)
        return requested_aspect;

    format = static_cast<VkFormat>(surface->eFormat);
    switch (format) {
    case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
    case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16:
    case VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16:
    case VK_FORMAT_G16_B16R16_2PLANE_420_UNORM:
    case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM:
    case VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM:
        return requested_aspect;
    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

static int append_copy_surface(OverlayCopySurface surfaces[OVERLAY_AMF_MAX_COPY_SURFACES],
                               unsigned *nb_surfaces,
                               amf::AMFVulkanSurface *surface,
                               VkImageAspectFlags aspect_mask)
{
    for (unsigned i = 0; i < *nb_surfaces; i++) {
        if (surfaces[i].surface == surface) {
            surfaces[i].aspect_mask |= aspect_mask;
            return 0;
        }
    }

    if (*nb_surfaces >= OVERLAY_AMF_MAX_COPY_SURFACES)
        return AVERROR(EINVAL);

    surfaces[*nb_surfaces].surface = surface;
    surfaces[*nb_surfaces].aspect_mask = aspect_mask;
    surfaces[*nb_surfaces].old_layout = static_cast<VkImageLayout>(surface->eCurrentLayout);
    (*nb_surfaces)++;
    return 0;
}

static int get_compute_queue_index(FFAMFOverlayComputeContext *ctx, uint32_t *queue_index)
{
    amf_int64 queue_index64 = 0;
    AMF_RESULT amf_res;

    if (!ctx || !queue_index)
        return AVERROR(EINVAL);

    amf_res = ctx->context1->GetProperty(AMF_CONTEXT_VULKAN_COMPUTE_QUEUE, &queue_index64);
    if (amf_res != AMF_OK) {
        *queue_index = 0;
        return 0;
    }

    if (queue_index64 < 0 || queue_index64 > UINT32_MAX) {
        av_log(ctx->log_ctx, AV_LOG_ERROR,
               "overlay_amf: invalid AMF Vulkan compute queue index %" PRId64 "\n",
               (int64_t)queue_index64);
        return AVERROR(EINVAL);
    }

    *queue_index = static_cast<uint32_t>(queue_index64);
    return 0;
}

static uint32_t find_queue_family_index(const amf::AMFVulkanDevice *device, VkQueue queue,
                                        uint32_t queue_index)
{
    uint32_t family_count = 0;
    std::vector<VkQueueFamilyProperties> family_props;

    vkGetPhysicalDeviceQueueFamilyProperties(device->hPhysicalDevice, &family_count, nullptr);
    if (!family_count)
        return UINT32_MAX;

    family_props.resize(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device->hPhysicalDevice, &family_count,
                                             family_props.data());

    for (uint32_t family = 0; family < family_count; family++) {
        VkQueue candidate = VK_NULL_HANDLE;
        if (queue_index >= family_props[family].queueCount)
            continue;

        vkGetDeviceQueue(device->hDevice, family, queue_index, &candidate);
        if (candidate == queue)
            return family;
    }

    return UINT32_MAX;
}

static void insert_memory_visibility_barrier(VkCommandBuffer cmd,
                                             VkAccessFlags src_access,
                                             VkPipelineStageFlags src_stage,
                                             VkAccessFlags dst_access,
                                             VkPipelineStageFlags dst_stage)
{
    VkMemoryBarrier barrier = {};

    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;

    vkCmdPipelineBarrier(cmd,
                         src_stage,
                         dst_stage,
                         0,
                         1, &barrier,
                         0, nullptr,
                         0, nullptr);
}

static int clamp_overlay_region(int main_width, int main_height,
                                int overlay_width, int overlay_height,
                                int x_position, int y_position,
                                VkOffset3D *src_offset,
                                VkOffset3D *dst_offset,
                                VkExtent3D *extent)
{
    int src_x = 0;
    int src_y = 0;
    int dst_x = x_position;
    int dst_y = y_position;
    int copy_width = overlay_width;
    int copy_height = overlay_height;

    if (dst_x < 0) {
        src_x = -dst_x;
        copy_width += dst_x;
        dst_x = 0;
    }
    if (dst_y < 0) {
        src_y = -dst_y;
        copy_height += dst_y;
        dst_y = 0;
    }
    if (dst_x >= main_width || dst_y >= main_height)
        return 0;

    copy_width = FFMIN(copy_width, main_width - dst_x);
    copy_height = FFMIN(copy_height, main_height - dst_y);
    copy_width = FFMIN(copy_width, overlay_width - src_x);
    copy_height = FFMIN(copy_height, overlay_height - src_y);
    if (copy_width <= 0 || copy_height <= 0)
        return 0;

    *src_offset = { src_x, src_y, 0 };
    *dst_offset = { dst_x, dst_y, 0 };
    *extent = {
        static_cast<uint32_t>(copy_width),
        static_cast<uint32_t>(copy_height),
        1,
    };
    return 1;
}

static int create_alpha_job_command_objects(FFAMFOverlayComputeContext *ctx, OverlayAlphaJob *job)
{
    VkCommandPoolCreateInfo pool_info = {};
    VkCommandBufferAllocateInfo cmd_info = {};
    VkFenceCreateInfo fence_info = {};
    VkResult res;

    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = ctx->queue_family_index;

    res = vkCreateCommandPool(ctx->device->hDevice, &pool_info, nullptr, &job->command_pool);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkCreateCommandPool", res);

    cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_info.commandPool = job->command_pool;
    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_info.commandBufferCount = 1;

    res = vkAllocateCommandBuffers(ctx->device->hDevice, &cmd_info, &job->command_buffer);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkAllocateCommandBuffers", res);

    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    res = vkCreateFence(ctx->device->hDevice, &fence_info, nullptr, &job->fence);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkCreateFence", res);

    return 0;
}

static int create_shader_module(FFAMFOverlayComputeContext *ctx,
                                const unsigned char *code, size_t code_size,
                                VkShaderModule *module, const char *what)
{
    VkShaderModuleCreateInfo info = {};
    VkResult res;

    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = code_size;
    info.pCode = reinterpret_cast<const uint32_t *>(code);

    res = vkCreateShaderModule(ctx->device->hDevice, &info, nullptr, module);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, what, res);

    return 0;
}

static int create_alpha_resources(FFAMFOverlayComputeContext *ctx)
{
    VkSamplerCreateInfo sampler_info = {};
    VkDescriptorSetLayoutBinding bindings[3] = {};
    VkDescriptorSetLayoutCreateInfo set_layout_info = {};
    VkPushConstantRange push_range = {};
    VkPipelineLayoutCreateInfo layout_info = {};
    VkResult res;
    int err;

    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.maxAnisotropy = 1.0f;
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = 0.0f;

    res = vkCreateSampler(ctx->device->hDevice, &sampler_info, nullptr, &ctx->alpha_sampler);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkCreateSampler", res);

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout_info.bindingCount = 3;
    set_layout_info.pBindings = bindings;

    res = vkCreateDescriptorSetLayout(ctx->device->hDevice, &set_layout_info, nullptr,
                                      &ctx->alpha_desc_set_layout);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkCreateDescriptorSetLayout", res);

    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(OverlayAlphaPushConstants);

    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &ctx->alpha_desc_set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;

    res = vkCreatePipelineLayout(ctx->device->hDevice, &layout_info, nullptr,
                                 &ctx->alpha_pipeline_layout);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkCreatePipelineLayout", res);

    /*
     * The fast alpha path binds the main surface as separate readonly and
     * writeonly storage images. This keeps the shader formatless, so the same
     * SPIR-V module can blend packed 8-bit, 10-bit and fp16 AMF Vulkan views.
     */
    err = create_shader_module(ctx, overlay_amf_alpha_comp_spv, overlay_amf_alpha_comp_spv_len,
                               &ctx->alpha_comp_module, "vkCreateShaderModule(comp)");
    if (err < 0)
        return err;

    return 0;
}

static int create_alpha_job_descriptors(FFAMFOverlayComputeContext *ctx, OverlayAlphaJob *job)
{
    VkDescriptorPoolSize pool_sizes[2] = {};
    VkDescriptorPoolCreateInfo pool_info = {};
    VkDescriptorSetAllocateInfo set_alloc_info = {};
    VkResult res;

    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[0].descriptorCount = 2;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[1].descriptorCount = 1;

    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes = pool_sizes;

    res = vkCreateDescriptorPool(ctx->device->hDevice, &pool_info, nullptr, &job->desc_pool);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkCreateDescriptorPool", res);

    set_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_alloc_info.descriptorPool = job->desc_pool;
    set_alloc_info.descriptorSetCount = 1;
    set_alloc_info.pSetLayouts = &ctx->alpha_desc_set_layout;

    res = vkAllocateDescriptorSets(ctx->device->hDevice, &set_alloc_info, &job->desc_set);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkAllocateDescriptorSets", res);

    return 0;
}

static int create_alpha_jobs(FFAMFOverlayComputeContext *ctx)
{
    int err;

    for (unsigned i = 0; i < OVERLAY_AMF_MAX_ASYNC_JOBS; i++) {
        OverlayAlphaJob *job = &ctx->alpha_jobs[i];

        memset(job, 0, sizeof(*job));
        job->command_pool = VK_NULL_HANDLE;
        job->command_buffer = VK_NULL_HANDLE;
        job->fence = VK_NULL_HANDLE;
        job->desc_pool = VK_NULL_HANDLE;
        job->desc_set = VK_NULL_HANDLE;

        err = create_alpha_job_command_objects(ctx, job);
        if (err < 0)
            return err;

        err = create_alpha_job_descriptors(ctx, job);
        if (err < 0)
            return err;
    }

    return 0;
}

static int ensure_alpha_pipeline(FFAMFOverlayComputeContext *ctx)
{
    VkPipelineShaderStageCreateInfo stage = {};
    VkComputePipelineCreateInfo pipeline_info = {};
    VkResult res;

    if (ctx->alpha_pipeline)
        return 0;

    destroy_alpha_pipeline(ctx);

    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = ctx->alpha_comp_module;
    stage.pName = "main";

    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = ctx->alpha_pipeline_layout;

    res = vkCreateComputePipelines(ctx->device->hDevice, VK_NULL_HANDLE, 1,
                                   &pipeline_info, nullptr, &ctx->alpha_pipeline);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkCreateComputePipelines", res);

    return 0;
}

static OverlayAlphaJob *select_alpha_job(FFAMFOverlayComputeContext *ctx, int async_submit)
{
    OverlayAlphaJob *job;
    int err;

    if (!async_submit)
        return &ctx->alpha_jobs[0];

    job = &ctx->alpha_jobs[ctx->next_alpha_job];
    ctx->next_alpha_job = (ctx->next_alpha_job + 1) % OVERLAY_AMF_MAX_ASYNC_JOBS;

    err = wait_alpha_job(ctx, job);
    if (err < 0)
        return nullptr;

    return job;
}

static amf::AMFVulkanTimeline *find_surface_timeline(amf::AMFVulkanSync *sync)
{
    void *ext = sync ? sync->pNext : nullptr;

    while (ext) {
        amf::AMFVulkanTimeline *timeline =
            reinterpret_cast<amf::AMFVulkanTimeline *>(ext);
        if (timeline->eExtensionType == amf::AMF_VARIANT_TIMELINE_SEMAPHORE)
            return timeline;
        ext = timeline->pNext;
    }

    return nullptr;
}

static int collect_surface_sync(FFAMFOverlayComputeContext *ctx,
                                amf::AMFVulkanSync *sync,
                                OverlayPendingSync *pending,
                                VkPipelineStageFlags wait_stage,
                                VkSemaphore *wait_semaphores,
                                VkPipelineStageFlags *wait_stages,
                                uint64_t *wait_values,
                                uint32_t *wait_count,
                                VkSemaphore *signal_semaphores,
                                uint64_t *signal_values,
                                uint32_t *signal_count)
{
    int signal_index = -1;
    int wait_index = -1;
    uint64_t value = 0;

    memset(pending, 0, sizeof(*pending));
    pending->sync = sync;
    pending->timeline = find_surface_timeline(sync);
    if (pending->timeline)
        value = pending->timeline->uiCount;

    if (!sync || sync->hSemaphore == VK_NULL_HANDLE)
        return 0;

    for (uint32_t i = 0; i < *signal_count; i++) {
        if (signal_semaphores[i] == sync->hSemaphore) {
            signal_index = (int)i;
            break;
        }
    }

    if (signal_index < 0) {
        if (sync->bSubmitted || pending->timeline) {
            if (*wait_count >= OVERLAY_AMF_MAX_SYNC_RESOURCES) {
                av_log(ctx->log_ctx, AV_LOG_ERROR,
                       "overlay_amf: too many AMF Vulkan wait semaphores for alpha submit\n");
                return AVERROR_EXTERNAL;
            }
            wait_semaphores[*wait_count] = sync->hSemaphore;
            wait_stages[*wait_count] = wait_stage;
            wait_values[*wait_count] = value;
            (*wait_count)++;
        }

        if (*signal_count >= OVERLAY_AMF_MAX_SYNC_RESOURCES) {
            av_log(ctx->log_ctx, AV_LOG_ERROR,
                   "overlay_amf: too many AMF Vulkan signal semaphores for alpha submit\n");
            return AVERROR_EXTERNAL;
        }

        if (pending->timeline)
            value = pending->timeline->uiCount + 1;

        signal_semaphores[*signal_count] = sync->hSemaphore;
        signal_values[*signal_count] = value;
        pending->signal_value = value;
        pending->signaled = 1;
        (*signal_count)++;
    } else {
        for (uint32_t i = 0; i < *wait_count; i++) {
            if (wait_semaphores[i] == sync->hSemaphore) {
                wait_index = (int)i;
                break;
            }
        }

        if (wait_index >= 0)
            wait_stages[wait_index] |= wait_stage;

        pending->signal_value = signal_values[signal_index];
        pending->signaled = 1;
    }

    return 0;
}

static void transition_surface(VkCommandBuffer cmd, amf::AMFVulkanSurface *surface,
                               VkImageAspectFlags aspect_mask,
                               VkImageLayout new_layout,
                               VkAccessFlags dst_access,
                               VkPipelineStageFlags dst_stage);

static int execute_alpha_blend(FFAMFOverlayComputeContext *ctx,
                               amf::AMFVulkanView *main_view,
                               amf::AMFVulkanView *overlay_view,
                               amf::AMFSurface *main_surface,
                               amf::AMFSurface *overlay_surface,
                               const VkOffset3D *src_offset,
                               const VkOffset3D *dst_offset,
                               const VkExtent3D *extent,
                               int async_submit,
                               int premultiplied_alpha,
                               float global_alpha)
{
    OverlayAlphaJob *job;
    OverlayPendingSync pending_syncs[OVERLAY_AMF_MAX_SYNC_RESOURCES] = {};
    VkCommandBufferBeginInfo begin_info = {};
    VkDescriptorImageInfo image_infos[3] = {};
    VkWriteDescriptorSet writes[3] = {};
    OverlayAlphaPushConstants push = {};
    VkSubmitInfo submit = {};
    VkTimelineSemaphoreSubmitInfo timeline_submit = {};
    VkSemaphore wait_semaphores[OVERLAY_AMF_MAX_SYNC_RESOURCES] = {};
    VkPipelineStageFlags wait_stages[OVERLAY_AMF_MAX_SYNC_RESOURCES] = {};
    uint64_t wait_values[OVERLAY_AMF_MAX_SYNC_RESOURCES] = {};
    VkSemaphore signal_semaphores[OVERLAY_AMF_MAX_SYNC_RESOURCES] = {};
    uint64_t signal_values[OVERLAY_AMF_MAX_SYNC_RESOURCES] = {};
    VkImageLayout main_old_layout;
    VkImageLayout overlay_old_layout;
    uint32_t wait_count = 0;
    uint32_t signal_count = 0;
    uint32_t group_count_x;
    uint32_t group_count_y;
    VkResult res;
    int err;

    err = ensure_alpha_pipeline(ctx);
    if (err < 0)
        return err;

    job = select_alpha_job(ctx, async_submit);
    if (!job)
        return AVERROR_EXTERNAL;

    if (!async_submit && job->in_flight) {
        err = wait_alpha_job(ctx, job);
        if (err < 0)
            return err;
    }

    err = collect_surface_sync(ctx, &main_view->pSurface->Sync, &pending_syncs[0],
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               wait_semaphores, wait_stages, wait_values, &wait_count,
                               signal_semaphores, signal_values, &signal_count);
    if (err < 0)
        return err;
    err = collect_surface_sync(ctx, &overlay_view->pSurface->Sync, &pending_syncs[1],
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               wait_semaphores, wait_stages, wait_values, &wait_count,
                               signal_semaphores, signal_values, &signal_count);
    if (err < 0)
        return err;

    res = vkResetFences(ctx->device->hDevice, 1, &job->fence);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkResetFences", res);

    res = vkResetCommandBuffer(job->command_buffer, 0);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkResetCommandBuffer", res);

    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    res = vkBeginCommandBuffer(job->command_buffer, &begin_info);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkBeginCommandBuffer", res);

    main_old_layout = (VkImageLayout)main_view->pSurface->eCurrentLayout;
    overlay_old_layout = (VkImageLayout)overlay_view->pSurface->eCurrentLayout;

    transition_surface(job->command_buffer, main_view->pSurface,
                       VK_IMAGE_ASPECT_COLOR_BIT,
                       VK_IMAGE_LAYOUT_GENERAL,
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    transition_surface(job->command_buffer, overlay_view->pSurface,
                       VK_IMAGE_ASPECT_COLOR_BIT,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_ACCESS_SHADER_READ_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    image_infos[0].imageView = main_view->hView;
    image_infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_infos[1].imageView = main_view->hView;
    image_infos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_infos[2].sampler = ctx->alpha_sampler;
    image_infos[2].imageView = overlay_view->hView;
    image_infos[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = job->desc_set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].pImageInfo = &image_infos[0];
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = job->desc_set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &image_infos[1];
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = job->desc_set;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &image_infos[2];
    vkUpdateDescriptorSets(ctx->device->hDevice, 3, writes, 0, nullptr);

    push.dst_origin[0] = dst_offset->x;
    push.dst_origin[1] = dst_offset->y;
    push.dst_size[0] = (int32_t)extent->width;
    push.dst_size[1] = (int32_t)extent->height;
    push.src_origin[0] = src_offset->x;
    push.src_origin[1] = src_offset->y;
    push.src_size[0] = (int32_t)extent->width;
    push.src_size[1] = (int32_t)extent->height;
    push.overlay_size[0] = overlay_view->iPlaneWidth;
    push.overlay_size[1] = overlay_view->iPlaneHeight;
    push.premultiplied_alpha = premultiplied_alpha;
    push.global_alpha = global_alpha;

    group_count_x = (extent->width + 7) / 8;
    group_count_y = (extent->height + 7) / 8;

    vkCmdBindPipeline(job->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->alpha_pipeline);
    vkCmdBindDescriptorSets(job->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->alpha_pipeline_layout, 0, 1, &job->desc_set,
                            0, nullptr);
    vkCmdPushConstants(job->command_buffer, ctx->alpha_pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(job->command_buffer, group_count_x, group_count_y, 1);

    insert_memory_visibility_barrier(job->command_buffer,
                                     VK_ACCESS_SHADER_WRITE_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    transition_surface(job->command_buffer, main_view->pSurface,
                       VK_IMAGE_ASPECT_COLOR_BIT,
                       main_old_layout,
                       VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    transition_surface(job->command_buffer, overlay_view->pSurface,
                       VK_IMAGE_ASPECT_COLOR_BIT,
                       overlay_old_layout,
                       VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    res = vkEndCommandBuffer(job->command_buffer);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkEndCommandBuffer", res);

    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = wait_count;
    submit.pWaitSemaphores = wait_count ? wait_semaphores : nullptr;
    submit.pWaitDstStageMask = wait_count ? wait_stages : nullptr;
    submit.signalSemaphoreCount = signal_count;
    submit.pSignalSemaphores = signal_count ? signal_semaphores : nullptr;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &job->command_buffer;

    timeline_submit.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_submit.waitSemaphoreValueCount = wait_count;
    timeline_submit.pWaitSemaphoreValues = wait_count ? wait_values : nullptr;
    timeline_submit.signalSemaphoreValueCount = signal_count;
    timeline_submit.pSignalSemaphoreValues = signal_count ? signal_values : nullptr;
    submit.pNext = &timeline_submit;

    res = vkQueueSubmit(ctx->queue, 1, &submit, job->fence);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkQueueSubmit", res);

    job->nb_tracked_syncs = 0;
    for (unsigned i = 0; i < OVERLAY_AMF_MAX_SYNC_RESOURCES; i++) {
        if (!pending_syncs[i].sync)
            continue;

        if (pending_syncs[i].timeline && pending_syncs[i].signaled)
            pending_syncs[i].timeline->uiCount = pending_syncs[i].signal_value;
        if (pending_syncs[i].sync->hSemaphore != VK_NULL_HANDLE && pending_syncs[i].signaled)
            pending_syncs[i].sync->bSubmitted = true;
        pending_syncs[i].sync->hFence = job->fence;
        job->tracked_syncs[job->nb_tracked_syncs++] = pending_syncs[i].sync;
    }

    main_surface->Acquire();
    job->main_surface_ref = main_surface;
    overlay_surface->Acquire();
    job->overlay_surface_ref = overlay_surface;
    job->in_flight = 1;

    if (!async_submit) {
        err = wait_alpha_job(ctx, job);
        if (err < 0)
            return err;
    }

    return 0;
}

static void transition_surface(VkCommandBuffer cmd, amf::AMFVulkanSurface *surface,
                               VkImageAspectFlags aspect_mask,
                               VkImageLayout new_layout,
                               VkAccessFlags dst_access,
                               VkPipelineStageFlags dst_stage)
{
    if ((VkImageLayout)surface->eCurrentLayout == new_layout)
        return;

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = static_cast<VkImageLayout>(surface->eCurrentLayout);
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = surface->hImage;
    barrier.subresourceRange.aspectMask = aspect_mask;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    switch (barrier.oldLayout) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
        barrier.srcAccessMask = 0;
        break;
    case VK_IMAGE_LAYOUT_PREINITIALIZED:
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        break;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        break;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        break;
    default:
        barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        break;
    }

    barrier.dstAccessMask = dst_access;

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         dst_stage,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &barrier);

    surface->eCurrentLayout = new_layout;
}

static int execute_opaque_copy(FFAMFOverlayComputeContext *ctx,
                               const OverlayCopyPlane *planes,
                               unsigned nb_planes,
                               amf::AMFSurface *main_surface,
                               amf::AMFSurface *overlay_surface)
{
    OverlayAlphaJob *job;
    OverlayPendingSync pending_syncs[OVERLAY_AMF_MAX_SYNC_RESOURCES] = {};
    OverlayCopySurface main_surfaces[OVERLAY_AMF_MAX_COPY_SURFACES] = {};
    OverlayCopySurface overlay_surfaces[OVERLAY_AMF_MAX_COPY_SURFACES] = {};
    VkCommandBufferBeginInfo begin_info = {};
    VkSubmitInfo submit = {};
    VkTimelineSemaphoreSubmitInfo timeline_submit = {};
    VkSemaphore wait_semaphores[OVERLAY_AMF_MAX_SYNC_RESOURCES] = {};
    VkPipelineStageFlags wait_stages[OVERLAY_AMF_MAX_SYNC_RESOURCES] = {};
    uint64_t wait_values[OVERLAY_AMF_MAX_SYNC_RESOURCES] = {};
    VkSemaphore signal_semaphores[OVERLAY_AMF_MAX_SYNC_RESOURCES] = {};
    uint64_t signal_values[OVERLAY_AMF_MAX_SYNC_RESOURCES] = {};
    unsigned nb_main_surfaces = 0;
    unsigned nb_overlay_surfaces = 0;
    unsigned pending_count = 0;
    uint32_t wait_count = 0;
    uint32_t signal_count = 0;
    VkResult res;
    int err;

    if (!nb_planes)
        return 0;

    for (unsigned i = 0; i < nb_planes; i++) {
        err = append_copy_surface(main_surfaces, &nb_main_surfaces,
                                  planes[i].main_view->pSurface,
                                  planes[i].aspect_mask);
        if (err < 0)
            return err;
        err = append_copy_surface(overlay_surfaces, &nb_overlay_surfaces,
                                  planes[i].overlay_view->pSurface,
                                  planes[i].aspect_mask);
        if (err < 0)
            return err;
    }

    job = select_alpha_job(ctx, 0);
    if (!job)
        return AVERROR_EXTERNAL;

    if (job->in_flight) {
        err = wait_alpha_job(ctx, job);
        if (err < 0)
            return err;
    }

    for (unsigned i = 0; i < nb_main_surfaces; i++) {
        if (pending_count >= OVERLAY_AMF_MAX_SYNC_RESOURCES)
            return AVERROR(EINVAL);
        err = collect_surface_sync(ctx, &main_surfaces[i].surface->Sync, &pending_syncs[pending_count++],
                                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   wait_semaphores, wait_stages, wait_values, &wait_count,
                                   signal_semaphores, signal_values, &signal_count);
        if (err < 0)
            return err;
    }
    for (unsigned i = 0; i < nb_overlay_surfaces; i++) {
        if (pending_count >= OVERLAY_AMF_MAX_SYNC_RESOURCES)
            return AVERROR(EINVAL);
        err = collect_surface_sync(ctx, &overlay_surfaces[i].surface->Sync, &pending_syncs[pending_count++],
                                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   wait_semaphores, wait_stages, wait_values, &wait_count,
                                   signal_semaphores, signal_values, &signal_count);
        if (err < 0)
            return err;
    }

    res = vkResetFences(ctx->device->hDevice, 1, &job->fence);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkResetFences", res);

    res = vkResetCommandBuffer(job->command_buffer, 0);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkResetCommandBuffer", res);

    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    res = vkBeginCommandBuffer(job->command_buffer, &begin_info);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkBeginCommandBuffer", res);

    for (unsigned i = 0; i < nb_main_surfaces; i++) {
        transition_surface(job->command_buffer, main_surfaces[i].surface,
                           main_surfaces[i].aspect_mask,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT);
    }
    for (unsigned i = 0; i < nb_overlay_surfaces; i++) {
        transition_surface(job->command_buffer, overlay_surfaces[i].surface,
                           overlay_surfaces[i].aspect_mask,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           VK_ACCESS_TRANSFER_READ_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    for (unsigned i = 0; i < nb_planes; i++) {
        VkImageCopy copy_region = {};

        copy_region.srcSubresource.aspectMask = planes[i].aspect_mask;
        copy_region.srcSubresource.mipLevel = 0;
        copy_region.srcSubresource.baseArrayLayer = 0;
        copy_region.srcSubresource.layerCount = 1;
        copy_region.srcOffset = planes[i].src_offset;
        copy_region.dstSubresource.aspectMask = planes[i].aspect_mask;
        copy_region.dstSubresource.mipLevel = 0;
        copy_region.dstSubresource.baseArrayLayer = 0;
        copy_region.dstSubresource.layerCount = 1;
        copy_region.dstOffset = planes[i].dst_offset;
        copy_region.extent = planes[i].extent;

        vkCmdCopyImage(job->command_buffer,
                       planes[i].overlay_view->pSurface->hImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       planes[i].main_view->pSurface->hImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &copy_region);
    }

    insert_memory_visibility_barrier(job->command_buffer,
                                     VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    for (unsigned i = 0; i < nb_main_surfaces; i++) {
        transition_surface(job->command_buffer, main_surfaces[i].surface,
                           main_surfaces[i].aspect_mask,
                           main_surfaces[i].old_layout,
                           VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }
    for (unsigned i = 0; i < nb_overlay_surfaces; i++) {
        transition_surface(job->command_buffer, overlay_surfaces[i].surface,
                           overlay_surfaces[i].aspect_mask,
                           overlay_surfaces[i].old_layout,
                           VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }

    res = vkEndCommandBuffer(job->command_buffer);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkEndCommandBuffer", res);

    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = wait_count;
    submit.pWaitSemaphores = wait_count ? wait_semaphores : nullptr;
    submit.pWaitDstStageMask = wait_count ? wait_stages : nullptr;
    submit.signalSemaphoreCount = signal_count;
    submit.pSignalSemaphores = signal_count ? signal_semaphores : nullptr;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &job->command_buffer;

    timeline_submit.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_submit.waitSemaphoreValueCount = wait_count;
    timeline_submit.pWaitSemaphoreValues = wait_count ? wait_values : nullptr;
    timeline_submit.signalSemaphoreValueCount = signal_count;
    timeline_submit.pSignalSemaphoreValues = signal_count ? signal_values : nullptr;
    submit.pNext = &timeline_submit;

    res = vkQueueSubmit(ctx->queue, 1, &submit, job->fence);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkQueueSubmit", res);

    job->nb_tracked_syncs = 0;
    for (unsigned i = 0; i < OVERLAY_AMF_MAX_SYNC_RESOURCES; i++) {
        if (!pending_syncs[i].sync)
            continue;

        if (pending_syncs[i].timeline && pending_syncs[i].signaled)
            pending_syncs[i].timeline->uiCount = pending_syncs[i].signal_value;
        if (pending_syncs[i].sync->hSemaphore != VK_NULL_HANDLE && pending_syncs[i].signaled)
            pending_syncs[i].sync->bSubmitted = true;
        pending_syncs[i].sync->hFence = job->fence;
        job->tracked_syncs[job->nb_tracked_syncs++] = pending_syncs[i].sync;
    }

    main_surface->Acquire();
    job->main_surface_ref = main_surface;
    overlay_surface->Acquire();
    job->overlay_surface_ref = overlay_surface;
    job->in_flight = 1;

    err = wait_alpha_job(ctx, job);
    if (err < 0)
        return err;

    return 0;
}

extern "C" int ff_amf_overlay_compute_init(FFAMFOverlayComputeContext **ctx,
                                           AVAMFDeviceContext *device_ctx,
                                           void *log_ctx)
{
    FFAMFOverlayComputeContext *compute_ctx;
    AMF_RESULT amf_res;
    int err;

    if (!ctx || !device_ctx || !device_ctx->context)
        return AVERROR(EINVAL);

    compute_ctx = new (std::nothrow) FFAMFOverlayComputeContext();
    if (!compute_ctx)
        return AVERROR(ENOMEM);
    memset(compute_ctx, 0, sizeof(*compute_ctx));

    compute_ctx->context = device_ctx->context;
    compute_ctx->queue = VK_NULL_HANDLE;
    compute_ctx->queue_family_index = UINT32_MAX;
    compute_ctx->alpha_sampler = VK_NULL_HANDLE;
    compute_ctx->alpha_desc_set_layout = VK_NULL_HANDLE;
    compute_ctx->alpha_pipeline_layout = VK_NULL_HANDLE;
    compute_ctx->alpha_pipeline = VK_NULL_HANDLE;
    compute_ctx->alpha_comp_module = VK_NULL_HANDLE;
    compute_ctx->lock = device_ctx->lock;
    compute_ctx->unlock = device_ctx->unlock;
    compute_ctx->lock_ctx = device_ctx->lock_ctx;
    compute_ctx->log_ctx = log_ctx;

    compute_ctx->context1 = amf::AMFContext1Ptr(compute_ctx->context);
    if (!compute_ctx->context1) {
        delete compute_ctx;
        return AVERROR_EXTERNAL;
    }

    av_log(log_ctx, AV_LOG_VERBOSE,
           "overlay_amf: requesting AMF Vulkan compute device\n");
    amf_res = compute_ctx->context->GetCompute(amf::AMF_MEMORY_VULKAN,
                                               &compute_ctx->compute);
    if (amf_res != AMF_OK || !compute_ctx->compute) {
        delete compute_ctx;
        return amf_result_to_averror(log_ctx, "AMFContext::GetCompute(VULKAN)", amf_res);
    }

    compute_ctx->device =
        reinterpret_cast<amf::AMFVulkanDevice *>(compute_ctx->context1->GetVulkanDevice());
    compute_ctx->queue =
        reinterpret_cast<VkQueue>(compute_ctx->compute->GetNativeCommandQueue());
    if (!compute_ctx->device || !compute_ctx->device->hDevice ||
        !compute_ctx->device->hPhysicalDevice || !compute_ctx->queue) {
        ff_amf_overlay_compute_uninit(&compute_ctx);
        return AVERROR_EXTERNAL;
    }

    err = get_compute_queue_index(compute_ctx, &compute_ctx->queue_index);
    if (err < 0) {
        ff_amf_overlay_compute_uninit(&compute_ctx);
        return err;
    }

    compute_ctx->queue_family_index = find_queue_family_index(compute_ctx->device,
                                                              compute_ctx->queue,
                                                              compute_ctx->queue_index);
    if (compute_ctx->queue_family_index == UINT32_MAX) {
        av_log(log_ctx, AV_LOG_ERROR,
               "overlay_amf: failed to match AMF Vulkan queue to a queue family at queue index %u\n",
               compute_ctx->queue_index);
        ff_amf_overlay_compute_uninit(&compute_ctx);
        return AVERROR_EXTERNAL;
    }

    av_log(log_ctx, AV_LOG_VERBOSE,
           "overlay_amf: using Vulkan queue family %u queue index %u for opaque copy and alpha blend\n",
           compute_ctx->queue_family_index, compute_ctx->queue_index);

    err = create_alpha_resources(compute_ctx);
    if (err < 0) {
        ff_amf_overlay_compute_uninit(&compute_ctx);
        return err;
    }

    err = create_alpha_jobs(compute_ctx);
    if (err < 0) {
        ff_amf_overlay_compute_uninit(&compute_ctx);
        return err;
    }

    *ctx = compute_ctx;
    return 0;
}

extern "C" void ff_amf_overlay_compute_uninit(FFAMFOverlayComputeContext **ctx)
{
    FFAMFOverlayComputeContext *compute_ctx;

    if (!ctx || !*ctx)
        return;

    compute_ctx = *ctx;

    for (unsigned i = 0; i < OVERLAY_AMF_MAX_ASYNC_JOBS; i++)
        destroy_alpha_job(compute_ctx, &compute_ctx->alpha_jobs[i]);
    destroy_alpha_objects(compute_ctx);

    if (compute_ctx->compute)
        compute_ctx->compute->Release();

    delete compute_ctx;
    *ctx = nullptr;
}

extern "C" int ff_amf_overlay_compute_run(FFAMFOverlayComputeContext *ctx,
                                          amf::AMFSurface *main_surface,
                                          amf::AMFSurface *overlay_surface,
                                          int main_width,
                                          int main_height,
                                          int overlay_width,
                                          int overlay_height,
                                          int x_position,
                                          int y_position,
                                          int overlay_has_alpha,
                                          int enable_alpha_blend,
                                          int async_submit,
                                          int premultiplied_alpha,
                                          float global_alpha)
{
    amf::AMFVulkanView *main_view = nullptr;
    amf::AMFVulkanView *overlay_view = nullptr;
    amf::AMF_PLANE_TYPE plane_types[OVERLAY_AMF_MAX_COPY_PLANES] = {};
    VkImageAspectFlags plane_aspects[OVERLAY_AMF_MAX_COPY_PLANES] = {};
    unsigned plane_shifts_x[OVERLAY_AMF_MAX_COPY_PLANES] = {};
    unsigned plane_shifts_y[OVERLAY_AMF_MAX_COPY_PLANES] = {};
    OverlayCopyPlane copy_planes[OVERLAY_AMF_MAX_COPY_PLANES] = {};
    unsigned nb_copy_planes = 0;
    unsigned nb_valid_copy_planes = 0;
    VkOffset3D src_offset = {};
    VkOffset3D dst_offset = {};
    VkExtent3D extent = {};
    int err = 0;

    if (!ctx || !main_surface || !overlay_surface)
        return AVERROR(EINVAL);

    if (ctx->lock)
        ctx->lock(ctx->lock_ctx);

    log_surface_plane_info("main", main_surface, ctx->log_ctx);
    log_surface_plane_info("overlay", overlay_surface, ctx->log_ctx);
    log_surface_plane_info("out", main_surface, ctx->log_ctx);

    if (!clamp_overlay_region(main_width, main_height,
                              overlay_width, overlay_height,
                              x_position, y_position,
                              &src_offset, &dst_offset, &extent)) {
        av_log(ctx->log_ctx, AV_LOG_VERBOSE,
               "overlay_amf: overlay region at (%d,%d) with size %dx%d is fully clipped against main %dx%d\n",
               x_position, y_position, overlay_width, overlay_height,
               main_width, main_height);
        if (ctx->unlock)
            ctx->unlock(ctx->lock_ctx);
        return 0;
    }

    if (overlay_has_alpha && enable_alpha_blend) {
        main_view = get_packed_vulkan_view(main_surface, ctx->log_ctx);
        overlay_view = get_packed_vulkan_view(overlay_surface, ctx->log_ctx);
        if (!main_view || !overlay_view) {
            err = AVERROR_EXTERNAL;
            goto fail;
        }

        av_log(ctx->log_ctx, AV_LOG_VERBOSE,
               "overlay_amf: main Vulkan image=%p layout=%u size=%dx%d overlay image=%p layout=%u size=%dx%d\n",
               (void *)main_view->pSurface->hImage,
               main_view->pSurface->eCurrentLayout,
               main_view->iPlaneWidth,
               main_view->iPlaneHeight,
               (void *)overlay_view->pSurface->hImage,
               overlay_view->pSurface->eCurrentLayout,
               overlay_view->iPlaneWidth,
               overlay_view->iPlaneHeight);

        if (async_submit && main_view->pSurface->Sync.hSemaphore == VK_NULL_HANDLE) {
            av_log(ctx->log_ctx, AV_LOG_VERBOSE,
                   "overlay_amf: async_submit=1 requested but main surface has no AMF Vulkan sync semaphore; falling back to blocking submit\n");
            async_submit = 0;
        }

        av_log(ctx->log_ctx, AV_LOG_VERBOSE,
               "overlay_amf: Vulkan compute alpha blend dst=(%d,%d) src=(%d,%d) size=%ux%u premultiplied=%d global_alpha=%g async=%d\n",
               dst_offset.x, dst_offset.y, src_offset.x, src_offset.y,
               extent.width, extent.height, premultiplied_alpha, global_alpha,
               async_submit);
        err = execute_alpha_blend(ctx, main_view, overlay_view,
                                  main_surface, overlay_surface,
                                  &src_offset, &dst_offset, &extent,
                                  async_submit,
                                  premultiplied_alpha, global_alpha);
        if (err < 0)
            goto fail;
    } else {
        err = get_copy_layout_from_surface_format(main_surface->GetFormat(),
                                                  plane_types, plane_aspects,
                                                  plane_shifts_x, plane_shifts_y,
                                                  &nb_copy_planes);
        if (err < 0) {
            av_log(ctx->log_ctx, AV_LOG_ERROR,
                   "overlay_amf: native opaque copy is not implemented for AMF surface format %d\n",
                   (int)main_surface->GetFormat());
            goto fail;
        }

        for (unsigned i = 0; i < nb_copy_planes; i++) {
            int plane_x = floor_rshift_int(x_position, plane_shifts_x[i]);
            int plane_y = floor_rshift_int(y_position, plane_shifts_y[i]);
            VkImageAspectFlags main_aspect;
            VkImageAspectFlags overlay_aspect;

            main_view = get_plane_vulkan_view(main_surface, plane_types[i], "copy-main", ctx->log_ctx);
            overlay_view = get_plane_vulkan_view(overlay_surface, plane_types[i], "copy-overlay", ctx->log_ctx);
            if (!main_view || !overlay_view) {
                err = AVERROR_EXTERNAL;
                goto fail;
            }

            if (!clamp_overlay_region(main_view->iPlaneWidth, main_view->iPlaneHeight,
                                      overlay_view->iPlaneWidth, overlay_view->iPlaneHeight,
                                      plane_x, plane_y,
                                      &copy_planes[nb_valid_copy_planes].src_offset,
                                      &copy_planes[nb_valid_copy_planes].dst_offset,
                                      &copy_planes[nb_valid_copy_planes].extent)) {
                continue;
            }

            main_aspect = resolve_copy_aspect(main_view->pSurface, plane_aspects[i]);
            overlay_aspect = resolve_copy_aspect(overlay_view->pSurface, plane_aspects[i]);
            if (main_aspect != overlay_aspect) {
                av_log(ctx->log_ctx, AV_LOG_ERROR,
                       "overlay_amf: opaque copy aspect mismatch between main and overlay planes\n");
                err = AVERROR(EINVAL);
                goto fail;
            }

            copy_planes[nb_valid_copy_planes].main_view = main_view;
            copy_planes[nb_valid_copy_planes].overlay_view = overlay_view;
            copy_planes[nb_valid_copy_planes].aspect_mask = main_aspect;
            nb_valid_copy_planes++;
        }

        if (!nb_valid_copy_planes) {
            av_log(ctx->log_ctx, AV_LOG_VERBOSE,
                   "overlay_amf: all opaque copy planes clipped at (%d,%d) for overlay %dx%d\n",
                   x_position, y_position, overlay_width, overlay_height);
            if (ctx->unlock)
                ctx->unlock(ctx->lock_ctx);
            return 0;
        }

        if (async_submit) {
            av_log(ctx->log_ctx, AV_LOG_VERBOSE,
                   "overlay_amf: async_submit=1 currently applies to alpha_blend=1 only; using blocking Vulkan copy path\n");
        }

        av_log(ctx->log_ctx, AV_LOG_VERBOSE,
               "overlay_amf: Vulkan opaque copy using %u plane(s) for AMF surface format %d%s\n",
               nb_valid_copy_planes, (int)main_surface->GetFormat(),
               overlay_has_alpha ? " (alpha currently copied opaquely)" : "");

        err = execute_opaque_copy(ctx, copy_planes, nb_valid_copy_planes,
                                  main_surface, overlay_surface);
        if (err < 0)
            goto fail;
    }

    if (ctx->unlock)
        ctx->unlock(ctx->lock_ctx);
    return 0;

fail:
    if (ctx->unlock)
        ctx->unlock(ctx->lock_ctx);
    return err;
}
