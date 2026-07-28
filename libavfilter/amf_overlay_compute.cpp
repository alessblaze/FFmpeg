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
#include <cstring>
#include <cstdint>
#include <new>
#include <vector>

struct FFAMFOverlayComputeContext {
    amf::AMFContext *context;
    amf::AMFCompute *compute;
    amf::AMFContext1 *context1;
    amf::AMFVulkanDevice *device;
    VkQueue queue;
    uint32_t queue_index;
    uint32_t queue_family_index;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
    VkSampler alpha_sampler;
    VkDescriptorSetLayout alpha_desc_set_layout;
    VkDescriptorPool alpha_desc_pool;
    VkDescriptorSet alpha_desc_set;
    VkPipelineLayout alpha_pipeline_layout;
    VkPipeline alpha_pipeline;
    VkShaderModule alpha_comp_module;
    VkFormat alpha_pipeline_format;

    void (*lock)(void *lock_ctx);
    void (*unlock)(void *lock_ctx);
    void *lock_ctx;
    void *log_ctx;
};

typedef struct OverlayAlphaPushConstants {
    int32_t dst_origin[2];
    int32_t dst_size[2];
    int32_t src_origin[2];
    int32_t src_size[2];
    int32_t overlay_size[2];
} OverlayAlphaPushConstants;

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

static void destroy_command_objects(FFAMFOverlayComputeContext *ctx)
{
    if (!ctx || !ctx->device || !ctx->device->hDevice)
        return;

    if (ctx->fence)
        vkDestroyFence(ctx->device->hDevice, ctx->fence, nullptr);
    if (ctx->command_pool)
        vkDestroyCommandPool(ctx->device->hDevice, ctx->command_pool, nullptr);

    ctx->command_pool = VK_NULL_HANDLE;
    ctx->command_buffer = VK_NULL_HANDLE;
    ctx->fence = VK_NULL_HANDLE;
}

static void destroy_alpha_pipeline(FFAMFOverlayComputeContext *ctx)
{
    if (!ctx || !ctx->device || !ctx->device->hDevice)
        return;

    if (ctx->alpha_pipeline)
        vkDestroyPipeline(ctx->device->hDevice, ctx->alpha_pipeline, nullptr);

    ctx->alpha_pipeline = VK_NULL_HANDLE;
    ctx->alpha_pipeline_format = VK_FORMAT_UNDEFINED;
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
    if (ctx->alpha_desc_pool)
        vkDestroyDescriptorPool(ctx->device->hDevice, ctx->alpha_desc_pool, nullptr);
    if (ctx->alpha_desc_set_layout)
        vkDestroyDescriptorSetLayout(ctx->device->hDevice, ctx->alpha_desc_set_layout, nullptr);
    if (ctx->alpha_sampler)
        vkDestroySampler(ctx->device->hDevice, ctx->alpha_sampler, nullptr);

    ctx->alpha_comp_module = VK_NULL_HANDLE;
    ctx->alpha_pipeline_layout = VK_NULL_HANDLE;
    ctx->alpha_desc_pool = VK_NULL_HANDLE;
    ctx->alpha_desc_set_layout = VK_NULL_HANDLE;
    ctx->alpha_sampler = VK_NULL_HANDLE;
    ctx->alpha_desc_set = VK_NULL_HANDLE;
}

static void log_surface_plane_info(const char *label, amf::AMFSurface *surface, void *log_ctx)
{
    amf::AMFPlane *packed;
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

static amf::AMFVulkanView *get_packed_vulkan_view(amf::AMFSurface *surface, void *log_ctx)
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

    amf::AMFPlane *plane = surface->GetPlane(amf::AMF_PLANE_PACKED);
    if (!plane) {
        av_log(log_ctx, AV_LOG_ERROR, "AMF overlay requires packed RGB Vulkan surfaces\n");
        return nullptr;
    }

    auto *view = reinterpret_cast<amf::AMFVulkanView *>(plane->GetNative());
    if (!view || !view->pSurface || !view->hView) {
        av_log(log_ctx, AV_LOG_ERROR,
               "AMF Vulkan packed plane did not expose a native image view\n");
        return nullptr;
    }

    return view;
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

static void transition_surface(VkCommandBuffer cmd, amf::AMFVulkanSurface *surface,
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
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
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

static int create_command_objects(FFAMFOverlayComputeContext *ctx)
{
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = ctx->queue_family_index;

    VkResult res = vkCreateCommandPool(ctx->device->hDevice, &pool_info, nullptr,
                                       &ctx->command_pool);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkCreateCommandPool", res);

    VkCommandBufferAllocateInfo cmd_info = {};
    cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_info.commandPool = ctx->command_pool;
    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_info.commandBufferCount = 1;

    res = vkAllocateCommandBuffers(ctx->device->hDevice, &cmd_info, &ctx->command_buffer);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkAllocateCommandBuffers", res);

    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    res = vkCreateFence(ctx->device->hDevice, &fence_info, nullptr, &ctx->fence);
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
    VkDescriptorSetLayoutBinding bindings[2] = {};
    VkDescriptorSetLayoutCreateInfo set_layout_info = {};
    VkDescriptorPoolSize pool_sizes[2] = {};
    VkDescriptorPoolCreateInfo pool_info = {};
    VkDescriptorSetAllocateInfo set_alloc_info = {};
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
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout_info.bindingCount = 2;
    set_layout_info.pBindings = bindings;

    res = vkCreateDescriptorSetLayout(ctx->device->hDevice, &set_layout_info, nullptr,
                                      &ctx->alpha_desc_set_layout);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkCreateDescriptorSetLayout", res);

    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[0].descriptorCount = 1;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[1].descriptorCount = 1;

    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes = pool_sizes;

    res = vkCreateDescriptorPool(ctx->device->hDevice, &pool_info, nullptr, &ctx->alpha_desc_pool);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkCreateDescriptorPool", res);

    set_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_alloc_info.descriptorPool = ctx->alpha_desc_pool;
    set_alloc_info.descriptorSetCount = 1;
    set_alloc_info.pSetLayouts = &ctx->alpha_desc_set_layout;

    res = vkAllocateDescriptorSets(ctx->device->hDevice, &set_alloc_info, &ctx->alpha_desc_set);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkAllocateDescriptorSets", res);

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
     * The fast alpha path uses a compute shader on the AMF queue.  The shader
     * is currently authored for rgba8 storage images, so alpha blending is
     * intentionally restricted to RGBA surfaces.
     */
    err = create_shader_module(ctx, overlay_amf_alpha_comp_spv, overlay_amf_alpha_comp_spv_len,
                               &ctx->alpha_comp_module, "vkCreateShaderModule(comp)");
    if (err < 0)
        return err;

    ctx->alpha_pipeline_format = VK_FORMAT_UNDEFINED;
    return 0;
}

static int ensure_alpha_pipeline(FFAMFOverlayComputeContext *ctx, VkFormat format)
{
    VkPipelineShaderStageCreateInfo stage = {};
    VkComputePipelineCreateInfo pipeline_info = {};
    VkResult res;

    if (ctx->alpha_pipeline && ctx->alpha_pipeline_format == format)
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

    ctx->alpha_pipeline_format = format;
    return 0;
}

static int execute_alpha_blend(FFAMFOverlayComputeContext *ctx,
                               amf::AMFVulkanView *main_view,
                               amf::AMFVulkanView *overlay_view,
                               const VkOffset3D *src_offset,
                               const VkOffset3D *dst_offset,
                               const VkExtent3D *extent)
{
    VkCommandBufferBeginInfo begin_info = {};
    VkDescriptorImageInfo image_infos[2] = {};
    VkWriteDescriptorSet writes[2] = {};
    OverlayAlphaPushConstants push = {};
    VkSubmitInfo submit = {};
    VkImageLayout main_old_layout;
    VkImageLayout overlay_old_layout;
    uint32_t group_count_x;
    uint32_t group_count_y;
    VkResult res;
    int err;

    err = ensure_alpha_pipeline(ctx, (VkFormat)main_view->pSurface->eFormat);
    if (err < 0)
        return err;

    vkResetFences(ctx->device->hDevice, 1, &ctx->fence);
    vkResetCommandBuffer(ctx->command_buffer, 0);

    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    res = vkBeginCommandBuffer(ctx->command_buffer, &begin_info);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkBeginCommandBuffer", res);

    main_old_layout = (VkImageLayout)main_view->pSurface->eCurrentLayout;
    overlay_old_layout = (VkImageLayout)overlay_view->pSurface->eCurrentLayout;

    transition_surface(ctx->command_buffer, main_view->pSurface,
                       VK_IMAGE_LAYOUT_GENERAL,
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    transition_surface(ctx->command_buffer, overlay_view->pSurface,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_ACCESS_SHADER_READ_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    image_infos[0].imageView = main_view->hView;
    image_infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_infos[1].sampler = ctx->alpha_sampler;
    image_infos[1].imageView = overlay_view->hView;
    image_infos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = ctx->alpha_desc_set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].pImageInfo = &image_infos[0];
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = ctx->alpha_desc_set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &image_infos[1];
    vkUpdateDescriptorSets(ctx->device->hDevice, 2, writes, 0, nullptr);

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

    group_count_x = (extent->width + 7) / 8;
    group_count_y = (extent->height + 7) / 8;

    vkCmdBindPipeline(ctx->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->alpha_pipeline);
    vkCmdBindDescriptorSets(ctx->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->alpha_pipeline_layout, 0, 1, &ctx->alpha_desc_set,
                            0, nullptr);
    vkCmdPushConstants(ctx->command_buffer, ctx->alpha_pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(ctx->command_buffer, group_count_x, group_count_y, 1);

    transition_surface(ctx->command_buffer, main_view->pSurface,
                       main_old_layout,
                       VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    transition_surface(ctx->command_buffer, overlay_view->pSurface,
                       overlay_old_layout,
                       VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    res = vkEndCommandBuffer(ctx->command_buffer);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkEndCommandBuffer", res);

    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &ctx->command_buffer;

    res = vkQueueSubmit(ctx->queue, 1, &submit, ctx->fence);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkQueueSubmit", res);

    res = vkWaitForFences(ctx->device->hDevice, 1, &ctx->fence, VK_TRUE, UINT64_MAX);
    if (res != VK_SUCCESS)
        return vk_result_to_averror(ctx->log_ctx, "vkWaitForFences", res);

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

    compute_ctx->context = device_ctx->context;
    compute_ctx->compute = nullptr;
    compute_ctx->context1 = nullptr;
    compute_ctx->device = nullptr;
    compute_ctx->queue = VK_NULL_HANDLE;
    compute_ctx->queue_index = 0;
    compute_ctx->queue_family_index = UINT32_MAX;
    compute_ctx->command_pool = VK_NULL_HANDLE;
    compute_ctx->command_buffer = VK_NULL_HANDLE;
    compute_ctx->fence = VK_NULL_HANDLE;
    compute_ctx->alpha_sampler = VK_NULL_HANDLE;
    compute_ctx->alpha_desc_set_layout = VK_NULL_HANDLE;
    compute_ctx->alpha_desc_pool = VK_NULL_HANDLE;
    compute_ctx->alpha_desc_set = VK_NULL_HANDLE;
    compute_ctx->alpha_pipeline_layout = VK_NULL_HANDLE;
    compute_ctx->alpha_pipeline = VK_NULL_HANDLE;
    compute_ctx->alpha_comp_module = VK_NULL_HANDLE;
    compute_ctx->alpha_pipeline_format = VK_FORMAT_UNDEFINED;
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
           "overlay_amf: using AMF Compute::CopyPlane plus Vulkan compute alpha blend on queue family %u queue index %u\n",
           compute_ctx->queue_family_index, compute_ctx->queue_index);

    err = create_command_objects(compute_ctx);
    if (err < 0) {
        ff_amf_overlay_compute_uninit(&compute_ctx);
        return err;
    }

    err = create_alpha_resources(compute_ctx);
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
    destroy_alpha_objects(compute_ctx);
    destroy_command_objects(compute_ctx);

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
                                          int enable_alpha_blend)
{
    amf::AMFVulkanView *main_view = nullptr;
    amf::AMFVulkanView *overlay_view = nullptr;
    amf::AMFPlane *main_plane = nullptr;
    amf::AMFPlane *overlay_plane = nullptr;
    VkOffset3D src_offset = {};
    VkOffset3D dst_offset = {};
    VkExtent3D extent = {};
    amf_size src_origin[3] = {};
    amf_size dst_origin[3] = {};
    amf_size region[3] = {};
    AMF_RESULT amf_res = AMF_OK;
    int err = 0;

    if (!ctx || !main_surface || !overlay_surface)
        return AVERROR(EINVAL);

    if (ctx->lock)
        ctx->lock(ctx->lock_ctx);

    log_surface_plane_info("main", main_surface, ctx->log_ctx);
    log_surface_plane_info("overlay", overlay_surface, ctx->log_ctx);
    log_surface_plane_info("out", main_surface, ctx->log_ctx);

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

    main_plane = main_surface->GetPlane(amf::AMF_PLANE_PACKED);
    overlay_plane = overlay_surface->GetPlane(amf::AMF_PLANE_PACKED);
    if (!main_plane || !overlay_plane) {
        av_log(ctx->log_ctx, AV_LOG_ERROR,
               "overlay_amf: missing packed plane after Vulkan conversion\n");
        err = AVERROR(EINVAL);
        goto fail;
    }

    if (!clamp_overlay_region(main_width, main_height,
                              overlay_width, overlay_height,
                              x_position, y_position,
                              &src_offset, &dst_offset, &extent)) {
        if (ctx->unlock)
            ctx->unlock(ctx->lock_ctx);
        return 0;
    }

    src_origin[0] = src_offset.x;
    src_origin[1] = src_offset.y;
    src_origin[2] = 0;
    dst_origin[0] = dst_offset.x;
    dst_origin[1] = dst_offset.y;
    dst_origin[2] = 0;
    region[0] = extent.width;
    region[1] = extent.height;
    region[2] = 1;

    if (overlay_has_alpha && enable_alpha_blend) {
        av_log(ctx->log_ctx, AV_LOG_VERBOSE,
               "overlay_amf: Vulkan compute alpha blend dst=(%d,%d) src=(%d,%d) size=%ux%u\n",
               dst_offset.x, dst_offset.y, src_offset.x, src_offset.y,
               extent.width, extent.height);
        err = execute_alpha_blend(ctx, main_view, overlay_view,
                                  &src_offset, &dst_offset, &extent);
        if (err < 0)
            goto fail;
    } else {
        av_log(ctx->log_ctx, AV_LOG_VERBOSE,
               "overlay_amf: CopyPlane overlay -> main dst=(%d,%d) src=(%d,%d) size=%ux%u%s\n",
               dst_offset.x, dst_offset.y, src_offset.x, src_offset.y,
               extent.width, extent.height,
               overlay_has_alpha ? " (alpha currently copied opaquely)" : "");

        amf_res = ctx->compute->CopyPlane(overlay_plane, src_origin, region, main_plane, dst_origin);
        if (amf_res != AMF_OK) {
            err = amf_result_to_averror(ctx->log_ctx, "AMFCompute::CopyPlane", amf_res);
            goto fail;
        }

        amf_res = ctx->compute->FinishQueue();
        if (amf_res != AMF_OK) {
            err = amf_result_to_averror(ctx->log_ctx, "AMFCompute::FinishQueue", amf_res);
            goto fail;
        }
    }
    if (ctx->unlock)
        ctx->unlock(ctx->lock_ctx);
    return 0;
fail:
    if (ctx->unlock)
        ctx->unlock(ctx->lock_ctx);
    return err;
}
