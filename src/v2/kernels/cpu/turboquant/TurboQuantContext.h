/**
 * @file TurboQuantContext.h
 * @brief Lifecycle management for TurboQuant random matrices
 *
 * Created once at model load time, held by the graph orchestrator, and
 * passed to KVCacheAppendStage (quantize) and attention stages (dequantize)
 * via stage params.
 *
 * Both the decorrelating rotation and the QJL Gaussian projection are
 * deterministic, so any process with the same head_dim and seeds produces the
 * same quantize/dequantize behavior.
 */

#pragma once

#include "TurboQuantRotation.h"
#include "TurboQuantQJL.h"

#include <memory>

namespace llaminar2
{

    /**
    * @brief Holds the TurboQuant matrices for a model's head_dim.
     *
     * Usage:
     *   // At model load time (DeviceGraphOrchestrator):
     *   auto tq_ctx = std::make_shared<TurboQuantContext>(head_dim);
     *
     *   // Pass to stages:
     *   kv_append_params.turboquant_ctx = tq_ctx.get();
     *   attention_params.turboquant_ctx = tq_ctx.get();
     */
    class TurboQuantContext
    {
    public:
        /**
                 * @brief Construct TurboQuantContext for the given head dimension.
         */
                explicit TurboQuantContext(int head_dim,
                                                                     uint64_t rotation_seed = 0,
                                                                     uint64_t projection_seed = 0)
                        : rotation_(generate_rotation_matrix(head_dim, rotation_seed)),
                            projection_(generate_qjl_projection_matrix(head_dim, projection_seed)),
              head_dim_(head_dim)
        {
        }

                /// Access the decorrelating rotation matrix
        const TurboQuantRotation &rotation() const { return rotation_; }

                /// Access the QJL Gaussian projection matrix used for the residual sketch
                const TurboQuantProjection &projection() const { return projection_; }

        /// Head dimension this context was created for
        int head_dim() const { return head_dim_; }

    private:
        TurboQuantRotation rotation_;
        TurboQuantProjection projection_;
        int head_dim_;
    };

} // namespace llaminar2
