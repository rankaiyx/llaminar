/**
 * @file ModelRegistrations.h
 * @brief Explicit model registration for static library compatibility
 *
 * Static-init self-registration (via file-scope objects) is unreliable
 * when the registration lives inside a static library (.a): the linker
 * strips unreferenced object files, losing the registrations.
 *
 * This file provides an explicit entry point that forces all built-in
 * model registrations to occur. It is called lazily (via std::call_once)
 * from GraphBuilderRegistry and SchemaFactoryRegistry on first access.
 *
 * To add a new model architecture:
 *   1. Create model files under models/<arch>/
 *   2. Add registration calls in ModelRegistrations.cpp
 */

#pragma once

namespace llaminar2
{
    /// Force-register all built-in model graph builders and schema factories.
    /// Idempotent: safe to call multiple times (internally guarded by std::once_flag).
    void registerBuiltinModels();

} // namespace llaminar2
