/**
 * @file SchemaFactoryRegistry.cpp
 * @brief Implementation of SchemaFactoryRegistry
 * @author David Sanftenberg
 * @date February 2026
 *
 * Uses self-registration: model-specific TUs register their schema factories
 * during static initialization. No model-specific includes in this file.
 */

#include "SchemaFactoryRegistry.h"
#include "../../../models/ModelRegistrations.h"
#include "utils/Logger.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace llaminar2
{

    // =========================================================================
    // Internal registry storage
    // =========================================================================

    using SchemaFactoryFn = std::function<std::unique_ptr<ISchemaFactory>()>;

    static std::unordered_map<std::string, SchemaFactoryFn> &schemaRegistry()
    {
        static std::unordered_map<std::string, SchemaFactoryFn> s_registry;
        return s_registry;
    }

    static std::mutex &schemaRegistryMutex()
    {
        static std::mutex s_mutex;
        return s_mutex;
    }

    static std::string normalizeArch(const std::string &architecture)
    {
        std::string result = architecture;
        std::transform(result.begin(), result.end(), result.begin(),
                       [](unsigned char c)
                       { return std::tolower(c); });
        return result;
    }

    // =========================================================================
    // Registration API (called from model TUs during static init)
    // =========================================================================

    void SchemaFactoryRegistry::registerFactory(const std::string &architecture,
                                                std::function<std::unique_ptr<ISchemaFactory>()> factory)
    {
        std::lock_guard<std::mutex> lock(schemaRegistryMutex());
        std::string key = normalizeArch(architecture);
        schemaRegistry()[key] = std::move(factory);
        LOG_DEBUG("[SchemaFactoryRegistry] Registered schema factory for '" << key << "'");
    }

    // =========================================================================
    // Lazy built-in registration
    // =========================================================================

    static std::once_flag s_schema_builtin_init;

    static void ensureSchemaBuiltins()
    {
        std::call_once(s_schema_builtin_init, registerBuiltinModels);
    }

    // =========================================================================
    // Query API
    // =========================================================================

    std::unique_ptr<ISchemaFactory> SchemaFactoryRegistry::getFactory(const std::string &architecture)
    {
        ensureSchemaBuiltins();
        std::lock_guard<std::mutex> lock(schemaRegistryMutex());
        std::string key = normalizeArch(architecture);

        auto &reg = schemaRegistry();
        auto it = reg.find(key);
        if (it == reg.end())
        {
            std::string supported;
            for (const auto &[name, _] : reg)
            {
                if (!supported.empty())
                    supported += ", ";
                supported += name;
            }

            throw std::runtime_error(
                "SchemaFactoryRegistry: Unsupported architecture '" + architecture + "'. "
                                                                                     "Registered architectures: [" +
                supported + "]. "
                            "To add support, register a factory in your model's .cpp file using "
                            "REGISTER_SCHEMA_FACTORY(\"your_arch\", YourSchemaFactory).");
        }

        return it->second();
    }

    WeightShardingConfig SchemaFactoryRegistry::getWeightShardingConfig(const std::string &architecture)
    {
        auto factory = getFactory(architecture);
        return factory->getWeightShardingConfig();
    }

    StageShardingConfig SchemaFactoryRegistry::getStageShardingConfig(const std::string &architecture)
    {
        auto factory = getFactory(architecture);
        return factory->getStageShardingConfig();
    }

    bool SchemaFactoryRegistry::isSupported(const std::string &architecture)
    {
        ensureSchemaBuiltins();
        std::lock_guard<std::mutex> lock(schemaRegistryMutex());
        return schemaRegistry().count(normalizeArch(architecture)) > 0;
    }

    std::vector<std::string> SchemaFactoryRegistry::supportedArchitectures()
    {
        ensureSchemaBuiltins();
        std::lock_guard<std::mutex> lock(schemaRegistryMutex());
        std::vector<std::string> result;
        result.reserve(schemaRegistry().size());
        for (const auto &[name, _] : schemaRegistry())
            result.push_back(name);
        std::sort(result.begin(), result.end());
        return result;
    }

} // namespace llaminar2
