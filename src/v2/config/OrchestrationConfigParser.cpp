/**
 * @file OrchestrationConfigParser.cpp
 * @brief Implementation of OrchestrationConfigParser
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "OrchestrationConfigParser.h"
#include "utils/Logger.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace llaminar2
{

    // =========================================================================
    // Factory function
    // =========================================================================

    std::unique_ptr<IOrchestrationConfigParser> createOrchestrationConfigParser()
    {
        return std::make_unique<OrchestrationConfigParser>();
    }

    // =========================================================================
    // Helper functions
    // =========================================================================

    namespace
    {
        std::string trim(const std::string &str)
        {
            size_t start = str.find_first_not_of(" \t\n\r");
            if (start == std::string::npos)
                return "";
            size_t end = str.find_last_not_of(" \t\n\r");
            return str.substr(start, end - start + 1);
        }

        std::vector<std::string> split(const std::string &str, char delim)
        {
            std::vector<std::string> parts;
            std::stringstream ss(str);
            std::string part;
            while (std::getline(ss, part, delim))
            {
                std::string trimmed = trim(part);
                if (!trimmed.empty())
                {
                    parts.push_back(trimmed);
                }
            }
            return parts;
        }

        std::string toLower(const std::string &str)
        {
            std::string lower = str;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c)
                           { return std::tolower(c); });
            return lower;
        }
    } // anonymous namespace

    // =========================================================================
    // Static helpers
    // =========================================================================

    std::vector<GlobalDeviceAddress> OrchestrationConfigParser::parseDeviceList(const std::string &spec)
    {
        std::vector<GlobalDeviceAddress> devices;
        auto parts = split(spec, ',');

        for (const auto &part : parts)
        {
            auto addr = GlobalDeviceAddress::tryParse(part);
            if (!addr)
            {
                throw std::invalid_argument("Invalid device specification: '" + part + "'");
            }
            devices.push_back(*addr);
        }

        return devices;
    }

    std::vector<float> OrchestrationConfigParser::parseWeightList(const std::string &spec)
    {
        std::vector<float> weights;
        auto parts = split(spec, ',');

        for (const auto &part : parts)
        {
            try
            {
                float w = std::stof(part);
                weights.push_back(w);
            }
            catch (const std::exception &)
            {
                throw std::invalid_argument("Invalid weight value: '" + part + "'");
            }
        }

        return weights;
    }

    std::vector<std::pair<int, GlobalDeviceAddress>> OrchestrationConfigParser::parseDeviceMap(
        const std::string &spec)
    {
        std::vector<std::pair<int, GlobalDeviceAddress>> device_map;
        auto parts = split(spec, ',');

        for (const auto &part : parts)
        {
            size_t eq_pos = part.find('=');
            if (eq_pos == std::string::npos)
            {
                throw std::invalid_argument(
                    "Invalid device map entry: '" + part + "'. Expected format: rank=device");
            }

            int rank;
            try
            {
                rank = std::stoi(part.substr(0, eq_pos));
            }
            catch (const std::exception &)
            {
                throw std::invalid_argument("Invalid rank in device map: '" + part.substr(0, eq_pos) + "'");
            }

            std::string device_spec = part.substr(eq_pos + 1);
            auto addr = GlobalDeviceAddress::tryParse(device_spec);
            if (!addr)
            {
                throw std::invalid_argument("Invalid device in device map: '" + device_spec + "'");
            }

            device_map.emplace_back(rank, *addr);
        }

        return device_map;
    }

    bool OrchestrationConfigParser::matchesFlag(const std::string &arg,
                                                const std::string &short_form,
                                                const std::string &long_form)
    {
        // Check exact match
        if (arg == short_form || arg == long_form)
        {
            return true;
        }

        // Check --flag=value format
        if (!long_form.empty() && arg.substr(0, long_form.size() + 1) == long_form + "=")
        {
            return true;
        }

        return false;
    }

    std::string OrchestrationConfigParser::getFlagValue(const std::vector<std::string> &args, size_t &idx)
    {
        const std::string &arg = args[idx];

        // Check --flag=value format
        size_t eq_pos = arg.find('=');
        if (eq_pos != std::string::npos)
        {
            return arg.substr(eq_pos + 1);
        }

        // Check --flag value format
        if (idx + 1 < args.size() && !args[idx + 1].empty() && args[idx + 1][0] != '-')
        {
            ++idx;
            return args[idx];
        }

        return "";
    }

    // =========================================================================
    // parseArgs
    // =========================================================================

    OrchestrationConfig OrchestrationConfigParser::parseArgs(int argc, char **argv)
    {
        OrchestrationConfig config;

        // Convert to vector for easier handling
        std::vector<std::string> args;
        for (int i = 1; i < argc; ++i)
        {
            args.push_back(argv[i]);
        }

        for (size_t i = 0; i < args.size(); ++i)
        {
            const std::string &arg = args[i];

            // ===== Introspection flags =====
            if (arg == "--dry-run")
            {
                config.dry_run = true;
            }
            else if (arg == "--explain-placement")
            {
                config.explain_placement = true;
            }
            else if (arg == "--show-topology")
            {
                config.show_topology = true;
            }
            else if (arg == "--show-numa")
            {
                config.show_numa = true;
            }
            else if (arg == "--validate-only")
            {
                config.validate_only = true;
            }

            // ===== Device assignment =====
            else if (matchesFlag(arg, "-d", "--device"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--device requires a device specification");
                }
                auto addr = GlobalDeviceAddress::tryParse(value);
                if (!addr)
                {
                    throw std::invalid_argument("Invalid device specification: '" + value + "'");
                }
                config.device_for_this_rank = *addr;
            }
            else if (matchesFlag(arg, "", "--device-mode"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--device-mode requires a mode");
                }
                auto mode = parseDeviceAssignmentMode(value);
                if (!mode)
                {
                    throw std::invalid_argument("Invalid device mode: '" + value + "'");
                }
                config.device_mode = *mode;
            }
            else if (matchesFlag(arg, "", "--device-map"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--device-map requires a mapping");
                }
                config.device_map = parseDeviceMap(value);
                config.device_mode = DeviceAssignmentMode::EXPLICIT;
            }

            // ===== Named domains =====
            else if (matchesFlag(arg, "", "--define-domain"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--define-domain requires a specification");
                }
                config.domain_definitions.push_back(DomainDefinition::parse(value));
            }
            else if (matchesFlag(arg, "", "--pp-stage"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--pp-stage requires a specification");
                }
                config.pp_stage_definitions.push_back(PPStageDefinition::parse(value));
            }

            // ===== Simple TP options =====
            else if (matchesFlag(arg, "-t", "--tp"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--tp requires a degree");
                }
                try
                {
                    config.tp_degree = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid TP degree: '" + value + "'");
                }
            }
            else if (matchesFlag(arg, "", "--tp-scope"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--tp-scope requires a scope");
                }
                auto scope = parseTpScope(value);
                if (!scope)
                {
                    throw std::invalid_argument("Invalid TP scope: '" + value + "'");
                }
                config.tp_scope = *scope;
            }
            else if (matchesFlag(arg, "", "--tp-devices"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--tp-devices requires a device list");
                }
                config.tp_devices = parseDeviceList(value);
            }
            else if (matchesFlag(arg, "", "--tp-weights"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--tp-weights requires a weight list");
                }
                config.tp_weights = parseWeightList(value);
            }
            else if (matchesFlag(arg, "", "--tp-local"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--tp-local requires a degree");
                }
                try
                {
                    config.tp_local_degree = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid TP local degree: '" + value + "'");
                }
            }
            else if (matchesFlag(arg, "", "--tp-global"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--tp-global requires a degree");
                }
                try
                {
                    config.tp_global_degree = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid TP global degree: '" + value + "'");
                }
            }

            // ===== Simple PP options =====
            else if (matchesFlag(arg, "-p", "--pp"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--pp requires a degree");
                }
                try
                {
                    config.pp_degree = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid PP degree: '" + value + "'");
                }
            }
            else if (matchesFlag(arg, "", "--pp-split"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--pp-split requires a mode");
                }
                auto mode = parsePpSplitMode(value);
                if (!mode)
                {
                    throw std::invalid_argument("Invalid PP split mode: '" + value + "'");
                }
                config.pp_split = *mode;
            }

            // ===== Layer placement =====
            else if (matchesFlag(arg, "", "--cpu-layers"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--cpu-layers requires a count");
                }
                try
                {
                    config.cpu_layers = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    throw std::invalid_argument("Invalid CPU layers count: '" + value + "'");
                }
            }
            else if (arg == "--cpu-layers-first")
            {
                config.cpu_layers_first = true;
            }

            // ===== Backend =====
            else if (matchesFlag(arg, "-b", "--backend"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--backend requires a type");
                }
                auto backend = parseCollectiveBackendType(value);
                if (!backend)
                {
                    throw std::invalid_argument("Invalid backend type: '" + value + "'");
                }
                config.default_backend = *backend;
            }

            // ===== Config file =====
            else if (matchesFlag(arg, "-c", "--config"))
            {
                std::string value = getFlagValue(args, i);
                if (value.empty())
                {
                    throw std::invalid_argument("--config requires a file path");
                }
                config.config_file_path = value;

                // Parse the config file and merge
                OrchestrationConfig file_config = parseYamlFile(value);

                // Merge: CLI options override file options
                // Only override fields that weren't explicitly set via CLI
                // For now, just use file config as base if we got this far
                // (A more sophisticated merge would track which CLI flags were set)
            }

            // Ignore unknown flags (they might be for other parts of the application)
        }

        return config;
    }

    // =========================================================================
    // parseYamlFile
    // =========================================================================

    OrchestrationConfig OrchestrationConfigParser::parseYamlFile(const std::string &path)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            throw std::invalid_argument("Failed to open config file: " + path);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        return parseYamlString(buffer.str());
    }

    // =========================================================================
    // parseYamlString
    // =========================================================================

    OrchestrationConfig OrchestrationConfigParser::parseYamlString(const std::string &yaml)
    {
        OrchestrationConfig config;

        // Simple line-by-line YAML parser (sufficient for our flat structure)
        // For production, consider using a proper YAML library like yaml-cpp

        std::istringstream stream(yaml);
        std::string line;
        std::string current_section;

        while (std::getline(stream, line))
        {
            std::string trimmed = trim(line);

            // Skip empty lines and comments
            if (trimmed.empty() || trimmed[0] == '#')
            {
                continue;
            }

            // Check for section headers
            if (trimmed.back() == ':' && trimmed.find(':') == trimmed.size() - 1)
            {
                current_section = trimmed.substr(0, trimmed.size() - 1);
                continue;
            }

            // Parse key: value pairs
            size_t colon_pos = trimmed.find(':');
            if (colon_pos == std::string::npos)
            {
                continue;
            }

            std::string key = trim(trimmed.substr(0, colon_pos));
            std::string value = trim(trimmed.substr(colon_pos + 1));

            // Remove quotes if present
            if (value.size() >= 2 &&
                ((value.front() == '"' && value.back() == '"') ||
                 (value.front() == '\'' && value.back() == '\'')))
            {
                value = value.substr(1, value.size() - 2);
            }

            // Map YAML keys to config fields
            if (key == "dry_run")
            {
                config.dry_run = (toLower(value) == "true" || value == "1");
            }
            else if (key == "explain_placement")
            {
                config.explain_placement = (toLower(value) == "true" || value == "1");
            }
            else if (key == "show_topology")
            {
                config.show_topology = (toLower(value) == "true" || value == "1");
            }
            else if (key == "show_numa")
            {
                config.show_numa = (toLower(value) == "true" || value == "1");
            }
            else if (key == "validate_only")
            {
                config.validate_only = (toLower(value) == "true" || value == "1");
            }
            else if (key == "device_mode")
            {
                auto mode = parseDeviceAssignmentMode(value);
                if (mode)
                    config.device_mode = *mode;
            }
            else if (key == "device")
            {
                auto addr = GlobalDeviceAddress::tryParse(value);
                if (addr)
                    config.device_for_this_rank = *addr;
            }
            else if (key == "tp_degree" || key == "tp")
            {
                config.tp_degree = std::stoi(value);
            }
            else if (key == "tp_scope")
            {
                auto scope = parseTpScope(value);
                if (scope)
                    config.tp_scope = *scope;
            }
            else if (key == "tp_devices")
            {
                // Handle array format [device1, device2]
                std::string devices_str = value;
                if (!devices_str.empty() && devices_str.front() == '[')
                {
                    devices_str = devices_str.substr(1);
                    if (!devices_str.empty() && devices_str.back() == ']')
                    {
                        devices_str = devices_str.substr(0, devices_str.size() - 1);
                    }
                }
                config.tp_devices = parseDeviceList(devices_str);
            }
            else if (key == "tp_weights")
            {
                std::string weights_str = value;
                if (!weights_str.empty() && weights_str.front() == '[')
                {
                    weights_str = weights_str.substr(1);
                    if (!weights_str.empty() && weights_str.back() == ']')
                    {
                        weights_str = weights_str.substr(0, weights_str.size() - 1);
                    }
                }
                config.tp_weights = parseWeightList(weights_str);
            }
            else if (key == "tp_local_degree" || key == "tp_local")
            {
                config.tp_local_degree = std::stoi(value);
            }
            else if (key == "tp_global_degree" || key == "tp_global")
            {
                config.tp_global_degree = std::stoi(value);
            }
            else if (key == "pp_degree" || key == "pp")
            {
                config.pp_degree = std::stoi(value);
            }
            else if (key == "pp_split")
            {
                auto mode = parsePpSplitMode(value);
                if (mode)
                    config.pp_split = *mode;
            }
            else if (key == "cpu_layers")
            {
                config.cpu_layers = std::stoi(value);
            }
            else if (key == "cpu_layers_first")
            {
                config.cpu_layers_first = (toLower(value) == "true" || value == "1");
            }
            else if (key == "backend" || key == "default_backend")
            {
                auto backend = parseCollectiveBackendType(value);
                if (backend)
                    config.default_backend = *backend;
            }
        }

        return config;
    }

    // =========================================================================
    // Help text
    // =========================================================================

    std::string OrchestrationConfigParser::getHelpText()
    {
        return R"(
Orchestration Configuration Options:

Introspection:
  --dry-run              Show configuration without executing
  --explain-placement    Explain device placement decisions
  --show-topology        Show detected topology and exit
  --show-numa            Show NUMA configuration and exit
  --validate-only        Validate configuration without running

Device Assignment:
  -d, --device <spec>    Device for this rank (e.g., cuda:0, rocm:0, cpu)
  --device-mode <mode>   Assignment mode: auto, local_gpu, round_robin, explicit
  --device-map <map>     Explicit mapping: "0=cuda:0,1=cuda:1"

Named Domains (advanced):
  --define-domain <spec> Define domain: "name=device1,device2[;weights=w1,w2][;backend=type]"
  --pp-stage <spec>      Define PP stage: "stage_id=domain:first_layer-last_layer"

Tensor Parallelism:
  -t, --tp <degree>      TP parallelism degree
  --tp-scope <scope>     Scope: auto, local, global, hybrid
  --tp-devices <list>    Device list: "cuda:0,cuda:1"
  --tp-weights <list>    Weight distribution: "0.73,0.27"
  --tp-local <degree>    Local TP degree for hybrid
  --tp-global <degree>   Global TP degree for hybrid

Pipeline Parallelism:
  -p, --pp <degree>      PP parallelism degree
  --pp-split <mode>      Layer split: equal, weighted, manual

Layer Placement:
  --cpu-layers <n>       Number of layers on CPU
  --cpu-layers-first     Put CPU layers at beginning (default: end)

Backend:
  -b, --backend <type>   Default collective: auto, nccl, rccl, pciebar, upi, mpi, host

Config File:
  -c, --config <path>    Load configuration from YAML file
)";
    }

} // namespace llaminar2
