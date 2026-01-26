/**
 * @file OrchestrationConfigParser.h
 * @brief Implementation of IOrchestrationConfigParser
 *
 * Parses orchestration configuration from CLI arguments and YAML files.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "IOrchestrationConfigParser.h"
#include <string>
#include <vector>
#include <functional>

namespace llaminar2
{

    /**
     * @brief Default implementation of IOrchestrationConfigParser
     *
     * Parses orchestration configuration from:
     * - Command line arguments
     * - YAML configuration files
     * - YAML strings (for testing)
     */
    class OrchestrationConfigParser : public IOrchestrationConfigParser
    {
    public:
        /**
         * @brief Construct parser with default settings
         */
        OrchestrationConfigParser() = default;

        /**
         * @brief Parse from command line arguments
         */
        OrchestrationConfig parseArgs(int argc, char **argv) override;

        /**
         * @brief Parse from YAML file
         */
        OrchestrationConfig parseYamlFile(const std::string &path) override;

        /**
         * @brief Parse from YAML string
         */
        OrchestrationConfig parseYamlString(const std::string &yaml) override;

        /**
         * @brief Get help text for CLI options
         * @return Formatted help string
         */
        static std::string getHelpText();

    private:
        /**
         * @brief Parse device list from comma-separated string
         * @param spec Comma-separated device specs (e.g., "cuda:0,cuda:1")
         * @return Vector of parsed device addresses
         */
        static std::vector<GlobalDeviceAddress> parseDeviceList(const std::string &spec);

        /**
         * @brief Parse weight list from comma-separated string
         * @param spec Comma-separated weights (e.g., "0.73,0.27")
         * @return Vector of parsed weights
         */
        static std::vector<float> parseWeightList(const std::string &spec);

        /**
         * @brief Parse device map from string
         * @param spec Format: "rank=device,rank=device,..." (e.g., "0=cuda:0,1=cuda:1")
         * @return Vector of (rank, device) pairs
         */
        static std::vector<std::pair<int, GlobalDeviceAddress>> parseDeviceMap(const std::string &spec);

        /**
         * @brief Check if argument matches a flag
         * @param arg Argument to check
         * @param short_form Short flag (e.g., "-t")
         * @param long_form Long flag (e.g., "--tp")
         * @return true if argument matches
         */
        static bool matchesFlag(const std::string &arg,
                                const std::string &short_form,
                                const std::string &long_form);

        /**
         * @brief Get value for a flag (handles --flag=value and --flag value)
         * @param args All arguments
         * @param idx Current index (will be incremented if value follows flag)
         * @return Value string, or empty if not found
         */
        static std::string getFlagValue(const std::vector<std::string> &args, size_t &idx);
    };

} // namespace llaminar2
