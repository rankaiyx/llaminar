/**
 * @file ParallelismTreeParser.h
 * @brief Parser for ParallelismTree from YAML or CLI strings
 *
 * Parses topology configuration in two formats:
 *
 * 1. YAML file format (primary for complex topologies):
 * @code yaml
 * topology:
 *   type: pp
 *   name: global
 *   children:
 *     - type: tp
 *       name: socket0_tp
 *       rank: 0
 *       backend: nccl
 *       devices: [cuda:0, cuda:1]
 *     - type: tp
 *       name: socket1_tp
 *       rank: 1
 *       backend: rccl
 *       devices: [rocm:0, rocm:1]
 * @endcode
 *
 * 2. Inline CLI format (for simple topologies):
 * @code
 * --topology "PP(global, TP(socket0, cuda:0, cuda:1), TP(socket1, rocm:0, rocm:1))"
 * @endcode
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "../execution/parallelism_tree/ParallelismTree.h"
#include <optional>
#include <string>
#include <vector>

namespace llaminar2 {

/**
 * @brief Parser for ParallelismTree from YAML or CLI strings
 */
class ParallelismTreeParser {
public:
    /**
     * @brief Parse result with tree or error messages
     */
    struct ParseResult {
        std::optional<ParallelismTree> tree;
        std::vector<std::string> errors;

        bool success() const { return tree.has_value() && errors.empty(); }
        std::string errorString() const;
    };

    /**
     * @brief Parse from YAML file
     *
     * @param yaml_path Path to YAML configuration file
     * @param total_layers Total model layers (for layer assignment)
     * @param world_size MPI world size
     * @return Parse result with tree or errors
     */
    static ParseResult parseFile(const std::string& yaml_path,
                                 int total_layers,
                                 int world_size);

    /**
     * @brief Parse from YAML string
     *
     * @param yaml_content YAML content as string
     * @param total_layers Total model layers
     * @param world_size MPI world size
     * @return Parse result
     */
    static ParseResult parseYAML(const std::string& yaml_content,
                                 int total_layers,
                                 int world_size);

    /**
     * @brief Parse from CLI inline string
     *
     * Format: "PP(name, child1, child2, ...)" or "TP(name, device1, device2, ...)"
     *
     * @param cli_topology CLI topology string
     * @param total_layers Total model layers
     * @param world_size MPI world size
     * @return Parse result
     */
    static ParseResult parseCLI(const std::string& cli_topology,
                                int total_layers,
                                int world_size);

    /**
     * @brief Convert tree back to YAML string (for round-trip testing)
     *
     * @param tree Parallelism tree to serialize
     * @return YAML string
     */
    static std::string toYAML(const ParallelismTree& tree);

private:
    // =========================================================================
    // Internal YAML Node Representation
    // =========================================================================

    /**
     * @brief Internal representation of a parsed YAML node
     *
     * Supports scalar values, lists (sequences), and maps (dicts).
     * This provides a yaml-cpp-like interface without the external dependency.
     */
    struct YAMLNode {
        enum class Type { NONE, SCALAR, LIST, MAP };

        Type type = Type::NONE;
        std::string scalar_value;
        std::vector<YAMLNode> list_items;
        std::vector<std::pair<std::string, YAMLNode>> map_entries;

        bool isNone() const { return type == Type::NONE; }
        bool isScalar() const { return type == Type::SCALAR; }
        bool isList() const { return type == Type::LIST; }
        bool isMap() const { return type == Type::MAP; }

        // Accessor for map by key
        const YAMLNode& operator[](const std::string& key) const;

        // Check if key exists in map
        bool hasKey(const std::string& key) const;

        // Get scalar as string
        std::string asString() const { return scalar_value; }

        // Get scalar as int
        int asInt() const;

        // Get scalar as float
        float asFloat() const;

        // Static none node for missing keys
        static const YAMLNode& none();
    };

    // =========================================================================
    // YAML Parsing Helpers
    // =========================================================================

    /**
     * @brief Tokenize YAML content into lines with indentation info
     */
    struct YAMLLine {
        int indent = 0;
        std::string key;
        std::string value;
        bool is_list_item = false;
        int line_number = 0;
    };

    static std::vector<YAMLLine> tokenizeYAML(const std::string& yaml_content);

    /**
     * @brief Parse a block of YAML lines into a YAMLNode
     */
    static YAMLNode parseYAMLBlock(const std::vector<YAMLLine>& lines,
                                   size_t& index,
                                   int base_indent);

    /**
     * @brief Parse a ParallelismNode from a YAMLNode
     */
    static ParallelismNode parseYAMLNode(const YAMLNode& yaml_node,
                                         int default_rank,
                                         std::vector<std::string>& errors);

    /**
     * @brief Parse a device address string
     */
    static std::optional<GlobalDeviceAddress> parseDevice(const std::string& device_str);

    /**
     * @brief Parse a backend type string
     */
    static std::optional<CollectiveBackendType> parseBackend(const std::string& backend_str);

    // =========================================================================
    // CLI Parsing Helpers
    // =========================================================================

    /**
     * @brief Parse a CLI node string recursively
     */
    static std::optional<ParallelismNode> parseCLINode(const std::string& node_str,
                                                       int default_rank,
                                                       std::vector<std::string>& errors);

    /**
     * @brief Tokenize CLI string respecting nested parentheses
     */
    static std::vector<std::string> tokenizeCLI(const std::string& input);

    /**
     * @brief Split string by delimiter, respecting parentheses nesting
     */
    static std::vector<std::string> splitRespectingParens(const std::string& input,
                                                          char delimiter);

    /**
     * @brief Trim whitespace from string
     */
    static std::string trim(const std::string& str);

    /**
     * @brief Convert string to lowercase
     */
    static std::string toLower(const std::string& str);

    // =========================================================================
    // YAML Serialization Helpers
    // =========================================================================

    /**
     * @brief Serialize a node to YAML with given indentation
     */
    static void nodeToYAML(std::ostringstream& oss,
                           const ParallelismNode& node,
                           int indent);
};

} // namespace llaminar2
