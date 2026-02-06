/**
 * @file ParallelismTreeParser.cpp
 * @brief Implementation of ParallelismTreeParser
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "ParallelismTreeParser.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace llaminar2 {

// =============================================================================
// ParseResult
// =============================================================================

std::string ParallelismTreeParser::ParseResult::errorString() const {
    if (errors.empty()) {
        return "";
    }
    std::ostringstream oss;
    for (size_t i = 0; i < errors.size(); ++i) {
        if (i > 0) {
            oss << "; ";
        }
        oss << errors[i];
    }
    return oss.str();
}

// =============================================================================
// YAMLNode Implementation
// =============================================================================

// Static none node for missing keys (thread-safe with C++11 magic statics)
const ParallelismTreeParser::YAMLNode& ParallelismTreeParser::YAMLNode::none() {
    static YAMLNode g_none_node;
    return g_none_node;
}

const ParallelismTreeParser::YAMLNode& ParallelismTreeParser::YAMLNode::operator[](
    const std::string& key) const {
    if (type != Type::MAP) {
        return none();
    }
    for (const auto& [k, v] : map_entries) {
        if (k == key) {
            return v;
        }
    }
    return none();
}

bool ParallelismTreeParser::YAMLNode::hasKey(const std::string& key) const {
    if (type != Type::MAP) {
        return false;
    }
    for (const auto& [k, _] : map_entries) {
        if (k == key) {
            return true;
        }
    }
    return false;
}

int ParallelismTreeParser::YAMLNode::asInt() const {
    try {
        return std::stoi(scalar_value);
    } catch (...) {
        return 0;
    }
}

float ParallelismTreeParser::YAMLNode::asFloat() const {
    try {
        return std::stof(scalar_value);
    } catch (...) {
        return 0.0f;
    }
}

// =============================================================================
// String Utilities
// =============================================================================

std::string ParallelismTreeParser::trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

std::string ParallelismTreeParser::toLower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

// =============================================================================
// YAML Tokenization
// =============================================================================

std::vector<ParallelismTreeParser::YAMLLine> ParallelismTreeParser::tokenizeYAML(
    const std::string& yaml_content) {
    std::vector<YAMLLine> lines;
    std::istringstream stream(yaml_content);
    std::string line;
    int line_number = 0;

    while (std::getline(stream, line)) {
        line_number++;

        // Skip empty lines and comments
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }

        YAMLLine yaml_line;
        yaml_line.line_number = line_number;

        // Calculate indentation (spaces only, tabs count as 1)
        size_t indent_end = 0;
        while (indent_end < line.size() && (line[indent_end] == ' ' || line[indent_end] == '\t')) {
            indent_end++;
        }
        yaml_line.indent = static_cast<int>(indent_end);

        // Check for list item
        if (trimmed.size() >= 2 && trimmed[0] == '-' && (trimmed[1] == ' ' || trimmed.size() == 1)) {
            yaml_line.is_list_item = true;
            trimmed = trim(trimmed.substr(trimmed[1] == ' ' ? 2 : 1));
            yaml_line.indent += 2; // List items add 2 to effective indent
        }

        // Parse key: value
        size_t colon_pos = trimmed.find(':');
        if (colon_pos != std::string::npos) {
            yaml_line.key = trim(trimmed.substr(0, colon_pos));
            std::string val = trimmed.substr(colon_pos + 1);
            yaml_line.value = trim(val);

            // Handle inline arrays [a, b, c]
            if (!yaml_line.value.empty() && yaml_line.value[0] == '[') {
                // Keep the brackets for array parsing
            }
        } else {
            // Bare value (for list items like "- cuda:0")
            yaml_line.value = trimmed;
        }

        lines.push_back(yaml_line);
    }

    return lines;
}

// =============================================================================
// YAML Block Parsing
// =============================================================================

ParallelismTreeParser::YAMLNode ParallelismTreeParser::parseYAMLBlock(
    const std::vector<YAMLLine>& lines,
    size_t& index,
    int base_indent) {
    YAMLNode node;

    if (index >= lines.size()) {
        return node;
    }

    const YAMLLine& first = lines[index];

    // Handle inline array [a, b, c]
    if (!first.value.empty() && first.value[0] == '[' && first.value.back() == ']') {
        node.type = YAMLNode::Type::LIST;
        std::string inner = first.value.substr(1, first.value.size() - 2);

        // Split by comma
        size_t start = 0;
        while (start < inner.size()) {
            size_t end = inner.find(',', start);
            if (end == std::string::npos) {
                end = inner.size();
            }
            std::string item = trim(inner.substr(start, end - start));
            if (!item.empty()) {
                YAMLNode scalar;
                scalar.type = YAMLNode::Type::SCALAR;
                scalar.scalar_value = item;
                node.list_items.push_back(scalar);
            }
            start = end + 1;
        }
        index++;
        return node;
    }

    // Handle scalar value on the same line as key
    if (!first.value.empty()) {
        node.type = YAMLNode::Type::SCALAR;
        // Remove quotes if present
        std::string val = first.value;
        if (val.size() >= 2 &&
            ((val.front() == '"' && val.back() == '"') ||
             (val.front() == '\'' && val.back() == '\''))) {
            val = val.substr(1, val.size() - 2);
        }
        node.scalar_value = val;
        index++;
        return node;
    }

    // Look ahead to determine if children are list or map
    if (index + 1 < lines.size()) {
        const YAMLLine& next = lines[index + 1];
        if (next.indent > first.indent) {
            if (next.is_list_item) {
                // Parse as list
                node.type = YAMLNode::Type::LIST;
                index++;

                while (index < lines.size()) {
                    const YAMLLine& item_line = lines[index];
                    if (item_line.indent < first.indent + 2) {
                        break; // End of list
                    }
                    if (!item_line.is_list_item && item_line.indent == first.indent + 2) {
                        // Not a list item at expected indent, done
                        break;
                    }

                    // Parse the list item (may be a nested map)
                    if (item_line.key.empty() && !item_line.value.empty()) {
                        // Simple scalar list item like "- cuda:0"
                        YAMLNode scalar;
                        scalar.type = YAMLNode::Type::SCALAR;
                        scalar.scalar_value = item_line.value;
                        node.list_items.push_back(scalar);
                        index++;
                    } else if (!item_line.key.empty()) {
                        // Map entry in list item
                        YAMLNode map_item;
                        map_item.type = YAMLNode::Type::MAP;

                        // First entry
                        map_item.map_entries.push_back({item_line.key,
                            parseYAMLBlock(lines, index, item_line.indent)});

                        // Continue parsing siblings at same indent
                        while (index < lines.size()) {
                            const YAMLLine& sibling = lines[index];
                            if (sibling.is_list_item || sibling.indent < item_line.indent) {
                                break;
                            }
                            if (sibling.indent == item_line.indent && !sibling.key.empty()) {
                                map_item.map_entries.push_back({sibling.key,
                                    parseYAMLBlock(lines, index, sibling.indent)});
                            } else {
                                break;
                            }
                        }
                        node.list_items.push_back(map_item);
                    } else {
                        index++;
                    }
                }
                return node;
            } else {
                // Parse as map
                node.type = YAMLNode::Type::MAP;
                index++;

                while (index < lines.size()) {
                    const YAMLLine& entry = lines[index];
                    if (entry.indent <= first.indent) {
                        break; // Back to parent level
                    }
                    if (!entry.key.empty()) {
                        node.map_entries.push_back({entry.key,
                            parseYAMLBlock(lines, index, entry.indent)});
                    } else {
                        index++;
                    }
                }
                return node;
            }
        }
    }

    // No children, treat empty value as empty scalar
    node.type = YAMLNode::Type::SCALAR;
    node.scalar_value = "";
    index++;
    return node;
}

// =============================================================================
// Device and Backend Parsing
// =============================================================================

std::optional<GlobalDeviceAddress> ParallelismTreeParser::parseDevice(
    const std::string& device_str) {
    return GlobalDeviceAddress::tryParse(trim(device_str));
}

std::optional<CollectiveBackendType> ParallelismTreeParser::parseBackend(
    const std::string& backend_str) {
    return parseCollectiveBackendType(toLower(trim(backend_str)));
}

// =============================================================================
// YAML Node to ParallelismNode
// =============================================================================

ParallelismNode ParallelismTreeParser::parseYAMLNode(
    const YAMLNode& yaml_node,
    int default_rank,
    std::vector<std::string>& errors) {
    ParallelismNode node;

    if (!yaml_node.isMap()) {
        errors.push_back("Expected a map node, got " +
            std::string(yaml_node.isList() ? "list" : "scalar"));
        return node;
    }

    // Parse type (required)
    if (!yaml_node.hasKey("type")) {
        errors.push_back("Missing required 'type' field");
        return node;
    }

    std::string type_str = toLower(yaml_node["type"].asString());
    if (type_str == "pp" || type_str == "pipeline_parallel") {
        node.type = ParallelismNodeType::PIPELINE_PARALLEL;
    } else if (type_str == "tp" || type_str == "tensor_parallel") {
        node.type = ParallelismNodeType::TENSOR_PARALLEL;
    } else if (type_str == "device") {
        node.type = ParallelismNodeType::DEVICE;
    } else {
        errors.push_back("Invalid type '" + type_str + "': must be 'pp', 'tp', or 'device'");
        return node;
    }

    // Parse name (required)
    if (!yaml_node.hasKey("name")) {
        errors.push_back("Missing required 'name' field");
        return node;
    }
    node.name = yaml_node["name"].asString();

    // Parse rank (optional, inherits from parent)
    if (yaml_node.hasKey("rank")) {
        node.owning_rank = yaml_node["rank"].asInt();
    } else {
        node.owning_rank = default_rank;
    }

    // Parse backend (optional, for TP nodes)
    if (yaml_node.hasKey("backend")) {
        auto backend = parseBackend(yaml_node["backend"].asString());
        if (backend) {
            node.backend = *backend;
        } else {
            errors.push_back("Invalid backend '" + yaml_node["backend"].asString() + "'");
        }
    }

    // Type-specific parsing
    if (node.type == ParallelismNodeType::DEVICE) {
        // Parse device address
        if (!yaml_node.hasKey("device")) {
            errors.push_back("DEVICE node '" + node.name + "' missing required 'device' field");
            return node;
        }
        auto device = parseDevice(yaml_node["device"].asString());
        if (device) {
            node.device = *device;
        } else {
            errors.push_back("Invalid device address '" +
                yaml_node["device"].asString() + "' in DEVICE node '" + node.name + "'");
        }
    } else if (node.type == ParallelismNodeType::TENSOR_PARALLEL) {
        // TP nodes can have either children or devices list
        if (yaml_node.hasKey("devices")) {
            const YAMLNode& devices_node = yaml_node["devices"];
            if (!devices_node.isList()) {
                errors.push_back("'devices' must be a list in TP node '" + node.name + "'");
            } else {
                for (const auto& device_item : devices_node.list_items) {
                    auto device = parseDevice(device_item.asString());
                    if (device) {
                        // Create a DEVICE child for each device
                        ParallelismNode device_child;
                        device_child.type = ParallelismNodeType::DEVICE;
                        device_child.name = device->toShortString();
                        device_child.device = *device;
                        device_child.owning_rank = node.owning_rank;
                        node.children.push_back(device_child);
                    } else {
                        errors.push_back("Invalid device '" + device_item.asString() +
                            "' in TP node '" + node.name + "'");
                    }
                }
            }
        }

        // Parse weights (optional)
        if (yaml_node.hasKey("weights")) {
            const YAMLNode& weights_node = yaml_node["weights"];
            if (weights_node.isList()) {
                for (const auto& w : weights_node.list_items) {
                    node.tp_weights.push_back(w.asFloat());
                }
            }
        }

        // Check for nested children (for complex TP trees)
        if (yaml_node.hasKey("children")) {
            const YAMLNode& children_node = yaml_node["children"];
            if (!children_node.isList()) {
                errors.push_back("'children' must be a list in TP node '" + node.name + "'");
            } else {
                for (const auto& child_yaml : children_node.list_items) {
                    node.children.push_back(parseYAMLNode(child_yaml, node.owning_rank, errors));
                }
            }
        }

        if (node.children.empty()) {
            errors.push_back("TP node '" + node.name + "' must have at least 2 devices or children");
        } else if (node.children.size() < 2) {
            errors.push_back("TP node '" + node.name + "' requires at least 2 children, got " +
                std::to_string(node.children.size()));
        }
    } else if (node.type == ParallelismNodeType::PIPELINE_PARALLEL) {
        // PP nodes require children
        if (!yaml_node.hasKey("children")) {
            errors.push_back("PP node '" + node.name + "' missing required 'children' field");
            return node;
        }

        const YAMLNode& children_node = yaml_node["children"];
        if (!children_node.isList()) {
            errors.push_back("'children' must be a list in PP node '" + node.name + "'");
        } else {
            for (const auto& child_yaml : children_node.list_items) {
                node.children.push_back(parseYAMLNode(child_yaml, node.owning_rank, errors));
            }
        }

        if (node.children.empty()) {
            errors.push_back("PP node '" + node.name + "' must have at least 1 child");
        }
    }

    return node;
}

// =============================================================================
// CLI Parsing
// =============================================================================

std::vector<std::string> ParallelismTreeParser::splitRespectingParens(
    const std::string& input, char delimiter) {
    std::vector<std::string> result;
    int paren_depth = 0;
    size_t start = 0;

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        if (c == '(') {
            paren_depth++;
        } else if (c == ')') {
            paren_depth--;
        } else if (c == delimiter && paren_depth == 0) {
            std::string part = trim(input.substr(start, i - start));
            if (!part.empty()) {
                result.push_back(part);
            }
            start = i + 1;
        }
    }

    // Last part
    std::string last = trim(input.substr(start));
    if (!last.empty()) {
        result.push_back(last);
    }

    return result;
}

std::vector<std::string> ParallelismTreeParser::tokenizeCLI(const std::string& input) {
    // Split by top-level commas, respecting nested parentheses
    return splitRespectingParens(input, ',');
}

std::optional<ParallelismNode> ParallelismTreeParser::parseCLINode(
    const std::string& node_str,
    int default_rank,
    std::vector<std::string>& errors) {
    std::string trimmed = trim(node_str);

    // Check for PP(...) or TP(...) pattern
    bool is_pp = (trimmed.size() > 3 && toLower(trimmed.substr(0, 3)) == "pp(" && trimmed.back() == ')');
    bool is_tp = (trimmed.size() > 3 && toLower(trimmed.substr(0, 3)) == "tp(" && trimmed.back() == ')');

    if (!is_pp && !is_tp) {
        // Try to parse as a bare device address (leaf node)
        auto device = parseDevice(trimmed);
        if (device) {
            ParallelismNode node;
            node.type = ParallelismNodeType::DEVICE;
            node.name = device->toShortString();
            node.device = *device;
            node.owning_rank = default_rank;
            return node;
        }
        errors.push_back("Invalid CLI node: '" + trimmed + "' (expected PP(...), TP(...), or device)");
        return std::nullopt;
    }

    // Extract content inside parentheses
    size_t open_paren = trimmed.find('(');
    std::string inner = trimmed.substr(open_paren + 1, trimmed.size() - open_paren - 2);
    std::vector<std::string> parts = tokenizeCLI(inner);

    if (parts.empty()) {
        errors.push_back((is_pp ? "PP" : "TP") + std::string("() requires at least a name"));
        return std::nullopt;
    }

    ParallelismNode node;
    node.type = is_pp ? ParallelismNodeType::PIPELINE_PARALLEL : ParallelismNodeType::TENSOR_PARALLEL;

    // First part is always the name
    node.name = parts[0];
    node.owning_rank = default_rank;

    // Parse remaining parts
    size_t child_start = 1;
    for (size_t i = 1; i < parts.size(); ++i) {
        std::string part = parts[i];

        // Check for option=value patterns
        if (part.find('=') != std::string::npos) {
            size_t eq_pos = part.find('=');
            std::string option = toLower(trim(part.substr(0, eq_pos)));
            std::string value = trim(part.substr(eq_pos + 1));

            if (option == "rank") {
                try {
                    node.owning_rank = std::stoi(value);
                } catch (...) {
                    errors.push_back("Invalid rank value: '" + value + "'");
                }
            } else if (option == "backend") {
                auto backend = parseBackend(value);
                if (backend) {
                    node.backend = *backend;
                } else {
                    errors.push_back("Invalid backend: '" + value + "'");
                }
            } else {
                errors.push_back("Unknown option: '" + option + "'");
            }
            child_start = i + 1;
        } else {
            // Not an option, remaining parts are children/devices
            break;
        }
    }

    // Parse children/devices
    for (size_t i = child_start; i < parts.size(); ++i) {
        std::string part = parts[i];

        // Check if it's a nested node (PP/TP call)
        if ((toLower(part.substr(0, 3)) == "pp(" || toLower(part.substr(0, 3)) == "tp(") 
            && part.back() == ')') {
            auto child = parseCLINode(part, node.owning_rank, errors);
            if (child) {
                node.children.push_back(*child);
            }
        } else {
            // Try to parse as device
            auto device = parseDevice(part);
            if (device) {
                ParallelismNode child;
                child.type = ParallelismNodeType::DEVICE;
                child.name = device->toShortString();
                child.device = *device;
                child.owning_rank = node.owning_rank;
                node.children.push_back(child);
            } else {
                errors.push_back("Invalid device or child: '" + part + "'");
            }
        }
    }

    // Validation
    if (is_tp && node.children.size() < 2) {
        errors.push_back("TP node '" + node.name + "' requires at least 2 children, got " +
            std::to_string(node.children.size()));
    }

    return node;
}

// =============================================================================
// Public API: parseFile
// =============================================================================

ParallelismTreeParser::ParseResult ParallelismTreeParser::parseFile(
    const std::string& yaml_path,
    int total_layers,
    int world_size) {
    ParseResult result;

    std::ifstream file(yaml_path);
    if (!file.is_open()) {
        result.errors.push_back("Failed to open file: " + yaml_path);
        return result;
    }

    std::ostringstream oss;
    oss << file.rdbuf();
    return parseYAML(oss.str(), total_layers, world_size);
}

// =============================================================================
// Public API: parseYAML
// =============================================================================

ParallelismTreeParser::ParseResult ParallelismTreeParser::parseYAML(
    const std::string& yaml_content,
    int total_layers,
    int world_size) {
    ParseResult result;

    // Tokenize
    std::vector<YAMLLine> lines = tokenizeYAML(yaml_content);
    if (lines.empty()) {
        result.errors.push_back("Empty YAML content");
        return result;
    }

    // Find topology key
    size_t index = 0;
    YAMLNode root;
    root.type = YAMLNode::Type::MAP;

    while (index < lines.size()) {
        const YAMLLine& line = lines[index];
        if (line.indent == 0 && !line.key.empty()) {
            root.map_entries.push_back({line.key, parseYAMLBlock(lines, index, line.indent)});
        } else {
            index++;
        }
    }

    // Look for 'topology' key
    if (!root.hasKey("topology")) {
        result.errors.push_back("Missing 'topology' key in YAML");
        return result;
    }

    const YAMLNode& topology_node = root["topology"];

    // Parse the root node
    ParallelismNode root_node = parseYAMLNode(topology_node, 0, result.errors);

    if (!result.errors.empty()) {
        return result;
    }

    // Create tree
    ParallelismTree tree;
    tree.root = std::move(root_node);
    tree.total_layers = total_layers;
    tree.world_size = world_size;

    // Assign layers
    tree.assignLayers(total_layers);

    // Validate
    auto validation_errors = tree.validate();
    result.errors.insert(result.errors.end(), validation_errors.begin(), validation_errors.end());

    if (result.errors.empty()) {
        result.tree = std::move(tree);
    }

    return result;
}

// =============================================================================
// Public API: parseCLI
// =============================================================================

ParallelismTreeParser::ParseResult ParallelismTreeParser::parseCLI(
    const std::string& cli_topology,
    int total_layers,
    int world_size) {
    ParseResult result;

    std::string trimmed = trim(cli_topology);
    if (trimmed.empty()) {
        result.errors.push_back("Empty CLI topology string");
        return result;
    }

    // Check for balanced parentheses
    int paren_count = 0;
    for (char c : trimmed) {
        if (c == '(') paren_count++;
        else if (c == ')') paren_count--;
        if (paren_count < 0) {
            result.errors.push_back("Unmatched ')' in CLI topology");
            return result;
        }
    }
    if (paren_count != 0) {
        result.errors.push_back("Unmatched '(' in CLI topology");
        return result;
    }

    // Parse the root node
    auto root_node = parseCLINode(trimmed, 0, result.errors);

    if (!root_node || !result.errors.empty()) {
        return result;
    }

    // Create tree
    ParallelismTree tree;
    tree.root = std::move(*root_node);
    tree.total_layers = total_layers;
    tree.world_size = world_size;

    // Assign layers
    tree.assignLayers(total_layers);

    // Validate
    auto validation_errors = tree.validate();
    result.errors.insert(result.errors.end(), validation_errors.begin(), validation_errors.end());

    if (result.errors.empty()) {
        result.tree = std::move(tree);
    }

    return result;
}

// =============================================================================
// Public API: toYAML
// =============================================================================

void ParallelismTreeParser::nodeToYAML(
    std::ostringstream& oss,
    const ParallelismNode& node,
    int indent) {
    std::string pad(indent, ' ');

    // Type
    oss << pad << "type: " << parallelismNodeTypeName(node.type) << "\n";

    // Name
    oss << pad << "name: " << node.name << "\n";

    // Rank (only for leaves or explicit overrides)
    if (node.owning_rank >= 0) {
        oss << pad << "rank: " << node.owning_rank << "\n";
    }

    // Backend (only for TP with non-AUTO)
    if (node.type == ParallelismNodeType::TENSOR_PARALLEL &&
        node.backend != CollectiveBackendType::AUTO) {
        oss << pad << "backend: " << collectiveBackendTypeToString(node.backend) << "\n";
    }

    // Device (for DEVICE nodes)
    if (node.type == ParallelismNodeType::DEVICE) {
        oss << pad << "device: " << node.device.toShortString() << "\n";
    }

    // TP weights
    if (!node.tp_weights.empty()) {
        oss << pad << "weights: [";
        for (size_t i = 0; i < node.tp_weights.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << node.tp_weights[i];
        }
        oss << "]\n";
    }

    // Children
    if (!node.children.empty()) {
        // For TP nodes with only DEVICE children, use compact devices format
        bool all_device_children = std::all_of(node.children.begin(), node.children.end(),
            [](const ParallelismNode& child) {
                return child.type == ParallelismNodeType::DEVICE;
            });

        if (node.type == ParallelismNodeType::TENSOR_PARALLEL && all_device_children) {
            oss << pad << "devices: [";
            for (size_t i = 0; i < node.children.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << node.children[i].device.toShortString();
            }
            oss << "]\n";
        } else {
            oss << pad << "children:\n";
            for (const auto& child : node.children) {
                oss << pad << "  - ";
                // Inline first line, then continue with proper indent
                std::ostringstream child_oss;
                nodeToYAML(child_oss, child, 0);
                std::string child_yaml = child_oss.str();

                // Split into lines and add proper indentation
                std::istringstream iss(child_yaml);
                std::string line;
                bool first = true;
                while (std::getline(iss, line)) {
                    if (first) {
                        oss << line << "\n";
                        first = false;
                    } else {
                        oss << pad << "    " << line << "\n";
                    }
                }
            }
        }
    }
}

std::string ParallelismTreeParser::toYAML(const ParallelismTree& tree) {
    std::ostringstream oss;
    oss << "topology:\n";
    nodeToYAML(oss, tree.root, 2);
    return oss.str();
}

} // namespace llaminar2
