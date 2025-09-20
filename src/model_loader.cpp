#include "model_loader.h"
#include "graph_compute.h"
#include "tensors/tensor_factory.h"
#include "logger.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <numeric>

// GGUFValue template implementation
template <typename T>
T GGUFValue::as() const
{
    if (data.size() >= sizeof(T))
    {
        T value;
        std::memcpy(&value, data.data(), sizeof(T));
        return value;
    }
    return T{};
}

std::string GGUFValue::asString() const
{
    if (type == GGUFValueType::STRING && data.size() >= 8)
    {
        uint64_t length;
        std::memcpy(&length, data.data(), sizeof(uint64_t));

        if (data.size() >= 8 + length)
        {
            return std::string(reinterpret_cast<const char *>(data.data() + 8), length);
        }
    }
    return "";
}

std::vector<std::string> GGUFValue::asStringArray() const
{
    std::vector<std::string> result;
    if (type == GGUFValueType::ARRAY && data.size() >= 12)
    {
        uint32_t array_type;
        uint64_t array_length;
        std::memcpy(&array_type, data.data(), sizeof(uint32_t));
        std::memcpy(&array_length, data.data() + 4, sizeof(uint64_t));

        if (array_type == static_cast<uint32_t>(GGUFValueType::STRING))
        {
            size_t offset = 12;
            for (uint64_t i = 0; i < array_length && offset < data.size(); ++i)
            {
                if (offset + 8 <= data.size())
                {
                    uint64_t str_length;
                    std::memcpy(&str_length, data.data() + offset, sizeof(uint64_t));
                    offset += 8;

                    if (offset + str_length <= data.size())
                    {
                        result.emplace_back(reinterpret_cast<const char *>(data.data() + offset), str_length);
                        offset += str_length;
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }
    }
    return result;
}

// GGUFTensorInfo implementation
bool GGUFTensorInfo::isQuantized() const
{
    return type != GGUFTensorType::F32 && type != GGUFTensorType::F16;
}

size_t GGUFTensorInfo::getTypeSize() const
{
    switch (type)
    {
    case GGUFTensorType::F32:
        return 4;
    case GGUFTensorType::F16:
        return 2;
    case GGUFTensorType::Q8_0:
        return 34; // 32 values + 2 bytes metadata per block
    case GGUFTensorType::Q4_0:
        return 18; // 32 values in 16 bytes + 2 bytes metadata
    case GGUFTensorType::Q4_1:
        return 20; // 32 values in 16 bytes + 4 bytes metadata
    default:
        return 1;
    }
}

size_t GGUFTensorInfo::getBlockSize() const
{
    switch (type)
    {
    case GGUFTensorType::Q8_0:
    case GGUFTensorType::Q4_0:
    case GGUFTensorType::Q4_1:
        return 32; // 32 elements per quantization block
    default:
        return 1;
    }
}

// GGUFModel implementation
bool GGUFModel::hasMetadata(const std::string &key) const
{
    return metadata.find(key) != metadata.end();
}

template <typename T>
T GGUFModel::getMetadata(const std::string &key, T default_value) const
{
    auto it = metadata.find(key);
    if (it != metadata.end())
    {
        return it->second.as<T>();
    }
    return default_value;
}

GGUFTensorInfo *GGUFModel::findTensor(const std::string &name)
{
    for (auto &tensor : tensors)
    {
        if (tensor.name == name)
        {
            return &tensor;
        }
    }
    return nullptr;
}

const GGUFTensorInfo *GGUFModel::findTensor(const std::string &name) const
{
    for (const auto &tensor : tensors)
    {
        if (tensor.name == name)
        {
            return &tensor;
        }
    }
    return nullptr;
}

// ModelLoader implementation
ModelLoader::ModelLoader() : loaded_(false) {}

bool ModelLoader::loadModel(const std::string &file_path)
{
    LOG_TRACE("ModelLoader::loadModel starting for: " << file_path);
    file_path_ = file_path;
    loaded_ = false;

    file_stream_.open(file_path, std::ios::binary);
    if (!file_stream_.is_open())
    {
        std::cerr << "Error: Cannot open model file: " << file_path << std::endl;
        return false;
    }

    std::cout << "Loading GGUF model: " << file_path << std::endl;
    LOG_TRACE("File opened successfully, starting header parsing");

    LOG_TRACE("Calling parseHeader()");
    if (!parseHeader())
    {
        std::cerr << "Error: Failed to parse GGUF header" << std::endl;
        return false;
    }
    LOG_TRACE("parseHeader() completed successfully");

    LOG_TRACE("Calling parseMetadata()");
    if (!parseMetadata())
    {
        std::cerr << "Error: Failed to parse GGUF metadata" << std::endl;
        return false;
    }
    LOG_TRACE("parseMetadata() completed successfully");

    LOG_TRACE("Calling parseTensorInfo() for " << model_.tensor_count << " tensors");
    if (!parseTensorInfo())
    {
        std::cerr << "Error: Failed to parse GGUF tensor info" << std::endl;
        return false;
    }
    LOG_TRACE("parseTensorInfo() completed successfully");

    if (!validateModel())
    {
        std::cerr << "Error: Model validation failed" << std::endl;
        return false;
    }

    extractModelMetadata();

    loaded_ = true;
    std::cout << "Model loaded successfully!" << std::endl;
    return true;
}

std::shared_ptr<llaminar::TensorBase> ModelLoader::loadTensor(const std::string &tensor_name)
{
    if (!loaded_)
    {
        std::cerr << "Error: Model not loaded" << std::endl;
        return nullptr;
    }

    const auto *tensor_info = model_.findTensor(tensor_name);
    if (!tensor_info)
    {
        std::cerr << "Error: Tensor not found: " << tensor_name << std::endl;
        return nullptr;
    }

    // Seek to tensor data
    file_stream_.seekg(tensor_info->offset, std::ios::beg);

    // Read raw tensor data
    std::vector<uint8_t> raw_data(tensor_info->size_bytes);
    file_stream_.read(reinterpret_cast<char *>(raw_data.data()), tensor_info->size_bytes);

    if (file_stream_.gcount() != static_cast<std::streamsize>(tensor_info->size_bytes))
    {
        std::cerr << "Error: Failed to read tensor data for " << tensor_name << std::endl;
        return nullptr;
    }

    // Calculate total elements
    size_t total_elements = 1;
    for (uint64_t dim : tensor_info->dimensions)
    {
        total_elements *= dim;
    }

    // Dequantize if needed
    std::vector<float> float_data;
    if (tensor_info->isQuantized())
    {
        float_data = dequantizeTensor(*tensor_info, raw_data);
    }
    else
    {
        // Convert to float
        float_data.resize(total_elements);
        if (tensor_info->type == GGUFTensorType::F32)
        {
            std::memcpy(float_data.data(), raw_data.data(), total_elements * sizeof(float));
        }
        else if (tensor_info->type == GGUFTensorType::F16)
        {
            // TODO: Implement F16 dequantization
            LOG_WARN("F16 dequantization not implemented, using zeros");
            float_data.resize(total_elements, 0.0f);
        }
    }

    // Convert to float for our TensorBase class
    std::vector<float> float_data_converted(float_data.begin(), float_data.end());

    // Convert dimensions to int
    std::vector<int> int_dimensions;
    for (uint64_t dim : tensor_info->dimensions)
    {
        int_dimensions.push_back(static_cast<int>(dim));
    }

    return llaminar::TensorFactory::create_simple(int_dimensions, float_data_converted);
}

void ModelLoader::printModelInfo() const
{
    if (!loaded_)
    {
        std::cout << "Model not loaded" << std::endl;
        return;
    }

    std::cout << "\n=== Model Information ===" << std::endl;
    std::cout << "File: " << file_path_ << std::endl;
    std::cout << "GGUF Version: " << model_.version << std::endl;
    std::cout << "Architecture: " << model_.architecture << std::endl;
    std::cout << "Context Length: " << model_.context_length << std::endl;
    std::cout << "Embedding Length: " << model_.embedding_length << std::endl;
    std::cout << "Block Count: " << model_.block_count << std::endl;
    std::cout << "Head Count: " << model_.head_count << std::endl;
    std::cout << "Head Count KV: " << model_.head_count_kv << std::endl;
    std::cout << "Tensors: " << model_.tensor_count << std::endl;
    std::cout << "Metadata entries: " << model_.metadata_kv_count << std::endl;
    std::cout << "=========================" << std::endl;
}

bool ModelLoader::parseHeader()
{
    // Read GGUF magic number
    char magic[4];
    if (!file_stream_.read(magic, 4) || std::string(magic, 4) != "GGUF")
    {
        return false;
    }

    // Read version
    if (!readValue(model_.version))
        return false;

    // Read tensor count
    if (!readValue(model_.tensor_count))
        return false;

    // Read metadata count
    if (!readValue(model_.metadata_kv_count))
        return false;

    std::cout << "GGUF Header: version=" << model_.version
              << ", tensors=" << model_.tensor_count
              << ", metadata=" << model_.metadata_kv_count << std::endl;

    return true;
}

bool ModelLoader::parseMetadata()
{
    LOG_TRACE("parseMetadata: Starting, metadata_kv_count=" << model_.metadata_kv_count);
    for (uint64_t i = 0; i < model_.metadata_kv_count; ++i)
    {
        LOG_TRACE("parseMetadata: Processing metadata entry " << i << "/" << model_.metadata_kv_count);
        std::string key;
        if (!readString(key))
            return false;
        LOG_TRACE("parseMetadata: Read key: " << key);

        uint32_t value_type;
        if (!readValue(value_type))
            return false;
        LOG_TRACE("parseMetadata: Read value_type=" << value_type);

        GGUFValue value;
        value.type = static_cast<GGUFValueType>(value_type);
        LOG_TRACE("parseMetadata: Cast to GGUFValueType: " << static_cast<int>(value.type));

        if (value.type == GGUFValueType::ARRAY)
        {
            if (!readArray(value))
                return false;
        }
        else
        {
            // Read simple value based on type
            size_t value_size = 0;
            switch (value.type)
            {
            case GGUFValueType::UINT8:
                value_size = 1;
                break;
            case GGUFValueType::INT8:
                value_size = 1;
                break;
            case GGUFValueType::UINT16:
                value_size = 2;
                break;
            case GGUFValueType::INT16:
                value_size = 2;
                break;
            case GGUFValueType::UINT32:
                value_size = 4;
                break;
            case GGUFValueType::INT32:
                value_size = 4;
                break;
            case GGUFValueType::FLOAT32:
                value_size = 4;
                break;
            case GGUFValueType::UINT64:
                value_size = 8;
                break;
            case GGUFValueType::INT64:
                value_size = 8;
                break;
            case GGUFValueType::FLOAT64:
                value_size = 8;
                break;
            case GGUFValueType::BOOL:
                value_size = 1;
                break;
            case GGUFValueType::STRING:
            {
                uint64_t str_len;
                if (!readValue(str_len))
                    return false;
                LOG_TRACE("parseMetadata: STRING type, str_len=" << str_len);
                if (str_len > 1000000)
                { // 1MB sanity check
                    LOG_ERROR("parseMetadata: STRING length too large: " << str_len);
                    return false;
                }
                LOG_TRACE("parseMetadata: Attempting to resize data to " << (8 + str_len));
                value.data.resize(8 + str_len);
                LOG_TRACE("parseMetadata: Resize successful, copying data");
                std::memcpy(value.data.data(), &str_len, 8);
                if (!file_stream_.read(reinterpret_cast<char *>(value.data.data() + 8), str_len))
                {
                    return false;
                }
                break;
            }
            default:
                return false;
            }

            if (value.type != GGUFValueType::STRING)
            {
                LOG_TRACE("parseMetadata: Non-string type, value_size=" << value_size);
                if (value_size > 1000000)
                { // 1MB sanity check
                    LOG_ERROR("parseMetadata: Value size too large: " << value_size);
                    return false;
                }
                LOG_TRACE("parseMetadata: Attempting to resize data to " << value_size);
                value.data.resize(value_size);
                LOG_TRACE("parseMetadata: Resize successful, reading data");
                if (!file_stream_.read(reinterpret_cast<char *>(value.data.data()), value_size))
                {
                    return false;
                }
            }
        }

        model_.metadata[key] = std::move(value);
        LOG_TRACE("parseMetadata: Completed entry " << i << " for key: " << key);
    }
    LOG_TRACE("parseMetadata: All metadata processed successfully");

    // Extract and validate architecture
    auto arch_it = model_.metadata.find("general.architecture");
    if (arch_it != model_.metadata.end() && arch_it->second.type == GGUFValueType::STRING)
    {
        const auto &arch_data = arch_it->second.data;
        if (arch_data.size() >= 8) // String has 8-byte length prefix
        {
            uint64_t str_len;
            std::memcpy(&str_len, arch_data.data(), 8);
            if (arch_data.size() >= 8 + str_len)
            {
                model_.architecture = std::string(reinterpret_cast<const char *>(arch_data.data() + 8), str_len);
                LOG_INFO("parseMetadata: Extracted architecture: " << model_.architecture);
            }
        }
    }

    if (model_.architecture.empty())
    {
        LOG_WARN("parseMetadata: Architecture not found or could not be extracted from metadata");
    }

    return true;
}

bool ModelLoader::parseTensorInfo()
{
    LOG_TRACE("parseTensorInfo: Resizing tensors vector to " << model_.tensor_count);
    model_.tensors.resize(model_.tensor_count);
    LOG_TRACE("parseTensorInfo: Vector resized successfully");

    for (uint64_t i = 0; i < model_.tensor_count; ++i)
    {
        LOG_TRACE("parseTensorInfo: Processing tensor " << i << "/" << model_.tensor_count);
        auto &tensor = model_.tensors[i];

        // Read tensor name
        if (!readString(tensor.name))
            return false;
        LOG_TRACE("parseTensorInfo: Read tensor name: " << tensor.name);

        // Read number of dimensions
        uint32_t n_dims;
        if (!readValue(n_dims))
            return false;

        // Read dimensions
        tensor.dimensions.resize(n_dims);
        for (uint32_t j = 0; j < n_dims; ++j)
        {
            if (!readValue(tensor.dimensions[j]))
                return false;
        }

        // Read tensor type
        uint32_t type_val;
        if (!readValue(type_val))
            return false;
        tensor.type = static_cast<GGUFTensorType>(type_val);

        // Read tensor offset
        if (!readValue(tensor.offset))
            return false;

        // Calculate tensor size
        size_t total_elements = 1;
        for (uint64_t dim : tensor.dimensions)
        {
            total_elements *= dim;
        }

        if (tensor.isQuantized())
        {
            size_t block_size = tensor.getBlockSize();
            size_t type_size = tensor.getTypeSize();
            size_t num_blocks = (total_elements + block_size - 1) / block_size;
            tensor.size_bytes = num_blocks * type_size;
        }
        else
        {
            tensor.size_bytes = total_elements * tensor.getTypeSize();
        }
    }

    return true;
}

template <typename T>
bool ModelLoader::readValue(T &value)
{
    return static_cast<bool>(file_stream_.read(reinterpret_cast<char *>(&value), sizeof(T)));
}

bool ModelLoader::readString(std::string &str)
{
    uint64_t length;
    if (!readValue(length))
        return false;

    LOG_TRACE("readString: Read length=" << length);
    if (length > 10000000)
    { // 10MB sanity check for string length
        LOG_ERROR("readString: String length too large: " << length);
        return false;
    }

    LOG_TRACE("readString: Attempting to resize string to " << length);
    str.resize(length);
    LOG_TRACE("readString: Resize successful, reading string data");
    return static_cast<bool>(file_stream_.read(str.data(), length));
}

void ModelLoader::extractModelMetadata()
{
    model_.architecture = model_.hasMetadata("general.architecture") ? model_.metadata.at("general.architecture").asString() : "unknown";

    model_.context_length = model_.getMetadata<uint32_t>("qwen2.context_length", 0);
    model_.embedding_length = model_.getMetadata<uint32_t>("qwen2.embedding_length", 0);
    model_.block_count = model_.getMetadata<uint32_t>("qwen2.block_count", 0);
    model_.head_count = model_.getMetadata<uint32_t>("qwen2.attention.head_count", 0);
    model_.head_count_kv = model_.getMetadata<uint32_t>("qwen2.attention.head_count_kv", 0);

    if (model_.hasMetadata("tokenizer.ggml.tokens"))
    {
        model_.token_list = model_.metadata.at("tokenizer.ggml.tokens").asStringArray();
    }
}

std::vector<float> ModelLoader::dequantizeQ8_0(const uint8_t *data, size_t n_elements)
{
    std::vector<float> result(n_elements);

    const size_t block_size = 32;
    const size_t num_blocks = (n_elements + block_size - 1) / block_size;

    for (size_t block = 0; block < num_blocks; ++block)
    {
        const size_t block_offset = block * 34; // 32 values + 2 bytes scale

        // Read scale factor (fp16)
        uint16_t scale_bits;
        std::memcpy(&scale_bits, data + block_offset, 2);

        // Convert fp16 to fp32 (simplified)
        float scale = static_cast<float>(scale_bits) / 32768.0f;

        // Dequantize 32 int8 values
        for (size_t i = 0; i < block_size && (block * block_size + i) < n_elements; ++i)
        {
            int8_t quantized_val = static_cast<int8_t>(data[block_offset + 2 + i]);
            result[block * block_size + i] = scale * static_cast<float>(quantized_val);
        }
    }

    return result;
}

bool ModelLoader::validateModel()
{
    // Check if architecture is specified
    if (model_.architecture.empty())
    {
        std::cerr << "Error: No architecture specified" << std::endl;
        return false;
    }

    // Validate that we support this architecture
    if (model_.architecture != "qwen2")
    {
        std::cerr << "Error: Unsupported architecture '" << model_.architecture << "' (expected 'qwen2')" << std::endl;
        return false;
    }

    LOG_INFO("validateModel: Architecture '" << model_.architecture << "' is supported");

    if (model_.tensors.empty())
    {
        std::cerr << "Error: No tensors found" << std::endl;
        return false;
    }

    return true;
}

std::vector<float> ModelLoader::dequantizeTensor(const GGUFTensorInfo &tensor_info,
                                                 const std::vector<uint8_t> &quantized_data)
{
    switch (tensor_info.type)
    {
    case GGUFTensorType::Q8_0:
        return dequantizeQ8_0(quantized_data.data(),
                              std::accumulate(tensor_info.dimensions.begin(),
                                              tensor_info.dimensions.end(), 1ULL,
                                              std::multiplies<uint64_t>()));
    default:
        std::cerr << "Error: Unsupported quantization type for tensor " << tensor_info.name << std::endl;
        return {};
    }
}

TransformerLayerConfig ModelLoader::createLayerConfig() const
{
    if (!loaded_)
    {
        LOG_ERROR("Model not loaded");
        throw std::runtime_error("Model not loaded");
    }

    TransformerLayerConfig config{};

    // Extract architecture parameters from GGUF metadata
    config.n_layers = static_cast<int>(model_.block_count);
    config.n_head = static_cast<int>(model_.head_count);
    config.n_head_kv = static_cast<int>(model_.head_count_kv);
    config.d_model = static_cast<int>(model_.embedding_length);
    config.vocab_size = static_cast<int>(model_.token_list.size());
    config.max_seq_len = static_cast<int>(model_.context_length);

    // Calculate head dimension
    config.head_dim = config.d_model / config.n_head;

    // Extract feed-forward dimension (if available, otherwise estimate)
    config.d_ff = model_.getMetadata<uint32_t>("qwen2.feed_forward_length",
                                               static_cast<uint32_t>(config.d_model * 8 / 3)); // Common ratio for Qwen2.5

    // RMS norm epsilon (common value for Qwen2.5)
    config.eps = 1e-6f;

    LOG_INFO("Created LayerConfig: " << config.n_layers << " layers, "
                                     << config.n_head << " heads, " << config.d_model << " d_model, "
                                     << config.vocab_size << " vocab_size, " << config.max_seq_len << " max_seq_len");

    return config;
}

bool ModelLoader::readArray(GGUFValue &value)
{
    // Read array type and length
    uint32_t array_type;
    uint64_t array_length;
    if (!readValue(array_type) || !readValue(array_length))
    {
        return false;
    }

    LOG_TRACE("readArray: type=" << array_type << ", length=" << array_length);

    // Reserve space for the array data
    value.data.clear();
    value.data.reserve(12 + array_length * 20); // Estimate

    // Store array type and length at the beginning
    value.data.resize(12);
    std::memcpy(value.data.data(), &array_type, 4);
    std::memcpy(value.data.data() + 4, &array_length, 8);

    // Read array elements based on type
    GGUFValueType element_type = static_cast<GGUFValueType>(array_type);

    switch (element_type)
    {
    case GGUFValueType::STRING:
    {
        // Read string array - this is what we need for vocabulary
        for (uint64_t i = 0; i < array_length; ++i)
        {
            uint64_t str_len;
            if (!readValue(str_len))
            {
                return false;
            }

            // Store string length
            size_t len_offset = value.data.size();
            value.data.resize(value.data.size() + 8);
            std::memcpy(value.data.data() + len_offset, &str_len, 8);

            // Store string data
            size_t str_offset = value.data.size();
            value.data.resize(value.data.size() + str_len);
            if (!file_stream_.read(reinterpret_cast<char *>(value.data.data() + str_offset), str_len))
            {
                LOG_ERROR("readArray: Failed to read string data");
                return false;
            }
        }
        LOG_INFO("readArray: Successfully read string array with " << array_length << " elements");
        break;
    }

    case GGUFValueType::UINT32:
    case GGUFValueType::INT32:
    {
        // Read 32-bit integer array
        for (uint64_t i = 0; i < array_length; ++i)
        {
            uint32_t val;
            if (!readValue(val))
            {
                return false;
            }
            size_t offset = value.data.size();
            value.data.resize(value.data.size() + 4);
            std::memcpy(value.data.data() + offset, &val, 4);
        }
        LOG_INFO("readArray: Successfully read int32 array with " << array_length << " elements");
        break;
    }

    default:
    {
        // For other types, skip the data for now
        LOG_WARN("readArray: Skipping unsupported array type " << array_type);
        size_t element_size = 0;
        switch (element_type)
        {
        case GGUFValueType::UINT8:
        case GGUFValueType::INT8:
        case GGUFValueType::BOOL:
            element_size = 1;
            break;
        case GGUFValueType::UINT16:
        case GGUFValueType::INT16:
            element_size = 2;
            break;
        case GGUFValueType::FLOAT32:
            element_size = 4;
            break;
        case GGUFValueType::UINT64:
        case GGUFValueType::INT64:
        case GGUFValueType::FLOAT64:
            element_size = 8;
            break;
        default:
            LOG_ERROR("readArray: Unknown array element type: " << array_type);
            return false;
        }

        if (element_size > 0)
        {
            size_t bytes_to_skip = array_length * element_size;
            if (!file_stream_.seekg(bytes_to_skip, std::ios::cur))
            {
                LOG_ERROR("readArray: Failed to skip array data");
                return false;
            }
        }
        value.data.clear(); // Clear data since we didn't read it
        break;
    }
    }

    return true;
}

// Explicit template instantiations for commonly used types
template std::string GGUFModel::getMetadata<std::string>(const std::string &key, std::string default_value) const;
template uint32_t GGUFModel::getMetadata<uint32_t>(const std::string &key, uint32_t default_value) const;
template int32_t GGUFModel::getMetadata<int32_t>(const std::string &key, int32_t default_value) const;
template float GGUFModel::getMetadata<float>(const std::string &key, float default_value) const;

template int32_t GGUFValue::as<int32_t>() const;
template uint32_t GGUFValue::as<uint32_t>() const;
template float GGUFValue::as<float>() const;
template std::string GGUFValue::as<std::string>() const;