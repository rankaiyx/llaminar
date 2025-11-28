/**
 * @file ChatTemplate.h
 * @brief Chat template detection and formatting for LLM chat interfaces
 * @author David Sanftenberg
 * @date 2025
 *
 * Provides chat template support for formatting multi-turn conversations
 * according to model-specific formats (ChatML, LLaMA, Mistral, etc.).
 */

#pragma once

#include <string>
#include <vector>
#include <memory>

namespace llaminar2
{

    /**
     * @brief Represents a single message in a chat conversation
     */
    struct ChatMessage
    {
        std::string role;    ///< Role: "system", "user", "assistant"
        std::string content; ///< Message content

        ChatMessage() = default;
        ChatMessage(const std::string &r, const std::string &c)
            : role(r), content(c) {}
    };

    /**
     * @brief Enumeration of supported chat template types
     *
     * These correspond to different model families and their expected
     * prompt formats for instruction-following.
     */
    enum class ChatTemplateType
    {
        CHATML,     ///< ChatML format: <|im_start|>role\ncontent<|im_end|> (Qwen, many others)
        LLAMA3,     ///< LLaMA 3: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
        LLAMA2,     ///< LLaMA 2: [INST] <<SYS>>system<</SYS>> user [/INST] assistant
        MISTRAL_V1, ///< Mistral v1: [INST] content [/INST]
        MISTRAL_V3, ///< Mistral v3: Similar but with tool support
        MISTRAL_V7, ///< Mistral v7: [SYSTEM_PROMPT]...[/SYSTEM_PROMPT] [INST]...[/INST]
        PHI3,       ///< Phi-3: <|role|>\ncontent<|end|>
        PHI4,       ///< Phi-4: <|im_start|>role<|im_sep|>content<|im_end|>
        GEMMA,      ///< Gemma: <start_of_turn>role\ncontent<end_of_turn>
        DEEPSEEK,   ///< DeepSeek: ### Instruction:\ncontent
        DEEPSEEK2,  ///< DeepSeek V2: User: content\n\nAssistant:
        DEEPSEEK3,  ///< DeepSeek V3: <｜User｜>content<｜Assistant｜>
        ZEPHYR,     ///< Zephyr: <|role|>\ncontent<|endoftext|>
        VICUNA,     ///< Vicuna: USER: content\nASSISTANT:
        COMMAND_R,  ///< Command-R: <|START_OF_TURN_TOKEN|><|USER_TOKEN|>content
        UNKNOWN     ///< Unknown format - will use raw prompt
    };

    /**
     * @brief Converts ChatTemplateType to human-readable string
     */
    const char *chatTemplateTypeName(ChatTemplateType type);

    /**
     * @brief Chat template processor
     *
     * Handles detection of chat template format from GGUF metadata and
     * application of templates to message sequences.
     *
     * Usage:
     * @code
     *   auto tmpl = ChatTemplate::create(gguf_template_string);
     *   std::vector<ChatMessage> messages = {
     *       {"system", "You are helpful."},
     *       {"user", "Hello!"}
     *   };
     *   std::string formatted = tmpl->apply(messages, true);
     *   // formatted now contains the properly templated prompt
     * @endcode
     */
    class ChatTemplate
    {
    public:
        /**
         * @brief Create a ChatTemplate from a template string
         *
         * The template string is typically the `tokenizer.chat_template`
         * metadata value from a GGUF file (Jinja2 format).
         *
         * @param template_str The raw template string from GGUF metadata
         * @return Unique pointer to ChatTemplate, never null
         */
        static std::unique_ptr<ChatTemplate> create(const std::string &template_str);

        /**
         * @brief Create a ChatTemplate with explicit type override
         *
         * @param type The template type to use
         * @return Unique pointer to ChatTemplate
         */
        static std::unique_ptr<ChatTemplate> create(ChatTemplateType type);

        /**
         * @brief Destructor
         */
        ~ChatTemplate() = default;

        /**
         * @brief Apply template to a sequence of messages
         *
         * @param messages Vector of chat messages in conversation order
         * @param add_generation_prompt If true, adds the assistant prompt prefix
         *                              for the model to complete
         * @return Formatted prompt string ready for tokenization
         */
        std::string apply(const std::vector<ChatMessage> &messages,
                          bool add_generation_prompt = true) const;

        /**
         * @brief Get the detected template type
         */
        ChatTemplateType type() const { return type_; }

        /**
         * @brief Get the raw template string (if available)
         */
        const std::string &rawTemplate() const { return raw_template_; }

        /**
         * @brief Check if this is an unknown/fallback template
         */
        bool isUnknown() const { return type_ == ChatTemplateType::UNKNOWN; }

    private:
        /**
         * @brief Private constructor - use create() factory methods
         */
        ChatTemplate(ChatTemplateType type, const std::string &raw_template);

        /**
         * @brief Detect template type from Jinja2 template string
         *
         * Uses heuristic matching on template content to determine format.
         */
        static ChatTemplateType detectType(const std::string &template_str);

        // Template application methods for each format
        std::string applyChatML(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyLlama3(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyLlama2(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyMistralV1(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyMistralV3(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyMistralV7(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyPhi3(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyPhi4(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyGemma(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyDeepSeek(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyDeepSeek2(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyDeepSeek3(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyZephyr(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyVicuna(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyCommandR(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyUnknown(const std::vector<ChatMessage> &messages, bool add_ass) const;

        ChatTemplateType type_;
        std::string raw_template_;
    };

} // namespace llaminar2
