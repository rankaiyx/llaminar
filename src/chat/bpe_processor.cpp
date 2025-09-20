#include "bpe_processor.h"
#include "../logger.h"
#include <regex>
#include <algorithm>
#include <sstream>
#include <cctype>

namespace llaminar
{
    namespace chat
    {
        void BPEProcessor::initialize(const std::unordered_map<std::string, int32_t> &token_to_id,
                                      const std::vector<std::string> &id_to_token)
        {
            LOG_DEBUG("Initializing BPE processor with " << token_to_id.size() << " tokens");

            // Initialize byte-level mappings
            initializeByteMapping();

            // Extract BPE merges from vocabulary
            extractBPEMerges(token_to_id);

            LOG_INFO("BPE processor initialized with " << bpe_merges_.size() << " merge rules");
        }

        std::vector<int32_t> BPEProcessor::tokenize(const std::string &text,
                                                    const std::unordered_map<std::string, int32_t> &token_to_id,
                                                    int32_t unk_token_id)
        {
            if (text.empty())
            {
                return {};
            }

            std::vector<int32_t> tokens;

            // Convert text to byte-level representation
            std::string byte_text = textToBytes(text);

            // Split into words (preserving whitespace information)
            std::vector<std::string> words = splitIntoWords(byte_text);

            for (const std::string &word : words)
            {
                if (word.empty())
                    continue;

                // Convert word to initial character sequence
                std::vector<std::string> word_tokens;
                for (size_t i = 0; i < word.length();)
                {
                    // Handle UTF-8 sequences by taking single bytes in byte-level BPE
                    word_tokens.push_back(word.substr(i, 1));
                    ++i;
                }

                // Apply BPE merges
                word_tokens = applyBPE(word_tokens);

                // Convert tokens to IDs
                for (const std::string &token : word_tokens)
                {
                    auto it = token_to_id.find(token);
                    if (it != token_to_id.end())
                    {
                        tokens.push_back(it->second);
                    }
                    else if (unk_token_id != -1)
                    {
                        tokens.push_back(unk_token_id);
                    }
                    // Skip unknown tokens if no UNK token available
                }
            }

            return tokens;
        }

        std::string BPEProcessor::detokenize(const std::vector<int32_t> &tokens,
                                             const std::vector<std::string> &id_to_token)
        {
            if (tokens.empty())
            {
                return "";
            }

            std::ostringstream result;

            for (int32_t token_id : tokens)
            {
                if (token_id >= 0 && token_id < static_cast<int32_t>(id_to_token.size()))
                {
                    result << id_to_token[token_id];
                }
                // Skip invalid token IDs
            }

            // Convert from byte-level representation back to text
            return bytesToText(result.str());
        }

        void BPEProcessor::initializeByteMapping()
        {
            // Standard GPT-2 style byte-level BPE mapping
            // Maps bytes 0-255 to Unicode characters that are safe for BPE

            // Printable ASCII characters map to themselves
            for (int i = 33; i <= 126; ++i)
            {
                if (i != 92)
                { // Skip backslash
                    byte_to_char_[i] = std::string(1, static_cast<char>(i));
                    char_to_byte_[std::string(1, static_cast<char>(i))] = static_cast<uint8_t>(i);
                }
            }

            // Space maps to itself
            byte_to_char_[32] = " ";
            char_to_byte_[" "] = 32;

            // Map remaining bytes to Unicode private use area
            int unicode_start = 256;
            for (int i = 0; i <= 255; ++i)
            {
                if (byte_to_char_.find(i) == byte_to_char_.end())
                {
                    // Use Unicode private use characters for non-printable bytes
                    std::string unicode_char;
                    int unicode_point = unicode_start++;

                    // Simple single-byte mapping for now
                    // In a full implementation, would use proper Unicode encoding
                    unicode_char = "Ġ" + std::to_string(i); // Simple placeholder

                    byte_to_char_[i] = unicode_char;
                    char_to_byte_[unicode_char] = static_cast<uint8_t>(i);
                }
            }
        }

        void BPEProcessor::extractBPEMerges(const std::unordered_map<std::string, int32_t> &token_to_id)
        {
            // Extract merge rules by analyzing the vocabulary
            // This is a simplified approach - in practice, merge rules are usually provided separately

            std::vector<std::string> all_tokens;
            all_tokens.reserve(token_to_id.size());

            for (const auto &pair : token_to_id)
            {
                all_tokens.push_back(pair.first);
            }

            // Sort tokens by ID to maintain merge order
            std::sort(all_tokens.begin(), all_tokens.end(),
                      [&token_to_id](const std::string &a, const std::string &b)
                      {
                          return token_to_id.at(a) < token_to_id.at(b);
                      });

            // Extract potential merges from multi-character tokens
            int32_t merge_rank = 0;
            for (const std::string &token : all_tokens)
            {
                if (token.length() > 1)
                {
                    // Try to split token into existing smaller tokens
                    for (size_t split = 1; split < token.length(); ++split)
                    {
                        std::string left = token.substr(0, split);
                        std::string right = token.substr(split);

                        // Check if both parts exist in vocabulary
                        if (token_to_id.find(left) != token_to_id.end() &&
                            token_to_id.find(right) != token_to_id.end())
                        {

                            std::pair<std::string, std::string> merge_pair = {left, right};

                            // Add merge rule if not already present
                            if (merge_ranks_.find(merge_pair) == merge_ranks_.end())
                            {
                                bpe_merges_.push_back(merge_pair);
                                merge_ranks_[merge_pair] = merge_rank++;
                            }
                            break; // Use first valid split
                        }
                    }
                }
            }

            LOG_DEBUG("Extracted " << bpe_merges_.size() << " BPE merge rules");
        }

        std::string BPEProcessor::textToBytes(const std::string &text)
        {
            std::string result;
            result.reserve(text.length() * 2); // Reserve space for potential expansion

            for (unsigned char byte : text)
            {
                auto it = byte_to_char_.find(byte);
                if (it != byte_to_char_.end())
                {
                    result += it->second;
                }
                else
                {
                    // Fallback for unmapped bytes
                    result += "Ġ" + std::to_string(byte);
                }
            }

            return result;
        }

        std::string BPEProcessor::bytesToText(const std::string &byte_text)
        {
            std::string result;
            result.reserve(byte_text.length());

            for (size_t i = 0; i < byte_text.length();)
            {
                // Look for the longest match in char_to_byte mapping
                bool found = false;
                for (size_t len = std::min(byte_text.length() - i, size_t(10)); len > 0; --len)
                {
                    std::string substr = byte_text.substr(i, len);
                    auto it = char_to_byte_.find(substr);
                    if (it != char_to_byte_.end())
                    {
                        result += static_cast<char>(it->second);
                        i += len;
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    // Handle unmapped characters - try to decode placeholder format
                    if (i + 1 < byte_text.length() && byte_text.substr(i, 1) == "Ġ")
                    {
                        // Try to parse "Ġ123" format
                        size_t num_start = i + 1;
                        size_t num_end = num_start;
                        while (num_end < byte_text.length() && std::isdigit(byte_text[num_end]))
                        {
                            ++num_end;
                        }

                        if (num_end > num_start)
                        {
                            int byte_val = std::stoi(byte_text.substr(num_start, num_end - num_start));
                            if (byte_val >= 0 && byte_val <= 255)
                            {
                                result += static_cast<char>(byte_val);
                                i = num_end;
                                continue;
                            }
                        }
                    }

                    // Fallback: copy character as-is
                    result += byte_text[i];
                    ++i;
                }
            }

            return result;
        }

        std::vector<std::string> BPEProcessor::applyBPE(std::vector<std::string> word)
        {
            if (word.size() <= 1)
            {
                return word;
            }

            while (true)
            {
                auto pairs = getPairs(word);
                if (pairs.empty())
                {
                    break;
                }

                // Find the pair with the highest priority (lowest rank)
                std::pair<std::string, std::string> best_pair;
                int32_t best_rank = std::numeric_limits<int32_t>::max();
                bool found_pair = false;

                for (const auto &pair : pairs)
                {
                    auto it = merge_ranks_.find(pair);
                    if (it != merge_ranks_.end() && it->second < best_rank)
                    {
                        best_pair = pair;
                        best_rank = it->second;
                        found_pair = true;
                    }
                }

                if (!found_pair)
                {
                    break;
                }

                // Apply the merge
                std::vector<std::string> new_word;
                for (size_t i = 0; i < word.size();)
                {
                    if (i + 1 < word.size() &&
                        word[i] == best_pair.first &&
                        word[i + 1] == best_pair.second)
                    {
                        // Merge this pair
                        new_word.push_back(word[i] + word[i + 1]);
                        i += 2;
                    }
                    else
                    {
                        new_word.push_back(word[i]);
                        ++i;
                    }
                }

                word = std::move(new_word);
            }

            return word;
        }

        std::unordered_set<std::pair<std::string, std::string>, BPEProcessor::PairHash>
        BPEProcessor::getPairs(const std::vector<std::string> &word)
        {
            std::unordered_set<std::pair<std::string, std::string>, PairHash> pairs;

            for (size_t i = 0; i + 1 < word.size(); ++i)
            {
                pairs.insert({word[i], word[i + 1]});
            }

            return pairs;
        }

        std::vector<std::string> BPEProcessor::splitIntoWords(const std::string &text)
        {
            std::vector<std::string> words;

            // Use regex to split on whitespace while preserving the whitespace
            // This is a simplified version - GPT-2 uses a more complex regex
            std::regex word_regex(R"([a-zA-Z]+|[0-9]+|[^\w\s]+|\s+)");
            std::sregex_iterator iter(text.begin(), text.end(), word_regex);
            std::sregex_iterator end;

            for (; iter != end; ++iter)
            {
                std::string word = iter->str();
                if (!word.empty())
                {
                    words.push_back(word);
                }
            }

            // Fallback: if regex failed, split on spaces
            if (words.empty() && !text.empty())
            {
                std::istringstream iss(text);
                std::string word;
                while (iss >> word)
                {
                    words.push_back(word);
                }
            }

            return words;
        }

        bool BPEProcessor::isWhitespace(char c)
        {
            return std::isspace(static_cast<unsigned char>(c));
        }

    } // namespace chat
} // namespace llaminar