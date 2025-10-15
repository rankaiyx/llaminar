"""
GGUF Tokenizer Loader

Loads tokenizer directly from GGUF metadata to ensure vocabulary matches the model.
This is critical for models with custom vocabularies (e.g., Gemini-Distill variants).

Supports:
- GPT2/BPE tokenization (qwen2.pre)
- Special tokens (BOS, EOS, PAD)
- Chat templates

Author: David Sanftenberg
"""

import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class GGUFTokenizer:
    """
    Tokenizer loaded from GGUF file metadata.
    
    Implements BPE (Byte Pair Encoding) tokenization compatible with GPT2/Qwen models.
    
    Args:
        tokens: List of token strings from GGUF metadata
        merges: List of BPE merge rules (e.g., ["a b", "c d", ...])
        bos_token_id: Begin-of-sequence token ID
        eos_token_id: End-of-sequence token ID
        pad_token_id: Padding token ID (optional)
        
    Example:
        from python.reference.loaders.gguf_parser import GGUFParser
        from python.reference.loaders.gguf_tokenizer import GGUFTokenizer
        
        parser = GGUFParser("model.gguf")
        parser.parse()
        
        tokenizer = GGUFTokenizer.from_gguf_metadata(parser.metadata)
        tokens = tokenizer.encode("Hello, world!")
        text = tokenizer.decode(tokens)
    """
    
    def __init__(
        self,
        tokens: List[str],
        merges: List[str],
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: Optional[int] = None,
        add_bos_token: bool = False
    ):
        self.tokens = tokens
        self.vocab_size = len(tokens)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else eos_token_id
        self.add_bos_token = add_bos_token
        
        # Build token to ID mapping
        self.token_to_id = {token: i for i, token in enumerate(tokens)}
        
        # Parse BPE merges
        self.bpe_ranks = {}
        for i, merge in enumerate(merges):
            parts = merge.split()
            if len(parts) == 2:
                self.bpe_ranks[(parts[0], parts[1])] = i
        
        # Byte encoder/decoder for handling UTF-8
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # Note: Full BPE tokenization requires complex regex and merge logic
        # For now, we provide basic encode/decode using vocab lookup
        # This is sufficient for testing as PyTorch's generate() only needs decode()
    
    @staticmethod
    def _bytes_to_unicode():
        """
        Create mapping from bytes to unicode characters (GPT2 byte encoder).
        """
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(c) for c in cs]
        return dict(zip(bs, cs))
    
    def _get_pairs(self, word: Tuple[str, ...]) -> set:
        """Get all adjacent pairs of characters in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _bpe(self, token: str) -> str:
        """
        Apply BPE merges to a token.
        
        Args:
            token: Single token string (already byte-encoded)
            
        Returns:
            Space-separated BPE subwords
        """
        if token in self.token_to_id:
            return token
        
        word = tuple(token)
        pairs = self._get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            # Find the pair with lowest merge rank
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                new_word.extend(word[i:j])
                i = j
                
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = tuple(new_word)
            if len(word) == 1:
                break
            
            pairs = self._get_pairs(word)
        
        return ' '.join(word)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        NOTE: This is a simplified implementation that uses HuggingFace tokenizer
        as a fallback. Full BPE encoding from scratch is complex and not needed
        for parity testing (we only need decode() for output comparison).
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS token (if configured)
            
        Returns:
            List of token IDs
            
        Raises:
            NotImplementedError: Full BPE encoding not yet implemented
        """
        # For parity testing, we don't actually need encode() - PyTorch model.generate()
        # only requires that we can decode() the output tokens back to text.
        # The input tokenization can still use HuggingFace tokenizer.
        raise NotImplementedError(
            "GGUF tokenizer encode() not yet implemented. "
            "For inference testing, use HuggingFace tokenizer for input, "
            "and GGUF tokenizer only for decode()."
        )
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip BOS/EOS/PAD tokens
            
        Returns:
            Decoded text string
        """
        # Filter special tokens if requested
        if skip_special_tokens:
            special_ids = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
            token_ids = [tid for tid in token_ids if tid not in special_ids]
        
        # Convert IDs to tokens
        tokens_str = []
        for tid in token_ids:
            if 0 <= tid < self.vocab_size:
                tokens_str.append(self.tokens[tid])
        
        # Join tokens
        text = ''.join(tokens_str)
        
        # Decode bytes
        try:
            text_bytes = bytes([self.byte_decoder.get(c, ord(c)) for c in text])
            return text_bytes.decode('utf-8', errors='replace')
        except Exception:
            return text
    
    @classmethod
    def from_gguf_metadata(cls, metadata: Dict) -> 'GGUFTokenizer':
        """
        Create tokenizer from GGUF metadata dictionary.
        
        Args:
            metadata: GGUF metadata dict from GGUFParser
            
        Returns:
            GGUFTokenizer instance
            
        Raises:
            ValueError: If required tokenizer metadata is missing
        """
        # Extract required fields
        tokens = metadata.get('tokenizer.ggml.tokens')
        merges = metadata.get('tokenizer.ggml.merges')
        bos_token_id = metadata.get('tokenizer.ggml.bos_token_id')
        eos_token_id = metadata.get('tokenizer.ggml.eos_token_id')
        
        if tokens is None:
            raise ValueError("Missing 'tokenizer.ggml.tokens' in GGUF metadata")
        if merges is None:
            raise ValueError("Missing 'tokenizer.ggml.merges' in GGUF metadata")
        if bos_token_id is None:
            raise ValueError("Missing 'tokenizer.ggml.bos_token_id' in GGUF metadata")
        if eos_token_id is None:
            raise ValueError("Missing 'tokenizer.ggml.eos_token_id' in GGUF metadata")
        
        # Optional fields
        pad_token_id = metadata.get('tokenizer.ggml.padding_token_id')
        add_bos_token = metadata.get('tokenizer.ggml.add_bos_token', False)
        
        return cls(
            tokens=tokens,
            merges=merges,
            bos_token_id=int(bos_token_id),
            eos_token_id=int(eos_token_id),
            pad_token_id=int(pad_token_id) if pad_token_id is not None else None,
            add_bos_token=bool(add_bos_token)
        )
    
    @classmethod
    def from_gguf_file(cls, gguf_path: str) -> 'GGUFTokenizer':
        """
        Load tokenizer directly from GGUF file.
        
        Args:
            gguf_path: Path to GGUF file
            
        Returns:
            GGUFTokenizer instance
        """
        from .gguf_parser import GGUFParser
        
        parser = GGUFParser(gguf_path)
        parser.parse()
        tokenizer = cls.from_gguf_metadata(parser.metadata)
        parser.close()
        
        return tokenizer
