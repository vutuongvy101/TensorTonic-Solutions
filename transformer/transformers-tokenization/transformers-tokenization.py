import numpy as np
from typing import List, Dict
import re
from collections import Counter

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        # clean
        tokens = []
        for txt in texts:
            sentence = re.sub(r'[^a-zA-Z0-9\s]', '', txt.lower())
            tokens.extend(sentence.split())
            
        # 3. Counting Frequencies
        word_counts = Counter(tokens)
        
        # 4. Creating Vocabulary (with minimum frequency 1)
        self.word_to_id = {self.pad_token: 0, self.unk_token: 1, self.bos_token: 2, self.eos_token: 3} # Special tokens
        self.id_to_word = {0: self.pad_token, 1:self.unk_token, 2:self.bos_token, 3:self.eos_token}
        self.vocab_size = 4
        for word in word_counts:
            if word not in self.word_to_id:
                self.word_to_id[word] = self.vocab_size
                self.id_to_word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        # clean 
        rs = []
        sentence = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        tokens = sentence.split()
        for w in tokens:
            if w in self.word_to_id.keys():
                rs.append(self.word_to_id[w])
            else:
                rs.append(self.word_to_id[self.unk_token])
        return rs
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        rs = []
        for id in ids:
            if id in self.id_to_word:
                rs.append(self.id_to_word[id])
            else: 
                rs.append(self.unk_token)
        return " ".join(rs)
