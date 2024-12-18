import json
from transformers import AutoTokenizer
import torch
import os

# Set up cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), "tokenizer_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def load_samples(file_path="samples.json"):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_tokenizer(force_download=False):
    try:
        if not force_download:
            # Try to load from cache first
            tokenizer = AutoTokenizer.from_pretrained(
                "baseten/Meta-Llama-3-tokenizer",
                cache_dir=CACHE_DIR,
                local_files_only=True
            )
        else:
            # Force download
            tokenizer = AutoTokenizer.from_pretrained(
                "baseten/Meta-Llama-3-tokenizer",
                cache_dir=CACHE_DIR
            )
        
        # Set up padding token
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except OSError as e:
        if not force_download:
            print("Tokenizer not found in cache. Downloading...")
            return get_tokenizer(force_download=True)
        else:
            raise e

def tokenize_samples(samples):
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Tokenize each sample
    tokenized = []
    for sample in samples:
        tokens = tokenizer(
            sample,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        tokenized.append({
            'input_ids': tokens.input_ids.tolist(),
            'attention_mask': tokens.attention_mask.tolist()
        })
    
    return tokenized

def save_tokenized(tokenized_samples, file_path="tokenized_samples.json"):
    with open(file_path, 'w') as f:
        json.dump(tokenized_samples, f, indent=2)

if __name__ == "__main__":
    # Load samples
    samples = load_samples()
    
    # Tokenize
    tokenized_samples = tokenize_samples(samples)
    
    # Save tokenized versions
    save_tokenized(tokenized_samples)
    
    # Print some stats
    print(f"=== Tokenization Complete ===")
    print(f"Number of samples: {len(tokenized_samples)}")
    print(f"First sample token count: {len(tokenized_samples[0]['input_ids'][0])}")