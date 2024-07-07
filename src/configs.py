GPT_CONFIG_124M_1024 = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "num_heads": 12,  # Number of attention heads
    "num_layers": 12,  # Number of layers
    "dropout": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}

GPT_CONFIG_124M_256 = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "num_heads": 12,  # Number of attention heads
    "num_layers": 12,  # Number of layers
    "dropout": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}

GPT_CONFIG_124M = GPT_CONFIG_124M_1024
