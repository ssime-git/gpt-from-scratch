


# Define the embedding matrix (for simplicity, using random initialization)
embedding_dim = 8  # Dimension of the embedding vectors
vocab_size = len(vocab)
embedding_matrix = np.random.rand(vocab_size, embedding_dim)

# Convert token IDs to embeddings
embeddings = np.array([embedding_matrix[token_id] for token_id in token_ids])
print("Embeddings:\n", embeddings)

# Function to get positional encodings
def get_positional_encoding(seq_len, d_model):
    positional_encoding = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            positional_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / d_model)))
    return positional_encoding

# Get positional encodings
positional_encodings = get_positional_encoding(len(token_ids), embedding_dim)

# Add positional encodings to embeddings
final_embeddings = embeddings + positional_encodings
print("Final Embeddings with Positional Encodings:\n", final_embeddings)