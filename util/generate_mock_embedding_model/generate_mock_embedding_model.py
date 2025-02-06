import os
import json

import torch
from sentence_transformers import SentenceTransformer, models
from transformers import GPT2Tokenizer, GPT2Model

print("Checkpoint 0")

# Get environment variables
HF_HOME = os.getenv("HF_HOME", "/hf")
MODEL_ORG = os.getenv("MODEL_ORG", "test")
MODEL_NAME = os.getenv("MODEL_NAME", "MockEmbedding")
COMMIT_HASH = os.getenv("COMMIT_HASH", "mockhash")

# Define the model save path
MODEL_PATH = os.path.join(HF_HOME, 'hub', f"models--{MODEL_ORG}--{MODEL_NAME}", "snapshots", COMMIT_HASH)
os.makedirs(MODEL_PATH, exist_ok=True)

print("Checkpoint 1")

class MinimalSentenceTransformer(SentenceTransformer):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.base_model = GPT2Model.from_pretrained(model_name)

        tokenizer = models.Transformer(model_name)
        # Add a pooling layer to compute embeddings
        pooling_layer = models.Pooling(
            tokenizer.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )

        # Initialize the SentenceTransformer with these components
        super().__init__(modules=[tokenizer, pooling_layer])



    def encode(self, texts, **kwargs):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.base_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Use the mean of the hidden states as the embedding
        return self.dummy_layer(embeddings)  # Apply the dummy layer

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        super().save_pretrained(save_directory)


        # Save a custom config.json to make it compatible with loading
        config = {
            "model_type": "gpt2",
            "sentence_transformers_modules": ["Transformer", "Pooling"],
        }

        # Write the config dictionary to config.json
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=4)


print("Checkpoint 2")

# Save the mock model config
model = MinimalSentenceTransformer()
model.save_pretrained(MODEL_PATH)

print(f"Mock embeddings model saved to {MODEL_PATH}")
