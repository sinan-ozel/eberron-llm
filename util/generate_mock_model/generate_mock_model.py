from transformers import PreTrainedModel, GPT2Config, GPT2Tokenizer
import torch.nn as nn
import torch
import os

# Define paths
cache_dir = "/hf"
model_name = "MockLLM"
org_name = "test"
commit_hash = "mockhash"

model_path = os.path.join(cache_dir, "hub", f"models--{org_name}--{model_name}", "snapshots", commit_hash)
os.makedirs(model_path, exist_ok=True)


class MinimalModel(PreTrainedModel):
    config_class = GPT2Config
    _no_split_modules = ["MinimalModel"]

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.dummy_layer = nn.Linear(1, 1)

    def forward(self, input_ids=None, **kwargs):
        return {"output": self.dummy_layer(torch.tensor([[0.0]]))}

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs):
        config = GPT2Config()  # Use BERT config
        return MinimalModel(config)

    def save_pretrained(self, save_directory, **kwargs):
        # Save the model to the specified directory
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        # Save model configuration and weights (even though minimal)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        self.config.save_pretrained(save_directory)

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Load the base tokenizer
        tokenizer.save_pretrained(save_directory)  # Save the tokenizer

# Save to Hugging Face cache
model = MinimalModel(GPT2Config())
model.save_pretrained(model_path)

print(f"Model saved to {model_path}")


