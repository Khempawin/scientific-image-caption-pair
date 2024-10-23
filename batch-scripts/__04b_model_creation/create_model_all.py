from transformers import (
            VisionTextDualEncoderModel,
                VisionTextDualEncoderProcessor,
                    AutoTokenizer,
                        CLIPImageProcessor, BertModel
                        )
from pathlib import Path
import torch.nn as nn
raise Exception("Warning overwrite of manually edited models")
output_dir = Path("/home/horton/datasets/meta-scir/models")
output_dir.mkdir(parents=True, exist_ok=True)

model_variations = [
    {
        "name": "roberta-special-token-base",
        "based_text_model": "roberta-base"
    },
    {
        "name": "miread-special-token-base",
        "based_text_model": "arazd/miread"
    },
    {
        "name": "scibert-special-token-base",
        "based_text_model": "allenai/scibert_scivocab_uncased"
    },
    {
        "name": "specter2-special-token-base",
        "based_text_model": "allenai/specter2_base"
    }
]

class CustomVisionTextDualEncoderModel(VisionTextDualEncoderModel):
    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the model's input embeddings.

        Returns:
            `nn.Module`: A torch module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        
        return base_model._modules["text_model"].get_input_embeddings()

    
    def set_input_embeddings(self, value: nn.Module):
        """
        Set model's input embeddings.

        Args:
            value (`nn.Module`): A module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        base_model._modules["text_model"].set_input_embeddings(value)


for variation in model_variations:
    model = CustomVisionTextDualEncoderModel.from_vision_text_pretrained(
                "openai/clip-vit-base-patch32", variation["based_text_model"]
                )

    tokenizer = AutoTokenizer.from_pretrained(variation["based_text_model"])
    tokens_added = tokenizer.add_tokens(["[START-TITLE]", "[END-TITLE]", "[START-CONCEPT]", "[END-CONCEPT]", "[START-SECTION]", "[END-SECTION]"], special_tokens=True)
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print(f"{tokens_added} have been added. {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))
    print(model._modules["text_model"])

    processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
    

    # save the model and processor
    model.save_pretrained(output_dir / variation["name"])
    processor.save_pretrained(output_dir / variation["name"])
