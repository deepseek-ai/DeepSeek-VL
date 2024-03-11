# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import torch
from threading import Thread
from transformers import AutoModelForCausalLM
from deepseek_vl.utils.io import load_pil_images
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM

# Enable faster download speed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
MODEL_NAME = "deepseek-ai/deepseek-vl-7b-base"
CACHE_DIR = "checkpoints"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR
        )
        self.tokenizer = self.vl_chat_processor.tokenizer
        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR
        )
        self.vl_gpt = vl_gpt.to('cuda')

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Input prompt", default="Describe the image"),
        max_new_tokens: int = Input(description="Maximum number of tokens to generate", default=512)
    ) -> str:
        """Run a single prediction on the model"""
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder>"+prompt,
                "images": [str(image)]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to('cuda')

        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer
