import os
import argparse
import torch
import json
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from huggingface_hub import hf_hub_download


with open("concepts.txt") as infile:
    CONCEPTS = [line.rstrip() for line in infile]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auth_token",
        required=True,
        help="Authentication token from Huggingface for downloaging weights.",
    )
    args = parser.parse_args()

    os.makedirs("pretrain/diffusers-cache", exist_ok=True)
    os.makedirs("pretrain/tokenizer", exist_ok=True)
    os.makedirs("pretrain/text_encoder", exist_ok=True)

    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        cache_dir="pretrain/tokenizer",
        use_auth_token=args.auth_token,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        cache_dir="pretrain/text_encoder",
        use_auth_token=args.auth_token,
    )

    # pipe = StableDiffusionPipeline.from_pretrained(
    #     pretrained_model_name_or_path,
    #     cache_dir="pretrain/diffuser",
    #     torch_dtype=torch.float16,
    #     text_encoder=text_encoder,
    #     tokenizer=tokenizer,
    #     use_auth_token=args.auth_token,
    # )

    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        cache_dir="pretrain/diffusers-cache",
        torch_dtype=torch.float16,
        use_auth_token=args.auth_token,
    )

    print("Downloading pre-trained concpets...")
    for concept in CONCEPTS:
        concept = concept.split(":")[0]
        os.makedirs(concept, exist_ok=True)
        embeds_path = hf_hub_download(repo_id=concept, filename="learned_embeds.bin", cache_dir=concept, use_auth_token=args.auth_token)
        token_path = hf_hub_download(repo_id=concept, filename="token_identifier.txt", cache_dir=concept, use_auth_token=args.auth_token)
    
        with open(token_path, 'r') as file:
            placeholder = file.read()
        print(f"{concept}: {placeholder}")

    print("All done!") # The permission of the cache-dir may need to change for the demo
