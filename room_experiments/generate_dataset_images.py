from diffusers import DiffusionPipeline
import torch
import yaml
import os
from tqdm import tqdm
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1

import argparse

device = "cuda"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True,
                    help="Path to the pretrained FLUX.1-schnell model.")
parser.add_argument("--output-root", type=str, default="generated_images",
                    help="Root folder for generated images, with subfolders per category and item.")
parser.add_argument("--only", nargs="+", default=None,
                    help="If set, generate only these item names (exact keys from the YAML). "
                         "Separate multi-word names with quotes, e.g. --only 'Couch+Armchair' 'Couch+Framed painting'")
parser.add_argument("--n", type=int, default=2000, help="Images per prompt")
parser.add_argument("--seed", type=int, default=1, help="Random seed for image generation")
parser.add_argument("--append", action="store_true",
                    help="Start image indices after the highest existing index in each folder "
                         "instead of overwriting from 1.")
args = parser.parse_args()


def get_next_index(folder):
    """Return max existing image index + 1, or 1 if the folder is empty/absent."""
    if not os.path.isdir(folder):
        return 1
    indices = []
    for f in os.listdir(folder):
        if f.startswith("image_") and f.endswith(".png"):
            try:
                indices.append(int(f[6:-4]))
            except ValueError:
                pass
    return max(indices) + 1 if indices else 1

with open(os.path.join(SCRIPT_DIR, "training_prompts.yaml"), "r") as f:
    prompts = yaml.safe_load(f)

if args.only is not None:
    prompts = {k: v for k, v in prompts.items() if k in args.only}
    print(f"Filtered to {len(prompts)} item(s): {list(prompts.keys())}")

seed = args.seed
generator = torch.Generator(device=device).manual_seed(seed)

model_path = args.model_path
pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)
print("Moving to GPU...")
pipe = pipe.to(device)
get_weighted_text_embeddings = get_weighted_text_embeddings_flux1

images_per_prompt = args.n
batch_size = 20
output_root = args.output_root

# Prepare jobs
jobs = []
for item_name, attrs in tqdm(prompts.items(), desc="Preparing jobs"):
    category = attrs["category"]
    prompt = attrs["prompt"]
    folder = os.path.join(output_root, category, item_name.replace(" ", "_").lower())
    start = get_next_index(folder) if args.append else 1
    if args.append:
        print(f"  {item_name}: starting from index {start}")
    (prompt_embeds, pooled_prompt_embeds) = get_weighted_text_embeddings(pipe, prompt=prompt)
    for i in range(images_per_prompt):
        jobs.append((item_name, category, prompt_embeds, pooled_prompt_embeds, start + i))

# Batched inference
for i in tqdm(range(0, len(jobs), batch_size), desc="Generating images"):
    batch = jobs[i:i + batch_size]
    batch_prompt_embeds = torch.cat([e for (_, _, e, _, _) in batch], dim=0)
    batch_pooled_prompt_embeds = torch.cat([pe for (_, _, _, pe, _) in batch], dim=0)

    images = pipe(
        prompt_embeds=batch_prompt_embeds,
        pooled_prompt_embeds=batch_pooled_prompt_embeds,
        num_inference_steps=6,
        width=256,
        height=256,
        guidance_scale=7.5,
        generator=generator,
    ).images

    for (item_name, category, _, _, img_index), image in zip(batch, images):
        folder = os.path.join(output_root, category, item_name.replace(" ", "_").lower())
        os.makedirs(folder, exist_ok=True)
        image.save(os.path.join(folder, f"image_{img_index}.png"))
