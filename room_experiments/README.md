# Room Experiments

This directory contains code for replicating the dataset and experimental results for the room experiments.


## Running
Before running, ensure you have installed all dependencies with `uv sync` from the root directory. `uv` can be installed [here](https://docs.astral.sh/uv/getting-started/installation/).

In order to generate images for the dataset, you will also need to download the [FLUX.1-schnell model](https://huggingface.co/black-forest-labs/FLUX.1-schnell).

Then, `cd` into this directory.


## Creating the Dataset
Creating the dataset is broken down into several steps.

### 1. Create training prompts

```
python create_prompts.py
```
creates prompts from the `perturbations.yaml` file for an empty (`control`) room, a room with only one specified object in it, or a room with exactly one each of two specified objects.


### 2. Generate raw images per prompt
Now use these prompts to generate `n` images per class with a text-to-image model. This requires you to supply the path to your downloaded text-to-image model.

```
python generate_dataset_images.py \
  --model-path /path/to/FLUX.1-schnell \
  --output-root /path/to/gen_images \
  --n 2000 \
  --batch-size 20
```

Images are written to `<output-root>/<category>/<item_name_lower>/image_<idx>.png` (so single-object classes land under e.g. `<output-root>/furniture/couch/` and pairs under `<output-root>/two_objects/couch+framed_painting/`).

The `--only` argument restricts image generation to a subset of the prompts, e.g. `--only "Couch" "Couch+Framed painting"`. `--append` continues numbering past the highest existing index instead of overwriting from `image_1.png`.


### 3 Label a susbet of the images
For each class, label some some of the images manually as either accept/reject, based on whether or not they satisfy the prompt. This is to ensure that rooms contain ONLY the specified objects. Mirroring the generated dataset layout, place the labelled images in a new directory, with the structure `<dir>/accept/<classes>/<images_for_class>` and `<dir>/reject/<classes>/<images_for_class>`. The unlabeled images will be fed through classifiers trained on the labeled images.

Experimenting with 1000 labeled images per class and witholding a validation set, we find that it is helpful to label at least 200 images per class, and that increasing the number of labeled images beyond this point has diminishing returns on classifier accuracy. Hence, we suggest manually labeling at least 200 images per class.

### 4. Train classifiers, build the per-class dataset

To train the classifiers to label the remaining images and construct the dataset, run

```
python build_dataset.py \
  --labeled_dir /path/to/labeled \
  --gen_dir /path/to/gen_images \
  --out_dir /path/to/dataset_out \
  --classifier_dir /path/to/classifiers
```

where `out_dir` is where you would like the dataset to be written to, and `classifier_dir` is where you would like classifier checkpoints to be saved to. (Classifier checkpoints can be safely deleted afterwards if you do not want to keep them.)

For each class in `--classes` (defaults to
`control coffee_table couch framed_painting couch+coffee_table
couch+framed_painting coffee_table+framed_painting`), the script:

  1. Embeds `--labeled_dir/{accept,reject}/<class>/*.png` with DINOv2
     and fits a per-class logistic-regression accept/reject classifier
     (saved to `--classifier_dir/<class>.pkl`).
  2. Copies the hand-labeled accepts into `--out_dir/<class>/`.
  3. Runs the classifier on the remaining unlabeled images located at
     `--gen_dir/.../<class>/*.png` (the script walks one or two levels
     under `--gen_dir` to find each class's folder), and copies the
     classifier-accepted images into `--out_dir/<class>/`.


Be careful when generating the dataset that accept/reject image file names do not overlap with the files to be classified (within the same class), as this can cause files to be overwritten. This should not occur if you do not rename any generated image files.



## Training the diffusion model

To run, there are two primary options: single conditional model, or separate models for each condition. You can run using conditions and data mixtures defined in `train_configs`. For example, to train a single conditional model on the Factorized Conditional In-Distribution conditional mixture as described in the paper:
```
python train.py \
  --data_dir /path/to/dataset_out \
  --output_dir /path/to/checkpoints \
  --conditions train_configs/FC_ID.yaml
```

To train separate models for each condition, run the same command, varying the ``--conditions`` parameter to one of ``train_configs/FC_ID_sep_control``, ``train_configs/FC_ID_sep_couch``, and ``train_configs/FC_ID_sep_painting``.


To train a single conditional model on (a susbet of) the raw class folders, without special mixture distributions:
```
python train.py \
  --data_dir /path/to/dataset \
  --output_dir /path/to/checkpoints \
  --classes coffee_table control couch framed_painting couch+coffee_table couch+framed_painting
```

Each checkpoint directory contains the base weights, an `ema_model/`
copy, the saved scheduler, `normalize.json`, `classes.json` (which
records either the raw class list or the condition names, depending on
the training mode), and `conditions.yaml` when applicable.

It is also recommended that you run training with `accelerate run` instead of `python` to speed up training or to utilize multiple GPUs, e.g. 
```
accelerate launch --mixed_precision bf16 --multi-gpu --num_processes <number of cpus>
```
(omit `--multi-gpu` if training on a single GPU).



### 6. FKC sampling
You can run FKC sampling with the trained model(s). 

**Single-conditional mode** (one checkpoint trained with multiple classes via `train.py --conditions ...` or `--classes ...`):

```
python run_fkc_sample.py \
  --checkpoint /path/to/checkpoints/checkpoint-epoch50 \
  --classes mostly_couch mostly_painting base \
  --betas 1.0 1.0 -1.0 \
  --n_output 10 \
  --n_particles 8 \
  --n_steps 100 \
  --out /path/to/fkc_sample.png
```

**Multi-model mode** (N separate checkpoints, each trained on a single class):

```
python run_fkc_sample.py \
  --checkpoint /path/to/mostly_couch_only/checkpoint-epoch50 \
               /path/to/mostly_painting_only/checkpoint-epoch50 \
               /path/to/base_only/checkpoint-epoch50 \
  --betas 1.0 1.0 -1.0 \
  --n_output 10 --n_particles 8 --n_steps 100 \
  --out /path/to/fkc_sample.png
```

All checkpoints must share the same noise schedule.

`--betas` denotes the composition weights for each condition, e.g. a `+1, +1, -1` triple draws from `p(x|c_1) * p(x|c_2) / p(x|c_3)`.

Running these commands will create a PNG grid of `n_output` sample images from the trained models using FKC with `n_particles` particles on the weighted composition defined by `--betas`.
