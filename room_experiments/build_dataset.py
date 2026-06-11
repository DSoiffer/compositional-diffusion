"""Build a per-class training dataset for the furniture diffusion model.

For each class (single-object or pair):
  1. Embed labeled accept/reject images with DINOv2.
  2. Fit a per-class logistic-regression accept/reject classifier.
  3. Copy all hand-labeled accepts into the output directory.
  4. Run the classifier over the unlabeled generated pool for that class
     and copy classifier-accepted images into the output directory.

Every missing input directory or empty source raises an exception.
"""

import argparse
import shutil
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


BATCH_SIZE = 64

DEFAULT_CLASSES = [
    "control",
    "coffee_table",
    "couch",
    "framed_painting",
    "couch+coffee_table",
    "couch+framed_painting",
    "coffee_table+framed_painting",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labeled_dir", type=Path, required=True,
        help="Directory with accept/<class>/ and reject/<class>/ subfolders "
        "(one set per class in --classes).",
    )
    parser.add_argument(
        "--gen_dir", type=Path, required=True,
        help="Root of unlabeled generated images. Each class in --classes "
        "must appear as a leaf folder somewhere under this root (e.g. "
        "<gen_dir>/furniture/couch/, <gen_dir>/two_objects/couch+framed_painting/).",
    )
    parser.add_argument(
        "--out_dir", type=Path, required=True,
        help="Output dataset root (will be created). Per-class subfolders "
        "contain copied (not symlinked) PNGs.",
    )
    parser.add_argument(
        "--classifier_dir", type=Path, required=True,
        help="Directory to save trained per-class classifiers (pkl).",
    )
    parser.add_argument(
        "--classes", type=str, nargs="+",
        default=DEFAULT_CLASSES,
        help="Class names. Default includes 4 single classes "
        "(control, coffee_table, couch, framed_painting) plus the three "
        "two-object combinations of the non-control objects.",
    )
    parser.add_argument(
        "--dinov2_model", type=str, default="facebook/dinov2-large",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_dinov2(name: str, device: str):
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device)
    model.eval()
    return model, processor


def embed_images(model, processor, paths, device, desc=""):
    """Embed a list of image paths, returns (N, D) float32 array."""
    if not paths:
        raise ValueError(f"embed_images called with no paths (desc={desc!r})")
    all_embs = []
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc=desc, leave=False):
        batch = paths[i : i + BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_embs.append(cls)
    return np.vstack(all_embs)


def load_class_embeddings(model, processor, root_dir: Path, cls: str, device: str):
    """Embed all PNGs in root_dir/cls/. Returns (embeddings, filenames)."""
    d = root_dir / cls
    if not d.is_dir():
        raise FileNotFoundError(f"Expected labeled directory not found: {d}")
    paths = sorted(d.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNGs found in labeled directory: {d}")
    embs = embed_images(model, processor, paths, device, desc=f"    {root_dir.name}/{cls}")
    return embs, [p.name for p in paths]


def train_classifier(accept_embs, reject_embs, cls: str):
    X = np.vstack([accept_embs, reject_embs])
    y = np.array([1] * len(accept_embs) + [0] * len(reject_embs))
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    acc = accuracy_score(y, clf.predict(X))
    print(f"  {cls}: {len(accept_embs)} accept + {len(reject_embs)} reject, train acc = {acc:.3f}")
    return clf


def classify_generated(clf, model, processor, gen_dir: Path, exclude_filenames, device: str):
    """Run the classifier over PNGs in gen_dir, skipping exclude_filenames."""
    if not gen_dir.is_dir():
        raise FileNotFoundError(f"Generated-images directory not found: {gen_dir}")
    all_paths = sorted(gen_dir.glob("*.png"))
    if not all_paths:
        raise FileNotFoundError(f"No PNGs found in generated-images directory: {gen_dir}")
    to_classify = [p for p in all_paths if p.name not in exclude_filenames]
    if not to_classify:
        raise RuntimeError(
            f"All {len(all_paths)} images in {gen_dir} are in exclude_filenames; "
            "nothing left to classify."
        )

    accepted = []
    for i in tqdm(
        range(0, len(to_classify), BATCH_SIZE),
        desc=f"    classifying {gen_dir.name}", leave=False,
    ):
        batch = to_classify[i : i + BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        embs = out.last_hidden_state[:, 0, :].cpu().numpy()
        preds = clf.predict(embs)
        for path, pred in zip(batch, preds):
            if pred == 1:
                accepted.append(path)
    return accepted


def find_class_dirs(gen_dir: Path, class_names) -> dict:
    """Walk gen_dir at depth <=2 and find a leaf directory named <class> for
    each class in class_names. The expected layout is
    <gen_dir>/<category>/<class>/, matching generate_dataset_images.py output.

    Raises on missing or ambiguous matches.
    """
    if not gen_dir.is_dir():
        raise FileNotFoundError(f"Generated-images root not found: {gen_dir}")
    found: dict[str, list[Path]] = {n: [] for n in class_names}
    for category in gen_dir.iterdir():
        if not category.is_dir():
            continue
        for leaf in category.iterdir():
            if leaf.is_dir() and leaf.name in found:
                found[leaf.name].append(leaf)
        # Also accept class folders sitting directly under gen_dir (no
        # intermediate category level).
        if category.name in found:
            found[category.name].append(category)
    missing = sorted(n for n, paths in found.items() if not paths)
    if missing:
        raise FileNotFoundError(
            f"Could not find generated-image folders for class(es) {missing} "
            f"anywhere under {gen_dir}."
        )
    ambiguous = sorted(n for n, paths in found.items() if len(paths) > 1)
    if ambiguous:
        details = "; ".join(
            f"{n} -> {[str(p) for p in found[n]]}" for n in ambiguous
        )
        raise RuntimeError(
            f"Ambiguous generated-image folders for class(es): {details}"
        )
    return {n: paths[0] for n, paths in found.items()}


def main():
    args = parse_args()

    if not args.labeled_dir.is_dir():
        raise FileNotFoundError(f"Labeled directory not found: {args.labeled_dir}")

    print(f"\nLoading {args.dinov2_model}...")
    dino_model, dino_proc = load_dinov2(args.dinov2_model, args.device)

    print("\nResolving generated-image folders...")
    class_to_gen = find_class_dirs(args.gen_dir, args.classes)
    for cls in args.classes:
        print(f"  {cls}: {class_to_gen[cls]}")

    print("\nTraining classifiers...")
    args.classifier_dir.mkdir(parents=True, exist_ok=True)
    classifiers = {}
    for cls in args.classes:
        print(f"  Embedding {cls}...")
        accept_embs, _ = load_class_embeddings(
            dino_model, dino_proc, args.labeled_dir / "accept", cls, args.device,
        )
        reject_embs, _ = load_class_embeddings(
            dino_model, dino_proc, args.labeled_dir / "reject", cls, args.device,
        )
        clf = train_classifier(accept_embs, reject_embs, cls)
        classifiers[cls] = clf
        joblib.dump(clf, args.classifier_dir / f"{cls}.pkl")

    print("\nBuilding dataset (copying)...")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for cls in args.classes:
        out_dir = args.out_dir / cls
        out_dir.mkdir(exist_ok=True)

        accept_src = args.labeled_dir / "accept" / cls
        reject_src = args.labeled_dir / "reject" / cls
        if not accept_src.is_dir():
            raise FileNotFoundError(f"Missing labeled accept directory: {accept_src}")
        if not reject_src.is_dir():
            raise FileNotFoundError(f"Missing labeled reject directory: {reject_src}")

        labeled_fnames = {f.name for f in accept_src.glob("*.png")}
        labeled_fnames |= {f.name for f in reject_src.glob("*.png")}

        n_labeled = 0
        for f in accept_src.glob("*.png"):
            shutil.copy2(f, out_dir / f.name)
            n_labeled += 1

        gen_dir = class_to_gen[cls]
        accepted_paths = classify_generated(
            classifiers[cls], dino_model, dino_proc, gen_dir, labeled_fnames, args.device,
        )
        n_classifier = 0
        for p in accepted_paths:
            shutil.copy2(p, out_dir / p.name)
            n_classifier += 1

        print(
            f"  {cls}: {n_labeled} labeled accepts + {n_classifier} classifier accepts "
            f"= {n_labeled + n_classifier} total"
        )

    print("\nDataset summary:")
    total = 0
    for cls_path in sorted(args.out_dir.iterdir()):
        if not cls_path.is_dir():
            continue
        n = sum(1 for f in cls_path.iterdir() if f.name.endswith(".png"))
        print(f"  {cls_path.name:<30} {n:>6} images")
        total += n
    print(f"  {'TOTAL':<30} {total:>6} images")
    print(f"\nDataset at {args.out_dir}")


if __name__ == "__main__":
    main()
