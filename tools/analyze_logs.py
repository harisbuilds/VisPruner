# [LOG] Analyze per-question logs written by model_vqa_loader --log_dir.
#
# Usage (the first run is the reference for the comparisons):
#   python tools/analyze_logs.py --gqa_questions /path/to/testdev_balanced_questions.json \
#       logs/gqa_n576 logs/gqa_n64 logs/gqa_n32
#
# Each run can be a log dir, a questions.jsonl log, or an answers .jsonl from model_vqa_loader
# (answers files only support the accuracy and prediction comparisons).
# --draw N --image_folder DIR draws the kept tokens of the first N questions of each run.

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from itertools import combinations

from PIL import Image, ImageDraw

TIMING_FIELDS = ["t_vision", "t_prune", "t_prefill", "t_decode", "n_output_tokens"]


def load_run(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "questions*.jsonl")))
    else:
        files = [path]
    records = {}
    for f in files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if "prediction" not in r:  # answers file from model_vqa_loader
                    r["prediction"] = r["text"]
                records[str(r["question_id"])] = r
    return records


def is_correct(prediction, answer):
    # Same normalisation as scripts/convert_gqa_for_eval.py; GQA scores exact match
    return prediction.rstrip(".").lower() == answer


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def kept(record):
    return set(record["important_idx"]) | set(record["diverse_idx"])


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def draw_kept_tokens(image_path, important_idx, diverse_idx, out_path, grid=24, input_size=336):
    """Draw kept patches on the original image. LLaVA-1.5 pads the image to a square
    (centered, see expand2square in llava/mm_utils.py), then resizes it to 336x336 with no crop."""
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    S = max(W, H)
    x0 = (S - W) // 2 if H > W else 0
    y0 = (S - H) // 2 if W > H else 0
    patch = input_size / grid * S / input_size  # patch size in original pixels

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for indices, color in [(important_idx, (255, 0, 0)), (diverse_idx, (0, 120, 255))]:
        for idx in indices:
            row, col = idx // grid, idx % grid
            left = max(0, col * patch - x0)
            top = max(0, row * patch - y0)
            right = min(W, (col + 1) * patch - x0)
            bottom = min(H, (row + 1) * patch - y0)
            if right > left and bottom > top:  # skip patches that fall entirely in the padding
                draw.rectangle([left, top, right, bottom], fill=color + (90,), outline=color + (255,))

    out = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.save(out_path)


parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+")
parser.add_argument("--gqa_questions", type=str, default=None, help="official GQA testdev_balanced_questions.json")
parser.add_argument("--draw", type=int, default=0, help="draw kept tokens for the first N questions of each run")
parser.add_argument("--image_folder", type=str, default=None)
args = parser.parse_args()

runs = {path: load_run(path) for path in args.runs}
gt = json.load(open(args.gqa_questions)) if args.gqa_questions else None
ref_path = args.runs[0]

for path, records in runs.items():
    print(f"\n=== {path}  ({len(records)} questions)")

    # Accuracy overall and per GQA question type
    if gt is not None:
        correct = {qid: is_correct(r["prediction"], gt[qid]["answer"]) for qid, r in records.items()}
        print(f"Accuracy: {100 * sum(correct.values()) / len(correct):.2f}")
        for type_key in ["structural", "semantic"]:
            by_type = defaultdict(list)
            for qid, c in correct.items():
                by_type[gt[qid]["types"][type_key]].append(c)
            print(f"  by {type_key}: " + "  ".join(
                f"{t}={100 * sum(v) / len(v):.2f} (n={len(v)})" for t, v in sorted(by_type.items())))

    # Comparison with the reference run
    if path != ref_path:
        ref = runs[ref_path]
        common = [qid for qid in records if qid in ref]
        same = sum(records[qid]["prediction"] == ref[qid]["prediction"] for qid in common)
        print(f"vs reference: {len(common)} common questions, identical predictions {same}/{len(common)} ({100 * same / max(1, len(common)):.2f}%)")
        if gt is not None:
            right_to_wrong = sum(is_correct(ref[q]["prediction"], gt[q]["answer"]) and not is_correct(records[q]["prediction"], gt[q]["answer"]) for q in common)
            wrong_to_right = sum(not is_correct(ref[q]["prediction"], gt[q]["answer"]) and is_correct(records[q]["prediction"], gt[q]["answer"]) for q in common)
            print(f"  right->wrong={right_to_wrong}  wrong->right={wrong_to_right}  McNemar exact p={mcnemar_exact(right_to_wrong, wrong_to_right):.4g}")

    logged = [r for r in records.values() if "important_idx" in r]
    if not logged:
        continue

    # Index checks
    problems = defaultdict(int)
    for r in logged:
        imp, div = r["important_idx"], r["diverse_idx"]
        k = min(r["k"], 576)
        n_imp = 576 if r["k"] >= 576 else int(r["k"] * r["important_ratio"])  # the code floors, see encode_images
        if len(imp) + len(div) != k: problems["total != k"] += 1
        if len(imp) != n_imp: problems["len(important) != int(k * ratio)"] += 1
        if any(i < 0 or i > 575 for i in imp + div): problems["index outside 0-575"] += 1
        if len(set(imp)) != len(imp) or len(set(div)) != len(div): problems["duplicate index"] += 1
        if set(imp) & set(div): problems["important/diverse overlap"] += 1
        if r.get("n_visual_tokens") is not None and r["n_visual_tokens"] != k: problems["visual tokens into LLM != k"] += 1
    print("Index checks: " + ("all OK" if not problems else ", ".join(f"{p}: {n}" for p, n in problems.items())))

    # Kept-token overlap across questions on the same image
    by_image = defaultdict(list)
    for r in logged:
        by_image[r["image_id"]].append(kept(r))
    pair_scores = [jaccard(a, b) for sets in by_image.values() for a, b in combinations(sets, 2)]
    n_images = sum(len(s) > 1 for s in by_image.values())
    if pair_scores:
        print(f"Jaccard of kept tokens, same image: mean={sum(pair_scores) / len(pair_scores):.4f}  min={min(pair_scores):.4f}  "
              f"({n_images} images with 2+ questions, {len(pair_scores)} pairs)")

    # Timing
    print("Mean " + "  ".join(f"{f}={sum(r[f] for r in logged) / len(logged):.4f}" for f in TIMING_FIELDS))

    if args.draw:
        out_dir = os.path.join(path if os.path.isdir(path) else os.path.dirname(path), "kept_tokens")
        for r in logged[:args.draw]:
            image_file = glob.glob(os.path.join(args.image_folder, r["image_id"] + ".*"))[0]
            draw_kept_tokens(image_file, r["important_idx"], r["diverse_idx"], os.path.join(out_dir, f"{r['question_id']}.png"))
        print(f"Drew {min(args.draw, len(logged))} images to {out_dir} (red = important, blue = diverse)")
