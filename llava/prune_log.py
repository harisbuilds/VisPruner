# [LOG] Per-question logging of VisPruner's kept tokens and timings.
# Model code writes into CURRENT; the eval loop reads it after each question and clears it.
# Everything here is a no-op unless ENABLED is set (model_vqa_loader sets it from --log_dir).

import json
import os
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

ENABLED = False
CURRENT = {}
_shape_printed = False


def now():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class FirstTokenTimer(LogitsProcessor):
    """Records when the logits for the first output token are ready (end of prefill). Returns scores unchanged."""

    def __call__(self, input_ids, scores):
        if "t_first_token" not in CURRENT:
            CURRENT["t_first_token"] = now()
        return scores


def add_first_token_timer(kwargs):
    processors = LogitsProcessorList(kwargs.get("logits_processor") or [])
    processors.append(FirstTokenTimer())
    kwargs["logits_processor"] = processors
    return kwargs


def record_selection(t_start, t_vision_end, important_indices, diverse_indices, cls_attn, k, n_patches):
    """Called from encode_images with batch size 1. cls_attn is the head-averaged [CLS]->patch attention, shape (1, 576)."""
    t_end = now()
    CURRENT["t_vision"] = t_vision_end - t_start
    CURRENT["t_prune"] = t_end - t_vision_end
    if k >= n_patches:
        CURRENT["important_idx"] = list(range(n_patches))
        CURRENT["diverse_idx"] = []
    else:
        CURRENT["important_idx"] = sorted(important_indices[0].tolist())
        CURRENT["diverse_idx"] = sorted(diverse_indices[0].tolist()) if diverse_indices is not None else []
    CURRENT["cls_attn"] = cls_attn[0].to(torch.float16).cpu().numpy()


def print_shape_once(shape):
    global _shape_printed
    if not _shape_printed:
        print(f"[LOG] visual tokens entering the LLM: {tuple(shape)}")
        _shape_printed = True


def write_question(log_file, cls_attn_dir, question_id, image_file, question, prediction, k, important_ratio, output_ids, n_visual_tokens):
    image_id = os.path.splitext(os.path.basename(image_file))[0]
    record = {
        "question_id": str(question_id),
        "image_id": image_id,
        "question": question,
        "prediction": prediction,
        "k": k,
        "important_ratio": important_ratio,
        "important_idx": CURRENT["important_idx"],
        "diverse_idx": CURRENT["diverse_idx"],
        "n_visual_tokens": n_visual_tokens,
        "t_vision": CURRENT["t_vision"],
        "t_prune": CURRENT["t_prune"],
        "t_prefill": CURRENT["t_first_token"] - CURRENT["t_llm_start"],
        "t_decode": CURRENT["t_llm_end"] - CURRENT["t_first_token"],
        "n_output_tokens": int(output_ids.shape[1]),  # includes the end-of-sequence token
    }
    log_file.write(json.dumps(record) + "\n")
    log_file.flush()

    # [CLS] attention from CLIP layer mm_vision_select_layer (-2, the penultimate of 24), averaged over the 16 heads
    cls_attn_path = os.path.join(cls_attn_dir, image_id + ".npy")
    if not os.path.exists(cls_attn_path):
        tmp_path = cls_attn_path + f".{os.getpid()}.tmp.npy"
        np.save(tmp_path, CURRENT["cls_attn"])
        os.replace(tmp_path, cls_attn_path)


def git_info():
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True, check=True).stdout.strip() != ""
        return commit, dirty
    except Exception:
        return None, None


def write_run_meta(path, meta, end=False):
    if end:
        meta["end_time"] = datetime.now().isoformat(timespec="seconds")
    else:
        import transformers
        commit, dirty = git_info()
        meta.update({
            "git_commit": commit,
            "git_dirty": dirty,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "start_time": datetime.now().isoformat(timespec="seconds"),
            "end_time": None,
        })
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
