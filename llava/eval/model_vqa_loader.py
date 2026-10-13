import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
from llava import prune_log  # [LOG]


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        if args.blank_image:  # [LOG] sanity check: all-black image of the same size
            image = Image.new('RGB', image.size, (0, 0, 0))
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        visual_token_num=args.visual_token_num,
        important_ratio=args.important_ratio,
    )

    # Data
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    if args.limit is not None:  # [LOG]
        questions = questions[:args.limit]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    if args.log_dir:  # [LOG]
        prune_log.ENABLED = True
        suffix = "" if args.num_chunks == 1 else f"_{args.num_chunks}_{args.chunk_idx}"
        cls_attn_dir = os.path.join(args.log_dir, "cls_attn")
        os.makedirs(cls_attn_dir, exist_ok=True)
        log_file = open(os.path.join(args.log_dir, f"questions{suffix}.jsonl"), "w")
        template = conv_templates[args.conv_mode].copy()
        template.append_message(template.roles[0], DEFAULT_IMAGE_TOKEN + "\n{text}")
        template.append_message(template.roles[1], None)
        run_meta_path = os.path.join(args.log_dir, f"run_meta{suffix}.json")
        run_meta = {
            "model_path": args.model_path,
            "dataset": os.path.basename(os.path.dirname(os.path.abspath(args.question_file))),
            "split": os.path.splitext(os.path.basename(args.question_file))[0],
            "question_file": args.question_file,
            "image_folder": args.image_folder,
            "answers_file": args.answers_file,
            "num_questions": len(questions),
            "limit": args.limit,
            "num_chunks": args.num_chunks,
            "chunk_idx": args.chunk_idx,
            "k": args.visual_token_num,
            "important_ratio": args.important_ratio,
            "conv_mode": args.conv_mode,
            "prompt_template": template.get_prompt(),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "blank_image": args.blank_image,
        }
        prune_log.write_run_meta(run_meta_path, run_meta)

    data_bar = tqdm(zip(data_loader, questions), total=len(questions))
    for (input_ids, image_tensors, image_sizes), line in data_bar:
        idx = line["question_id"]
        cur_prompt = line["text"]
        if args.log_dir:  # [LOG]
            prune_log.CURRENT.clear()

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensors = image_tensors.to(dtype=torch.float16, device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids, visual_token_num = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
            data_bar.set_postfix({"visual_token_num": visual_token_num})

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()

        if args.log_dir:  # [LOG] question text without the instruction suffix, e.g. "\nAnswer the question using a single word or phrase."
            prune_log.write_question(log_file, cls_attn_dir, idx, line["image"], cur_prompt.split("\n")[0], outputs,
                                     args.visual_token_num, args.important_ratio, output_ids, visual_token_num)
    ans_file.close()

    if args.log_dir:  # [LOG]
        log_file.close()
        prune_log.write_run_meta(run_meta_path, run_meta, end=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--visual_token_num", type=int, default=576)
    parser.add_argument("--important_ratio", type=float, default=0.5)
    parser.add_argument("--log_dir", type=str, default=None)  # [LOG] turns on per-question logging
    parser.add_argument("--blank_image", action="store_true")  # [LOG]
    parser.add_argument("--limit", type=int, default=None)  # [LOG] only the first N questions
    args = parser.parse_args()

    eval_model(args)
