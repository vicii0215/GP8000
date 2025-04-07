import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import argparse
import os, json, math
from tqdm import tqdm
import base64

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def eval1(args):
    device = "cuda"
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir='/data/qiw')

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=f"cuda:{args.device}",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        # torch_dtype="auto",
        cache_dir='/data/qiw'
    )

    # build output dir
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # load questions
    with open(os.path.join(args.meta_data_path), 'r') as f:
        data = json.load(f)

    ans_file = open(answers_file, "w")
    for i, x in enumerate(tqdm(data)):

        messages = [
            {"role": "system", "content": "You are a helpful assistant in predicting human sleep efficiency."},
            {"role": "user", "content": x['prompt']}
            ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                        model_inputs.input_ids,
                        max_new_tokens=512
                        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        ans_file.write(json.dumps({"question_id": i,
                                   "response": response,
                                   "prompt": x['prompt'],
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-7B-Instruct") #Qwen/Qwen2-VL-7B-Instruct
    parser.add_argument("--answers-file", type=str, default="eval_results")  # output
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    args.answers_file = os.path.join(args.answers_file, args.model_name + '.jsonl')
    args.meta_data_path = 'prompts/prompts_json.json'

    eval1(args)