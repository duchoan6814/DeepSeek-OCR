import os
import fitz
import img2pdf
import io
import re
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor


if torch.version.cuda == "11.8":
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ["VLLM_USE_V1"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from config import (
    MODEL_PATH,
    INPUT_PATH,
    OUTPUT_PATH,
    PROMPT,
    SKIP_REPEAT,
    MAX_CONCURRENCY,
    NUM_WORKERS,
    CROP_MODE,
)

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepseek_ocr import DeepseekOCRForCausalLM

from vllm.model_executor.models.registry import ModelRegistry

from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    enforce_eager=False,
    trust_remote_code=True,
    max_model_len=8192,
    swap_space=0,
    max_num_seqs=MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    disable_mm_preprocessor_cache=True,
)

logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822}
    )
]  # window for fast；whitelist_token_ids: <td>,</td>

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
    include_stop_str_in_output=True,
)


class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


INPUT_PATH = "assets/images/1_0.jpg"


def describe_image(image_path: str) -> str:
    """Mô tả hình ảnh"""
    image = Image.open(image_path).convert("RGB")

    outputs_list = llm.generate(
        [
            {
                "prompt": PROMPT,
                "multi_modal_data": {
                    "image": DeepseekOCRProcessor().tokenize_with_images(
                        images=[image], bos=True, eos=True, cropping=CROP_MODE
                    )
                },
            },
        ],
        sampling_params=sampling_params,
    )

    contents = ""

    for output in outputs_list:
        content = output.outputs[0].text

        if "<｜end▁of▁sentence｜>" in content:  # repeat no eos
            content = content.replace("<｜end▁of▁sentence｜>", "")
        else:
            if SKIP_REPEAT:
                continue

        contents += content.strip()

    return contents


if __name__ == "__main__":

    image = Image.open(INPUT_PATH).convert("RGB")

    outputs_list = llm.generate(
        [
            {
                "prompt": PROMPT,
                "multi_modal_data": {
                    "image": DeepseekOCRProcessor().tokenize_with_images(
                        images=[image], bos=True, eos=True, cropping=CROP_MODE
                    )
                },
            },
            # {"prompt": prompt, "multi_modal_data": {"image": image_2}},
        ],
        sampling_params=sampling_params,
    )

    for output in outputs_list:
        print("Output:\n", output.outputs[0].text)
