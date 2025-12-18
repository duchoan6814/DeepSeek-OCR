from typing import Optional
from .ocr_page import open_image_from_url, high_quality_image
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from vllm import SamplingParams

from config import CROP_MODE, SKIP_REPEAT

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


def describle_image_service(
    image_url: str, llm, custom_prompt: Optional[str] = None
) -> str:
    """Dùng AI để mô tả hình ảnh."""

    image = open_image_from_url(image_url)
    image = high_quality_image(image)

    prompt = custom_prompt or "<image>\nDescribe this image in detail."

    outputs_list = llm.generate(
        [
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": DeepseekOCRProcessor().tokenize_with_images(
                        images=[image],
                        bos=True,
                        eos=True,
                        cropping=CROP_MODE,
                        conversation=prompt,
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
