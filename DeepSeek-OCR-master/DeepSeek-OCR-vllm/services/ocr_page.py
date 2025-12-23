import asyncio
from tkinter import Image
from typing import Optional
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from config import (
    SKIP_REPEAT,
    CROP_MODE,
)
import re
from vllm import LLM, SamplingParams
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests
from io import BytesIO
import httpx
from configs import STRAPI_API_URL, STRAPI_API_TOKEN
from tqdm import tqdm
import logging
import uuid

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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


def re_match(text):
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if "<|ref|>image<|/ref|>" in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def high_quality_image(image: Image.Image) -> Image.Image:
    """Tăng chất lượng hình ảnh chuyền vào, giống logic của pdf_to_images_high_quality"""
    dpi = 144
    zoom = dpi / 72.0
    new_width = int(image.width * zoom)
    new_height = int(image.height * zoom)
    high_res_image = image.resize((new_width, new_height), Image.LANCZOS)
    return high_res_image


def open_image_from_url(image_url: str) -> Image.Image:
    """Mở hình ảnh từ đường dẫn url"""
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image


def process_single_image(image, prompt: str) -> dict:
    """single image"""
    prompt_in = prompt
    cache_item = {
        "prompt": prompt_in,
        "multi_modal_data": {
            "image": DeepseekOCRProcessor().tokenize_with_images(
                images=[image],
                bos=True,
                eos=True,
                cropping=CROP_MODE,
                conversation=prompt_in,
            )
        },
    }
    return cache_item


def extract_coordinates_and_label(ref_text, image_width, image_height):

    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, content: str):
    """Cắt Image theo tọa độ trong refs và vẽ bounding box lên image"""

    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    #     except IOError:
    font = ImageFont.load_default()

    img_idx = 0

    cropped_images = []

    for ref in refs:
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                color = (
                    np.random.randint(0, 200),
                    np.random.randint(0, 200),
                    np.random.randint(0, 255),
                )

                color_a = color + (20,)
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == "image":
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            # cropped.save(f"{OUTPUT_PATH}/images/{jdx}_{img_idx}.jpg")

                            # replace ref placeholder with other placeholder to avoid conflict
                            unique_id = str(uuid.uuid4())

                            placeholder = f"<|image_{unique_id}|>"

                            logger.debug("Replacing %s with %s", ref[0], placeholder)

                            content = content.replace(ref[0], placeholder, 1)

                            cropped_images.append(
                                {
                                    "image": cropped,
                                    "placeholder": placeholder,
                                }
                            )
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1

                    try:
                        if label_type == "title":
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle(
                                [x1, y1, x2, y2],
                                fill=color_a,
                                outline=(0, 0, 0, 0),
                                width=1,
                            )
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle(
                                [x1, y1, x2, y2],
                                fill=color_a,
                                outline=(0, 0, 0, 0),
                                width=1,
                            )

                        text_x = x1
                        text_y = max(0, y1 - 15)

                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle(
                            [text_x, text_y, text_x + text_width, text_y + text_height],
                            fill=(255, 255, 255, 30),
                        )

                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return cropped_images, img_draw, content


async def process_image(image: Image.Image, index: int, *, for_id: str):
    try:
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        image_bytes = img_byte_arr.getvalue()

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                url=f"{STRAPI_API_URL}/api/media/upload",
                files={"file": (f"image_{index}.png", image_bytes, "image/png")},
                data={"folderId": 8},
                headers={"Authorization": f"Bearer {STRAPI_API_TOKEN}"},
                timeout=60,
            )
            response.raise_for_status()
            response_data = response.json().get("data", [])
            return_data = response_data[0] if len(response_data) > 0 else None

        return {"id": for_id, "data": return_data}
    except Exception as e:
        logger.error("Error uploading image %d: %s", index, e)
        return e


async def process_each_page(
    image_url: str, llm, custom_prompt: Optional[str] = None
) -> str:
    """Nhận vào url hình ảnh của 1 page, trả về content markdown của page đó"""
    try:
        image = open_image_from_url(image_url)
        image = high_quality_image(image)

        prompt = (
            custom_prompt or "<image>\n<|grounding|>Convert the document to markdown."
        )

        input_item = process_single_image(image, prompt)

        outputs = llm.generate([input_item], sampling_params=sampling_params)

        content = outputs[0].outputs[0].text

        if "<｜end▁of▁sentence｜>" in content:  # repeat no eos
            content = content.replace("<｜end▁of▁sentence｜>", "")
        else:
            if SKIP_REPEAT:
                return ""

        image_draw = image.copy()

        matches_ref, matches_images, mathes_other = re_match(content)

        # print(matches_ref)
        cropped_images, image_draw, content = draw_bounding_boxes(
            image_draw, matches_ref, content
        )
        logger.debug("Cropped images count: %d", len(cropped_images))

        upload_results = []

        for index, result_image in enumerate(cropped_images):
            upload_results.append(
                await process_image(
                    result_image["image"], index, for_id=result_image["placeholder"]
                )
            )

        for idx, upload_result in enumerate(upload_results):
            if isinstance(upload_result, Exception):
                logger.error("Error uploading image: %s", upload_result)
                continue

            if not upload_result["id"]:
                logger.error("No ID in upload result: %s", upload_result)
                continue

            if not upload_result["data"]:
                logger.error("No data in upload result: %s", upload_result)
                continue

            content = content.replace(
                upload_result["id"], f"![]({upload_result['data']['id']})\n"
            )

        for idx, a_match_other in enumerate(mathes_other):
            content = (
                content.replace(a_match_other, "")
                .replace("\\coloneqq", ":=")
                .replace("\\eqqcolon", "=:")
                .replace("\n\n\n\n", "\n\n")
                .replace("\n\n\n", "\n\n")
            )

        return content
    except Exception as e:
        logger.error("Error processing page: %s", e)
        raise e
