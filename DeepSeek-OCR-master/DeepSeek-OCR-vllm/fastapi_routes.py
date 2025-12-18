from fastapi import Depends, FastAPI
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager


from deepseek_ocr import DeepseekOCRForCausalLM
import os
import torch

if torch.version.cuda == "11.8":
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ["VLLM_USE_V1"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from vllm.model_executor.models.registry import ModelRegistry

from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from config import (
    MODEL_PATH,
    MAX_CONCURRENCY,
)
from services.ocr_page import process_each_page
from services.describle_image import describle_image_service


class DeepseekOCRVLLM:
    def __init__(self):
        ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
        self.llm = LLM(
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


config_service: DeepseekOCRVLLM = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config_service
    config_service = DeepseekOCRVLLM()
    yield


app = FastAPI(lifespan=lifespan)


def get_config_service() -> DeepseekOCRVLLM:
    return config_service


@app.get("/")
async def root():
    return {"message": "Hello World"}


class OCRRequest(BaseModel):
    image_url: str
    custom_prompt: Optional[str] = None


@app.post("/api/ocr-page")
async def ocr_page(
    request: OCRRequest, config: DeepseekOCRVLLM = Depends(get_config_service)
):
    """Chuyền lên thông tin hình ảnh, trả về kết quả OCR dưới dạng JSON."""

    result = process_each_page(
        request.image_url, custom_prompt=request.custom_prompt, llm=config.llm
    )

    return {"result": result}


@app.post("/api/describe-image")
async def describe_image(
    request: OCRRequest, config: DeepseekOCRVLLM = Depends(get_config_service)
):
    """Chuyền lên thông tin hình ảnh, trả về mô tả hình ảnh dưới dạng văn bản."""

    descriptions = describle_image_service(
        image_url=request.image_url, custom_prompt=request.custom_prompt, llm=config.llm
    )

    return {"description": descriptions}
