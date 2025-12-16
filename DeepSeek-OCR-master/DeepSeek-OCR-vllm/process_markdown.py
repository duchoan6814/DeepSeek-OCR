from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
import re
import os
import uuid
from pydantic import BaseModel, Field
from typing import Optional
from vision_test import describe_image
from tqdm import tqdm

INPUT_PATH = (
    "assets/sach-giao-khoa-toan-12-tap-1-ket-noi-tri-thuc-voi-cuoc-song-17-19.mmd"
)


def process_markdown(
    markdown_content: str, chunk_size: int = 1024, chunk_overlap: int = 256
) -> list[Document]:
    """Cắt markdown_content thành các chunk nhỏ hơn dựa trên tiêu đề và kích thước tối đa."""
    # MD splits
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split
    splits = text_splitter.split_documents(md_header_splits)
    return splits


class ProcessedImageRef(BaseModel):
    id: uuid.UUID
    origin_path: str
    real_path: str
    url: Optional[str] = Field(
        None, description="URL của hình ảnh đã upload lên server"
    )
    description: Optional[str] = Field(
        None, description="Mô tả hình ảnh được generate bởi AI"
    )


class ProcessedSingleChunkResult(BaseModel):
    content: str  # nội dung đã replace placeholder
    images: list[ProcessedImageRef]  # danh sách hình ảnh đã xử lý


def handle_images_with_ai_vision_model(
    image_refs: list[ProcessedImageRef],
) -> list[ProcessedImageRef]:
    """Xử lý hình ảnh với AI vision model, trả về danh sách hình ảnh đã qua xử lý với id và description."""
    for img_ref in tqdm(image_refs, desc="Processing images with AI vision model"):
        description = describe_image(img_ref.real_path)
        img_ref.description = description

    return image_refs


def process_single_chunk(content: str):
    """Xử lý image cho 1 chunk markdown.
    return: {
        "content": str, # nội dung đã replace placeholder
        "images": list[unknown], # danh sách hình ảnh đã xử lý
    }
    NOTE: danh sách hình ảnh đã qua sử lý sẽ bao gồm id được sinh ra từ uuid và description của hình ảnh xử dụng AI để generate ra, đồng thời cũng nên upload image lên server để cần thì dùng lại
    """
    # Regex detect markdown image: ![](path)
    pattern = r"!\[[^\]]*\]\(([^)]+)\)"

    image_paths = re.findall(pattern, content)

    if not image_paths:
        return ProcessedSingleChunkResult(
            content=content,
            images=[],
        )

    # Tiền xử lý đường dẫn hình ảnh
    image_paths = [
        ProcessedImageRef(
            origin_path=path,
            real_path=os.path.join(os.path.dirname(INPUT_PATH), path),
            id=uuid.uuid4(),
        )
        for path in image_paths
    ]

    # Xử lý hình ảnh với AI vision model
    image_paths = handle_images_with_ai_vision_model(image_paths)

    # Replace từng ảnh bằng placeholder
    def replace_image(match):
        path = match.group(1)

        # Find id from image_paths based on the current path
        id = next((img.id for img in image_paths if img.origin_path == path), None)

        return f"[IMAGE: {id}]"

    processed_text = re.sub(pattern, replace_image, content)

    return ProcessedSingleChunkResult(
        content=processed_text,
        images=image_paths,
    )


def process_chunks(chunks: list[Document]) -> list[Document]:
    """Detect nếu như trong chunk có chứa hình ảnh thì phải process hình ảnh đó."""
    processed_chunks = []
    for chunk in tqdm(chunks, desc="Processing chunks"):
        result = process_single_chunk(chunk.page_content)
        print("Processed chunk result:", result)

        if len(result.images) == 0:
            processed_chunks.append(chunk)
            continue

        processed_chunks.append(
            chunk.model_copy(
                update={
                    "page_content": result.content,
                    "metadata": {
                        **chunk.metadata,
                        "image_refs": [img.id for img in result.images],
                    },
                }
            )
        )
        for img in result.images:
            print(f"Image ID: {img.id}, Description: {img.description}")
            processed_chunks.append(
                Document(
                    page_content=img.description or "",
                    metadata={
                        "image_id": str(img.id),
                        "url": img.url,
                    },
                )
            )

    return processed_chunks


if __name__ == "__main__":
    """Cho file markdown, nhiệm vụ là plit thành từng chunk nhỏ hơn dựa trên các tiêu đề và trả về danh sách các chunk đó."""
    with open(INPUT_PATH, "r", encoding="utf-8") as afile:
        markdown_content = afile.read()

    print("Processing markdown file:", INPUT_PATH)

    chunks = process_markdown(markdown_content, chunk_size=1024, chunk_overlap=256)
    chunks = process_chunks(chunks)

    for idx, chunk in enumerate(chunks):
        print(f"\n--- Chunk {idx + 1} ---\n")
        print(chunk.page_content)
    print("\n--- End of chunks ---\n")
    print(f"Total chunks: {len(chunks)}")
