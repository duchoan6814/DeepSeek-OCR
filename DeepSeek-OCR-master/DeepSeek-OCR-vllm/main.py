import asyncio
from tqdm import tqdm
import time


async def main():
    # TODO: cung cấp một file chứa danh sách các file pdf cần import
    # Các row sẽ phải chứa meta data của từng file pdf như:
    # - đường dẫn file pdf
    # - môn học: ví dụ Toán, Lý, Hóa, Sinh, Văn, Sử, Địa,...
    # - lớp: Ví dụ 6,7,8,9,10,11,12
    # - môn học theo lớp: Ví dụ Toán tập 1
    pdf_file_list = []  # Đọc file và lấy danh sách các file pdf cần OCR

    # TODO: duyệt qua danh sách đó và OCR
    # log lại kết quả OCR vào một file tổng hợp

    start_time = time.time()

    for pdf_file in tqdm(pdf_file_list, desc="Processing PDFs"):
        # Gọi hàm OCR cho từng file pdf
        pass

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time} seconds")


if __name__ == "__main__":
    """Pipeline OCR"""
    asyncio.run(main())
