import os

import fitz  # PyMuPDF

def extract_images_pymupdf(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_name = f"page_{page_num+1}_img_{img_index+1}.{image_ext}"
            with open(os.path.join(output_dir, image_name), "wb") as f:
                f.write(image_bytes)
    doc.close()

current_directory = os.getcwd()
test_path = "/home/zhanghaoyu7/下载/Essential Java.《Java 编程要点》.pdf"
extract_images_pymupdf(test_path, current_directory)
