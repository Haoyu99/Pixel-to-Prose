from google import generativeai as genai
import os
from typing import List, Dict
import base64
import time
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz
import uuid
import atexit

class PDFToMarkdownConverter:
    def __init__(self, api_key: str, api_endpoint: str = None, chunk_size: int = 20, max_retries: int = 3):
        """
        初始化转换器
        :param api_key: Gemini API密钥
        :param api_endpoint: 代理服务器地址
        :param chunk_size: 每个分块的页数
        :param max_retries: 最大重试次数
        """
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        # 基础配置
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        # 创建临时目录
        self.temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
        self.logger.info(f"临时文件目录: {self.temp_dir}")

        # 设置API端点
        if api_endpoint:
            os.environ['GOOGLE_API_BASE_URL'] = api_endpoint
        # 配置API
        genai.configure(
            api_key=api_key,
            transport="rest"
        )

        # 初始化模型
        self.model = genai.GenerativeModel('gemini-2.0-flash')

        # 注册退出时的清理函数
        atexit.register(self._cleanup_temp_dir)



    def _cleanup_temp_dir(self):
        """
        清理临时目录
        """
        try:
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    try:
                        file_path = os.path.join(self.temp_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        self.logger.warning(f"清理临时文件失败: {str(e)}")
                os.rmdir(self.temp_dir)
                self.logger.info("临时目录已清理")
        except Exception as e:
            self.logger.warning(f"清理临时目录失败: {str(e)}")


    def _split_pdf(self, pdf_path: str) -> List[str]:
        """
        将PDF分割成小块
        :param pdf_path: PDF文件路径
        :return: 临时PDF文件路径列表
        """
        temp_files = []
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        self.logger.info(f"PDF总页数: {total_pages}")
        for start in range(0, total_pages, self.chunk_size):
            end = min(start + self.chunk_size, total_pages)
            # 创建新的PDF文档
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
            # 使用UUID创建唯一的临时文件名
            temp_file_path = os.path.join(self.temp_dir, f'chunk_{uuid.uuid4().hex}.pdf')
            new_doc.save(temp_file_path)
            new_doc.close()
            temp_files.append(temp_file_path)
        doc.close()
        return temp_files

    @staticmethod
    def _read_pdf_file(pdf_path: str) -> dict:
        """
        读取PDF文件
        :param pdf_path: PDF文件路径
        :return: 包含文件内容的字典
        """
        try:
            with open(pdf_path, 'rb') as file:
                content = file.read()
                base64_content = base64.b64encode(content).decode('utf-8')
            return {
                "mime_type": "application/pdf",
                "data": base64_content
            }
        except Exception as e:
            raise Exception(f"PDF文件读取失败: {str(e)}")




    def _process_chunk(self, pdf_path: str, chunk_index: int) -> Dict:
        """
        处理单个PDF分块
        :param pdf_path: PDF分块文件路径
        :param chunk_index: 分块索引
        :return: 包含处理结果的字典
        """
        for attempt in range(self.max_retries):
            try:
                pdf_data = self._read_pdf_file(pdf_path)
                # 获取当前分块的页数信息
                doc = fitz.open(pdf_path)
                page_info = f"(页码 {chunk_index * self.chunk_size + 1} - {chunk_index * self.chunk_size + doc.page_count})"
                doc.close()
                prompt = """You are an expert OCR assistant. Your job is to extract all text from the provided image and convert it into a well-structured, easy-to-read Markdown document that mirrors the intended structure of the original. Follow these precise guidelines:
                - Use Markdown headings, paragraphs, lists, and tables to match the document’s hierarchy and flow.
                - For tables, use standard Markdown table syntax and merge cells if needed. If a table has a title, include it as plain text above the table.
                - Render mathematical formulas with LaTeX syntax: use $...$ for inline and $$...$$ for display equations.
                - For images, use the syntax ![descriptive alt text](link) with a clear, descriptive alt text.
                - Remove unnecessary line breaks so that the text flows naturally without awkward breaks.
                - Your final Markdown output must be direct text (do not wrap it in code blocks). Ensure your output is clear, accurate, and faithfully reflects the original image’s content and structure."""

                response = self.model.generate_content([prompt, pdf_data])
                return {
                    'index': chunk_index,
                    'content': response.text,
                    'success': True
                }
            except Exception as e:
                self.logger.error(f"处理分块{chunk_index}第{attempt + 1}次尝试失败: {str(e)}")
                time.sleep(2 ** attempt)  # 指数退避策略

                if attempt == self.max_retries - 1:
                    return {
                        'index': chunk_index,
                        'content': f"<!-- 处理失败: {str(e)} -->",
                        'success': False
                    }
    def convert_to_markdown(self, pdf_path: str,
                            parallel: bool = True) -> str:
        """
        将PDF转换为Markdown
        :param pdf_path: PDF文件路径
        :param parallel: 是否并行处理
        :return: Markdown格式的文本
        """
        start_time = time.time()
        self.logger.info("开始转换... " )

        temp_files = []
        try:
            # 分割PDF
            temp_files = self._split_pdf(pdf_path)
            chunks_count = len(temp_files)
            self.logger.info(f"PDF已分割为{chunks_count}个块")
            results = []
            # processed_count = 0  # Removed this line

            if parallel:
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(self._process_chunk, temp_file, idx)
                        for idx, temp_file in enumerate(temp_files)
                    ]

                    # 使用tqdm显示进度
                    with tqdm(total=chunks_count, desc="处理进度", unit="块") as pbar:
                        for future in as_completed(futures):
                            # processed_count += 1 # Removed this line
                            # if progress_callback:  # Removed callback since tqdm handles it
                            #     progress_callback(processed_count, chunks_count)
                            results.append(future.result())
                            pbar.update(1)  # Update tqdm progress bar
            else:
                # 使用tqdm显示进度
                with tqdm(total=chunks_count, desc="处理进度", unit="块") as pbar:
                    for idx, temp_file in enumerate(temp_files):
                        result = self._process_chunk(temp_file, idx)
                        # processed_count += 1 # Removed this line
                        # if progress_callback:  # Removed callback
                        #     progress_callback(processed_count, chunks_count)
                        results.append(result)
                        pbar.update(1)

            # 按索引排序并合并结果
            results.sort(key=lambda x: x['index'])
            markdown_content = "\n\n".join(result['content'] for result in results)
            # 计算并显示耗时
            duration = time.time() - start_time
            if duration < 60:
                time_str = f"{duration:.2f}秒"
            elif duration < 3600:
                time_str = f"{duration / 60:.2f}分钟"
            else:
                time_str = f"{duration / 3600:.2f}小时"
            # 输出转换统计信息
            self.logger.info(f"转换完成！耗时: {time_str}")
            # 统计成功率
            success_count = sum(1 for r in results if r['success'])
            self.logger.info(f"处理成功率: {success_count}/{chunks_count} " +
                             f"({success_count / chunks_count * 100:.2f}%)")

            return markdown_content
        except Exception as e:
            error_msg = f"转换过程出错: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    self.logger.warning(f"清理临时文件失败: {str(e)}")



    def save_markdown(self, markdown_text: str, output_path: str):
        """
        保存Markdown文件
        :param markdown_text: Markdown文本
        :param output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            self.logger.info(f"Markdown文件已保存至: {output_path}")
        except Exception as e:
            raise Exception(f"文件保存错误: {str(e)}")



def main():
    # 配置信息
    api_key = ""  # 你的API密钥
    api_endpoint = "http://xxx.xxxxxxx.com"  # 你的代理服务器地址
    # 文件路径
    pdf_path = ""  # 替换为你的PDF文件路径
    output_path = ("output-doc1.md")
    try:
        # 根据转换模式设置输出文件名
        base_name, ext = os.path.splitext(output_path)

        # 初始化转换器
        converter = PDFToMarkdownConverter(
            api_key=api_key,
            api_endpoint=api_endpoint,
            chunk_size=3,  # 这个值建议不要超过5, 太大会导致请求失败
            max_retries=3
        )

        markdown_content = converter.convert_to_markdown(
            pdf_path=pdf_path,
            parallel=True
        )
        # 保存结果
        converter.save_markdown(markdown_content, output_path)
        print(f"转换完成！文件已保存至: {output_path}")
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()