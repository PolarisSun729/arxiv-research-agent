from pdfplumber import PDF
import pandas as pd
from typing import List, Dict
import re

class PDFParser:
    def __init__(self, pdf_path: str):
        # 使用 PDF.open() 而不是 pdfplumber.open()
        self.pdf = PDF.open(pdf_path)
        
    def extract_text_by_page(self, page) -> str:
        """提取单页文本,按照从上到下、从左到右的顺序"""
        text_elements = []
        
        # 获取页面上所有文本元素
        for element in page.extract_words(x_tolerance=3, y_tolerance=3):
            # 保存文本及其位置信息
            text_elements.append({
                'text': element['text'],
                'x0': element['x0'],
                'top': element['top'],
                'bottom': element['bottom']
            })
        
        # 按照位置信息排序文本(先上下后左右)
        line_height = 10  # 估计的行高
        sorted_elements = sorted(text_elements, 
                               key=lambda x: (round(x['top']/line_height), x['x0']))
        
        # 合并同一行的文本
        current_line = []
        result = []
        current_line_number = 0
        
        for element in sorted_elements:
            line_number = round(element['top']/line_height)
            
            if line_number == current_line_number:
                current_line.append(element['text'])
            else:
                if current_line:
                    result.append(' '.join(current_line))
                current_line = [element['text']]
                current_line_number = line_number
                
        if current_line:
            result.append(' '.join(current_line))
            
        return '\n'.join(result)
    
    def extract_tables(self, page) -> List[pd.DataFrame]:
        """提取页面中的表格"""
        tables = []
        for table in page.extract_tables():
            if table:  # 确保表格不为空
                # 过滤掉空行和只包含空格的行
                filtered_table = [[cell.strip() if isinstance(cell, str) else cell 
                                 for cell in row] 
                                for row in table 
                                if any(cell and str(cell).strip() for cell in row)]
                
                if filtered_table:  # 确保过滤后的表格不为空
                    try:
                        df = pd.DataFrame(filtered_table[1:], columns=filtered_table[0])
                        tables.append(df)
                    except Exception as e:
                        print(f"处理表格时出错: {str(e)}")
                        continue
        return tables
    
    def parse(self) -> Dict:
        """解析整个PDF文档"""
        result = {
            'text': [],
            'tables': []
        }
        
        for i, page in enumerate(self.pdf.pages):
            try:
                # 提取文本
                page_text = self.extract_text_by_page(page)
                result['text'].append(page_text)
                
                # 提取表格
                page_tables = self.extract_tables(page)
                if page_tables:
                    result['tables'].extend(page_tables)
                    
            except Exception as e:
                print(f"处理第 {i+1} 页时出错: {str(e)}")
                continue
        
        return result
    
    def clean_text(self, text: str) -> str:
        """清理提取的文本"""
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        # 移除特殊字符，但保留某些有用的符号
        text = re.sub(r'[^\w\s.,;:!?\-–—()[\]{}"/]', '', text)
        return text
    
    def close(self):
        """关闭PDF文件"""
        self.pdf.close()

def process_pdf(pdf_path: str) -> Dict:
    """处理PDF文件的主函数"""
    try:
        parser = PDFParser(pdf_path)
        result = parser.parse()
        
        # 清理文本
        result['text'] = [parser.clean_text(text) for text in result['text']]
        
        # 合并所有文本
        result['full_text'] = '\n\n'.join(result['text'])
        
        return result
    except Exception as e:
        print(f"处理PDF文件时出错: {str(e)}")
        raise
    finally:
        if 'parser' in locals():
            parser.close()

# 使用示例
if __name__ == "__main__":
    pdf_path = "Boeing-Sustainability-Report-2023.pdf"  # 确保文件路径正确
    try:
        result = process_pdf(pdf_path)
        
        # 打印文本内容
        print("提取的文本内容:")
        print(result['full_text'][:1000])  # 打印前1000个字符
        
        # 打印表格信息
        print(f"\n提取的表格数量: {len(result['tables'])}")
        for i, table in enumerate(result['tables'][:6]):
            print(f"\n表格 {i+1}:")
            print(table.head())
            
    except FileNotFoundError:
        print(f"找不到PDF文件: {pdf_path}")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")