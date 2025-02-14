from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Title, 
    NarrativeText,
    Table,
    ListItem,
    PageBreak,
    Image,
    Text
)
from typing import List, Dict
import pandas as pd

class UnstructuredPDFParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def process_table(self, table_element: Table) -> pd.DataFrame:
        """处理表格元素"""
        if hasattr(table_element, 'text'):
            # 将表格文本分割成行
            rows = table_element.text.split('\n')
            # 将每行分割成单元格
            table_data = [row.split() for row in rows if row.strip()]
            
            if table_data:
                try:
                    # 如果第一行的列数比其他行多，可能是标题行
                    if len(table_data[0]) > max(len(row) for row in table_data[1:]):
                        headers = table_data[0]
                        data = table_data[1:]
                    else:
                        # 使用默认列名
                        headers = [f'Column_{i}' for i in range(len(table_data[0]))]
                        data = table_data
                        
                    # 确保所有行的长度相同
                    max_cols = len(headers)
                    padded_data = [row + [''] * (max_cols - len(row)) for row in data]
                    
                    return pd.DataFrame(padded_data, columns=headers)
                except Exception as e:
                    print(f"转换表格时出错: {str(e)}")
                    return pd.DataFrame()
        return pd.DataFrame()

    def parse(self) -> Dict:
        """解析PDF文档并保留结构化信息"""
        elements = partition_pdf(
            filename=self.file_path,
            include_page_breaks=True,
            strategy="hi_res",
            extract_images_in_pdf=True,
            include_metadata=True,
            include_coords=True
        )
        
        result = {
            'titles': [],
            'text': [],
            'tables': [],
            'coordinates': [],
            'images': [],
            'metadata': []
        }
        
        for element in elements:
            # 提取坐标信息
            coords = None
            if hasattr(element, 'coordinates'):
                coords = {
                    'points': element.coordinates.points,
                    'page': element.coordinates.page_number if hasattr(element.coordinates, 'page_number') else None
                }
                result['coordinates'].append(coords)
            
            # 提取元数据
            metadata = None
            if hasattr(element, 'metadata'):
                metadata = element.metadata.to_dict()
                result['metadata'].append(metadata)
            
            # 根据元素类型处理
            if isinstance(element, Title):
                result['titles'].append({
                    'text': element.text,
                    'coords': coords,
                    'metadata': metadata
                })
                
            elif isinstance(element, Table):
                df = self.process_table(element)
                if not df.empty:
                    result['tables'].append({
                        'data': df,
                        'coords': coords,
                        'metadata': metadata,
                        'raw_text': element.text if hasattr(element, 'text') else ''
                    })
                        
            elif isinstance(element, Image):
                result['images'].append({
                    'metadata': metadata,
                    'coords': coords
                })
                
            elif isinstance(element, (Text, NarrativeText)):
                result['text'].append({
                    'text': element.text,
                    'coords': coords,
                    'metadata': metadata
                })
        
        return result

    def get_text_by_page(self, page_num: int) -> List[Dict]:
        """获取特定页面的文本内容"""
        page_elements = partition_pdf(
            filename=self.file_path,
            include_page_breaks=True,
            strategy="hi_res",
            include_coords=True,
            include_metadata=True,
            start_page=page_num,
            end_page=page_num
        )
        
        return [{
            'text': element.text,
            'coords': element.coordinates if hasattr(element, 'coordinates') else None,
            'metadata': element.metadata.to_dict() if hasattr(element, 'metadata') else None
        } for element in page_elements if isinstance(element, (Text, NarrativeText, Title))]

def process_pdf(file_path: str) -> Dict:
    """处理PDF文件的主函数"""
    parser = UnstructuredPDFParser(file_path)
    try:
        result = parser.parse()
        
        # 打印基本统计信息
        print(f"找到 {len(result['titles'])} 个标题")
        print(f"找到 {len(result['tables'])} 个表格")
        print(f"找到 {len(result['images'])} 个图片")
        print(f"找到 {len(result['text'])} 段文本")
        
        return result
    
    except Exception as e:
        print(f"处理PDF时出错: {str(e)}")
        raise

if __name__ == "__main__":
    file_path = "山西-en.pdf"
    try:
        result = process_pdf(file_path)
        
        # 打印第一个标题（如果存在）
        if result['titles']:
            print("\n第一个标题:")
            print(result['titles'][0]['text'])
        
        # 打印第一个表格（如果存在）
        if result['tables']:
            print("\n第一个表格:")
            print(result['tables'][0]['data'].head())
            print("\n表格原始文本:")
            print(result['tables'][0]['raw_text'])
        
        # 打印第一段文本的坐标（如果存在）
        if result['coordinates']:
            print("\n第一段文本的坐标:")
            print(result['coordinates'][0])
            
    except FileNotFoundError:
        print(f"找不到PDF文件: {file_path}")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")