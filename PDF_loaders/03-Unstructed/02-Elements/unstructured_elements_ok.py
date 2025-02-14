from collections import defaultdict, Counter
from typing import List, Any
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    ListItem,
    NarrativeText,
    Title,
    Header,
    Image,
    Text,
    FigureCaption,
    Table,
    Footer
)

def format_element_content(element: Any, max_length: int = 200) -> str:
    """格式化元素内容，添加合适的截断和格式"""
    content = []
    
    # 文本内容
    if hasattr(element, 'text'):
        text = element.text
        if len(text) > max_length:
            text = text[:max_length] + '...'
        content.append(f"文本内容: {text}")
    
    # 表格内容
    if isinstance(element, Table):
        content.append("表格内容:")
        if hasattr(element, 'text'):
            content.append(element.text[:max_length] + ('...' if len(element.text) > max_length else ''))
    
    # 图片信息
    if isinstance(element, Image):
        content.append("图片元素")
        if hasattr(element, 'source'):
            content.append(f"图片来源: {element.source}")
    
    # 坐标信息
    if hasattr(element, 'coordinates'):
        coord_info = element.coordinates
        content.append(f"坐标信息: {coord_info}")
    
    # 元数据
    if hasattr(element, 'metadata'):
        try:
            metadata = element.metadata.to_dict()
            content.append(f"元数据: {metadata}")
        except:
            content.append(f"元数据: {element.metadata}")
    
    return '\n'.join(content)

def show_all_element_types(file_path: str, num_examples: int = 5) -> None:
    """
    展示所有类型元素的示例
    
    Args:
        file_path: PDF文件路径
        num_examples: 每种类型要展示的示例数量
    """
    # 解析PDF
    elements = partition_pdf(
        filename=file_path,
        strategy="fast",
        include_metadata=True,
        include_page_breaks=True,
        include_coords=True
    )
    
    # 获取类型统计
    type_counter = Counter(type(element) for element in elements)
    
    # 按类型分组元素
    element_groups = defaultdict(list)
    for element in elements:
        element_groups[type(element)].append(element)
    
    # 打印总体统计
    print("\n元素类型统计:")
    print("="*50)
    for element_type, count in type_counter.most_common():
        print(f"{element_type.__name__}: {count}")
    
    # 为每种类型打印示例
    for element_type, elements_list in element_groups.items():
        print(f"\n{'='*50}")
        print(f"\n{element_type.__name__} - 总数: {len(elements_list)}")
        print(f"{'='*50}")
        
        # 获取指定数量的示例
        examples = elements_list[:num_examples]
        
        for i, example in enumerate(examples, 1):
            print(f"\n示例 {i}:")
            print("-" * 40)
            print(format_element_content(example))
            print("-" * 40)

def get_element_type_samples(elements: List[Any], num_samples: int = 5) -> dict:
    """
    获取每种类型元素的样本
    
    Args:
        elements: 元素列表
        num_samples: 每种类型要获取的样本数量
    
    Returns:
        包含每种类型样本的字典
    """
    samples = defaultdict(list)
    for element in elements:
        element_type = type(element).__name__
        if len(samples[element_type]) < num_samples:
            samples[element_type].append(element)
    return dict(samples)

if __name__ == "__main__":
    file_path = "山西-en.pdf"
    try:
        print("正在分析PDF文件...")
        show_all_element_types(file_path)
        
    except FileNotFoundError:
        print(f"找不到PDF文件: {file_path}")
    except Exception as e:
        print(f"处理PDF时出错: {str(e)}")
        raise