from unstructured.partition.pdf import partition_pdf
from collections import defaultdict

# 提取 PDF 的表格内容
elements = partition_pdf(
    filename="data/Boeing-Sustainability-Report/1_page/Boeing-Sustainability-Report_074.pdf",
    strategy="hi_res",
    infer_table_structure=True,
)

# 创建父ID到子元素的映射
parent_to_children = defaultdict(list)

# 第一遍遍历：建立父子关系映射
for element in elements:
    parent_id = getattr(element.metadata, 'parent_id', None)
    if parent_id:
        parent_to_children[parent_id].append(element)

# 第二遍遍历：只输出有父子关系的元素
for element in elements:
    # 获取当前元素的ID
    element_id = getattr(element, 'id', None)
    
    # 如果这个元素有子元素，打印它和它的所有子元素
    if element_id and element_id in parent_to_children:
        print("\n" + "="*50)
        print("Parent Element:")
        print(f"Type: {element.__class__.__name__}")
        print(f"Content: {element.text[:200]}...")  # 限制内容长度
        
        print("\nChildren Elements:")
        for child in parent_to_children[element_id]:
            print(f"\n  - Type: {child.__class__.__name__}")
            print(f"    Content: {child.text[:200]}...")  # 限制内容长度
        
        print("="*50)