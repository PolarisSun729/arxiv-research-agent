from unstructured.partition.pdf import partition_pdf
from collections import defaultdict

# 提取 PDF 的表格内容
elements = partition_pdf(
    filename="data/Boeing-Sustainability-Report/1_page/Boeing-Sustainability-Report_074.pdf",
    strategy="hi_res",
    infer_table_structure=True,
)

# 创建ID到元素的映射
id_to_element = {}
# 创建父ID到子元素的映射
parent_to_children = defaultdict(list)

# 第一遍遍历：建立映射关系
for element in elements:
    # 获取元素的ID（如果有的话）
    element_id = getattr(element, 'id', None)
    if element_id:
        id_to_element[element_id] = element
    
    # 获取父ID并建立关系
    parent_id = getattr(element.metadata, 'parent_id', None)
    if parent_id:
        parent_to_children[parent_id].append(element)

# 第二遍遍历：输出层级关系
for element in elements:
    print("\n" + "="*50)
    print(f"Element Type: {element.__class__.__name__}")
    print(f"Element Content: {element.text}")
    
    # 打印元数据
    # print("\nMetadata:")
    # for key, value in element.metadata.__dict__.items():
    #     print(f"  {key}: {value}")
    
    # 如果是表格，额外打印表格特定信息
    # if element.__class__.__name__ == "Table":
    #     print("\nTable Specific Info:")
    #     print(f"Table HTML: {element.metadata.text_as_html if hasattr(element.metadata, 'text_as_html') else 'N/A'}")
    
    # 打印层级关系
    element_id = getattr(element, 'id', None)
    parent_id = getattr(element.metadata, 'parent_id', None)
    
    if element_id:
        print(f"\nElement ID: {element_id}")
        # 打印子元素
        children = parent_to_children.get(element_id, [])
        if children:
            print("\nChild Elements:")
            for child in children:
                print(f"  - Type: {child.__class__.__name__}")
                print(f"    Content: {child.text[:100]}...")  # 只显示前100个字符
    
    if parent_id:
        print(f"\nParent ID: {parent_id}")
        parent = id_to_element.get(parent_id)
        if parent:
            print(f"Parent Element:")
            print(f"  Type: {parent.__class__.__name__}")
            print(f"  Content: {parent.text[:100]}...")  # 只显示前100个字符
    
    print("="*50)



