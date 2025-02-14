from unstructured.partition.pdf import partition_pdf
import json
# Returns a List[Element] present in the pages of the parsed pdf document
elements = partition_pdf(
    # "data/Boeing-Sustainability-Report.pdf",
    "data/test.pdf",
    abs = True,
    strategy="fast",
    # chunking_strategy="by_title",
    include_orig_eleme2nts=True,
    abs = True
)

# Applies the English and Swedish language pack for ocr. OCR is only applied
# if the text is not available in the PDF.
# elements = partition_pdf("example-docs/pdf/layout-parser-paper-fast.pdf", languages=["eng", "swe"])

print(len(elements))

# # 打印前2个元素的详细信息
# for i, element in enumerate(elements[:2], 1):
#     print(f"\n--- Element {i} ---")
#     print(f"Type: {element.__class__.__name__}")
#     print(f"Content: {str(element)}")
#     print("Metadata:")
#     for key, value in element.metadata.__dict__.items():
#         print(f"  {key}: {value}")


# # 打印每一种不同类型的元素2个
# for element_type in set(element.__class__.__name__ for element in elements):
#     print(f"\n--- Element Type: {element_type} ---")
#     count = 0
#     for element in elements:
#         if element.__class__.__name__ == element_type and count < 2:
#             print(f"\nElement {count + 1}:")
#             print(f"Content: {str(element)}")
#             print("Metadata:")
#             for key, value in element.metadata.__dict__.items():
#                 print(f"  {key}: {value}")
#             count += 1


# #总结一下，这个PDF包含多少类型的元素，每种元素个数  
# print(f"Total number of element types: {len(set(element.__class__.__name__ for element in elements))}")
# for element_type in set(element.__class__.__name__ for element in elements):
#     print(f"Number of {element_type} elements: {sum(1 for element in elements if element.__class__.__name__ == element_type)}")

# # 保存所有元素到json
# elements_list = []
# for i, element in enumerate(elements, 1):
#     # 将 metadata 对象转换为可序列化的字典
#     metadata_dict = {}
#     for key, value in element.metadata.__dict__.items():
#         # 处理不可序列化的对象
#         if key == 'coordinates':
#             if value:  # 如果coordinates存在
#                 points = value.points
#                 metadata_dict[key] = {
#                     'points': [
#                         {'x': point[0], 'y': point[1]} 
#                         for point in points
#                     ]
#                 }
#             else:
#                 metadata_dict[key] = None
#         elif isinstance(value, frozenset):
#             metadata_dict[key] = list(value)  # 将 frozenset 转换为 list
#         else:
#             metadata_dict[key] = value

#     element_dict = {
#         "element_number": i,
#         "type": element.__class__.__name__,
#         "content": str(element),
#         "metadata": metadata_dict
#     }
#     elements_list.append(element_dict)

# with open("PDF_loaders/elements.json", "w") as f:
#     json.dump(elements_list, f, indent=4)


