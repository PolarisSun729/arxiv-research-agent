from unstructured.chunking.basic import chunk_elements
from unstructured.partition.html import partition_html

url = "https://understandingwar.org/backgrounder/russian-offensive-campaign-assessment-august-27-2023-0"
elements = partition_html(url=url)
# 打印元素
for element in elements:
    print(f"Type: {element.__class__.__name__}")
    print(f"Content: {element.text[:200]}...")  # Limit content length for readability
    print("-" * 50)

chunks = chunk_elements(elements,
                        # multipage_sections=True,
                        )

print(chunks)

for chunk in chunks:
    print(chunk)
    print("\n\n" + "-"*80)


from unstructured.chunking.title import chunk_by_title

chunks = chunk_by_title(elements,
                        # multipage_sections=True,
                        include_orig_elements=True)

for chunk in chunks:
    print(chunk)
    print("\n\n" + "-"*80)