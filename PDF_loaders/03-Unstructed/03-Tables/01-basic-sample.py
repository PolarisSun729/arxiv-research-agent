from unstructured.partition.pdf import partition_pdf

# 提取 PDF 的表格内容
elements = partition_pdf(
    filename="data/Boeing-Sustainability-Report/1_page/Boeing-Sustainability-Report_074.pdf",
    # strategy="hi_res",
    strategy="ocr_only",
    infer_table_structure=True,  # 启用表格结构推断
)

# 提取并打印元素内容
for element in elements:
    print("\n" + "="*50)
    print(f"Element Type: {element.__class__.__name__}")
    print(f"Element Content: {element.text}")
    
    # 打印元数据
    print("\nMetadata:")
    for key, value in element.metadata.__dict__.items():
        print(f"  {key}: {value}")
    
    # 如果是表格，额外打印表格特定信息
    if element.__class__.__name__ == "Table":
        print("\nTable Specific Info:")
        print(f"Table HTML: {element.metadata.text_as_html if hasattr(element.metadata, 'text_as_html') else 'N/A'}")
    
    print("="*50)



'''
Some weights of the model checkpoint at microsoft/table-transformer-structure-recognition were not used when initializing TableTransformerForObjectDetection: ['model.backbone.conv_encoder.model.layer2.0.downsample.1.num_batches_tracked', 'model.backbone.conv_encoder.model.layer3.0.downsample.1.num_batches_tracked', 'model.backbone.conv_encoder.model.layer4.0.downsample.1.num_batches_tracked']
- This IS expected if you are initializing TableTransformerForObjectDetection from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing TableTransformerForObjectDetection from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).

==================================================
Element Type: NarrativeText
Element Content: 2023 Boeing Sustainability Report

Metadata:
  detection_class_prob: 0.429416686296463
  coordinates: CoordinatesMetadata(points=((93.09353637695312, 95.73809814453125), (93.09353637695312, 161.88992309570312), (371.705322265625, 161.88992309570312), (371.705322265625, 95.73809814453125)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c6ad0>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: NarrativeText
Element Content: Contents

Metadata:
  detection_class_prob: 0.5818367600440979
  coordinates: CoordinatesMetadata(points=((96.05791473388672, 280.80731201171875), (96.05791473388672, 312.9308166503906), (211.20396423339844, 312.9308166503906), (211.20396423339844, 280.80731201171875)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c6080>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: NarrativeText
Element Content: Introduction

Metadata:
  detection_class_prob: 0.5733087062835693
  coordinates: CoordinatesMetadata(points=((97.63151550292969, 374.3335266113281), (97.63151550292969, 406.4699401855469), (246.94886779785156, 406.4699401855469), (246.94886779785156, 374.3335266113281)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c4c40>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: NarrativeText
Element Content: Approach & Governance

Metadata:
  detection_class_prob: 0.5653443932533264
  coordinates: CoordinatesMetadata(points=((100.0, 467.27496337890625), (100.0, 498.27801513671875), (385.87750244140625, 498.27801513671875), (385.87750244140625, 467.27496337890625)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c5b10>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: NarrativeText
Element Content: People

Metadata:
  detection_class_prob: 0.6341114640235901
  coordinates: CoordinatesMetadata(points=((96.72066497802734, 560.71630859375), (96.72066497802734, 593.1286010742188), (182.09583333333333, 593.1286010742188), (182.09583333333333, 560.71630859375)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c59c0>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: NarrativeText
Element Content: Products & Services

Metadata:
  detection_class_prob: 0.38687294721603394
  coordinates: CoordinatesMetadata(points=((99.82726287841797, 657.293212890625), (99.82726287841797, 687.9769287109375), (343.2833557128906, 687.9769287109375), (343.2833557128906, 657.293212890625)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c5870>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: NarrativeText
Element Content: Operations

Metadata:
  detection_class_prob: 0.6187150478363037
  coordinates: CoordinatesMetadata(points=((97.57228088378906, 751.3436889648438), (97.57228088378906, 783.2388916015625), (232.6566162109375, 783.2388916015625), (232.6566162109375, 751.3436889648438)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c5720>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: NarrativeText
Element Content: Communities

Metadata:
  detection_class_prob: 0.547609806060791
  coordinates: CoordinatesMetadata(points=((97.15229797363281, 843.9389038085938), (97.15229797363281, 874.807861328125), (262.3667907714844, 874.807861328125), (262.3667907714844, 843.9389038085938)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c55d0>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: Reporting

Metadata:
  coordinates: CoordinatesMetadata(points=((100.0, 940.2580555555555), (100.0, 966.6469444444444), (224.58194444444442, 966.6469444444444), (224.58194444444442, 940.2580555555555)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c5450>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: Key ESG Data

Metadata:
  coordinates: CoordinatesMetadata(points=((100.0, 990.2927777777778), (100.0, 1014.5983333333334), (254.85069444444449, 1014.5983333333334), (254.85069444444449, 990.2927777777778)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c51b0>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: GRI Index

Metadata:
  coordinates: CoordinatesMetadata(points=((100.0, 1038.73375), (100.0, 1063.0393055555555), (208.01388888888886, 1063.0393055555555), (208.01388888888886, 1038.73375)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c5090>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: SASB Index

Metadata:
  coordinates: CoordinatesMetadata(points=((100.0, 1087.1747222222223), (100.0, 1111.480277777778), (230.52083333333331, 1111.480277777778), (230.52083333333331, 1087.1747222222223)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c4dc0>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: TCFD Index

Metadata:
  coordinates: CoordinatesMetadata(points=((100.0, 1135.6156944444447), (100.0, 1159.92125), (229.18402777777774, 1159.92125), (229.18402777777774, 1135.6156944444447)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c4a30>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: NarrativeText
Element Content: U.N. Sustainable Development Goals

Metadata:
  detection_class_prob: 0.46554189920425415
  coordinates: CoordinatesMetadata(points=((93.7938003540039, 1179.353271484375), (93.7938003540039, 1245.013671875), (341.3489990234375, 1245.013671875), (341.3489990234375, 1179.353271484375)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c4b20>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  parent_id: 9dcd82de6188704cd0dac625c114a1c8
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: NarrativeText
Element Content: Awards and Recognition

Metadata:
  detection_class_prob: 0.3508531451225281
  coordinates: CoordinatesMetadata(points=((93.59639739990234, 1261.9342041015625), (93.59639739990234, 1292.0953369140625), (389.89617919921875, 1292.0953369140625), (389.89617919921875, 1261.9342041015625)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c49d0>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  parent_id: 9dcd82de6188704cd0dac625c114a1c8
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: NarrativeText
Element Content: Select Memberships and Partnerships

Metadata:
  detection_class_prob: 0.5430867671966553
  coordinates: CoordinatesMetadata(points=((97.56839752197266, 1310.9390869140625), (97.56839752197266, 1371.7149658203125), (340.5425109863281, 1371.7149658203125), (340.5425109863281, 1310.9390869140625)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c4880>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  parent_id: 9dcd82de6188704cd0dac625c114a1c8
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: NarrativeText
Element Content: Forward-Looking Statements

Metadata:
  detection_class_prob: 0.6954914927482605
  coordinates: CoordinatesMetadata(points=((96.65028381347656, 1391.9527587890625), (96.65028381347656, 1452.5465087890625), (299.5461120605469, 1452.5465087890625), (299.5461120605469, 1391.9527587890625)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c4730>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  parent_id: 9dcd82de6188704cd0dac625c114a1c8
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Footer
Element Content: 74

Metadata:
  detection_class_prob: 0.6663417816162109
  coordinates: CoordinatesMetadata(points=((100.0, 1593.9400634765625), (100.0, 1625.1864013671875), (124.71111111111111, 1625.1864013671875), (124.71111111111111, 1593.9400634765625)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c4430>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: Key ESG Data

Metadata:
  detection_class_prob: 0.7763946652412415
  coordinates: CoordinatesMetadata(points=((563.5057983398438, 208.2432098388672), (563.5057983398438, 299.8047790527344), (1082.8265380859375, 299.8047790527344), (1082.8265380859375, 208.2432098388672)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c61d0>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: NarrativeText
Element Content: Environmental Data

Metadata:
  detection_class_prob: 0.312610924243927
  coordinates: CoordinatesMetadata(points=((571.7937622070312, 308.58203125), (571.7937622070312, 337.34893798828125), (783.3422222222222, 337.34893798828125), (783.3422222222222, 308.58203125)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c5f00>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  parent_id: ab5a06ab3f755f7b4dcdce9f07f5f7f9
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Table
Element Content: 2022 2021 2020 Energy1 Megawatt hours Terajoules Megawatt hours Terajoules Megawatt hours Terajoules Natural gas 1,928,000 6,941 1,712,000 6,163 1,686,000 6,070 Jet kerosene 861,000 3,100 804,000 2,894 544,000 1,958 Fuel oil #2 127,000 457 153,000 551 149,000 536 Motor gasoline 24,000 86 21,000 76 21,000 76 Propane 11,000 40 10,000 36 12,000 43 Liquefied petroleum gas 2,000 7 1,000 4 – – Total nonrenewable fuels 2,953,000 10,631 2,701,000 9,724 2,412,000 8,683 Sustainable aviation fuel 9,000 32 4,000 14 2,000 7 Total renewable fuels 9,000 32 4,000 14 2,000 7 Purchased nonrenewable electricity 1,350,000 4,860 1,482,000 5,335 1,686,000 6,070 Purchased renewable electricity2 720,000 2,592 574,000 2,066 392,000 1,411 Total purchased electricity 2,070,000 7,452 2,056,000 7,402 2,078,000 7,481 Total energy use 5,033,000 18,119 4,761,000 17,410 4,492,000 16,171

Metadata:
  detection_class_prob: 0.9127820134162903
  coordinates: CoordinatesMetadata(points=((556.7009887695312, 341.2417907714844), (556.7009887695312, 962.9207153320312), (2693.298583984375, 962.9207153320312), (2693.298583984375, 341.2417907714844)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c5db0>)
  last_modified: 2024-11-28T10:49:00
  text_as_html: <table><thead><tr><th rowspan="2">Energy'</th><th colspan="2">2022</th><th colspan="2">2021</th><th colspan="2">2020</th></tr><tr><th>Megawatt hours</th><th>Terajoules</th><th>Megawatt hours</th><th>Terajoules</th><th>Megawatt hours</th><th>Terajoules</th></tr></thead><tbody><tr><td>Natural gas</td><td>1,928,000</td><td>6,941</td><td>1,712,000</td><td>6,163</td><td>1,686,000</td><td>6,070</td></tr><tr><td>Jet kerosene</td><td>861,000</td><td>3,100</td><td>804,000</td><td>2,894</td><td>544,000</td><td>1,958</td></tr><tr><td>Fuel oil #2</td><td>127,000</td><td>457</td><td>153,000</td><td>551</td><td>149,000</td><td>536</td></tr><tr><td>Motor gasoline</td><td>24,000</td><td>86</td><td>21,000</td><td>76</td><td>21,000</td><td>76</td></tr><tr><td>Propane</td><td>11,000</td><td>40</td><td>10,000</td><td>36</td><td>12,000</td><td>43</td></tr><tr><td>Liquefied petroleum gas</td><td>2,000</td><td>7</td><td>1,000</td><td>4</td><td>-</td><td>-</td></tr><tr><td>Total nonrenewable fuels</td><td>2,953,000</td><td>10,631</td><td>2,701,000</td><td>9,724</td><td>2,412,000</td><td>8,683</td></tr><tr><td>Sustainable aviation fuel</td><td>9,000</td><td>32</td><td>4,000</td><td>14</td><td>2,000</td><td>7</td></tr><tr><td>Total renewable fuels</td><td>9,000</td><td>32</td><td>4,000</td><td>14</td><td>2,000</td><td>7</td></tr><tr><td>Purchased nonrenewable electricity</td><td>1,350,000</td><td>4,860</td><td>1,482,000</td><td>5,335</td><td>1,686,000</td><td>6,070</td></tr><tr><td>Purchased renewable electricity”</td><td>720,000</td><td>2,592</td><td>574,000</td><td>2,066</td><td>392,000</td><td>1,411</td></tr><tr><td>Total purchased electricity</td><td>2,070,000</td><td>7,452</td><td>2,056,000</td><td>7,402</td><td>2,078,000</td><td>7,481</td></tr><tr><td>Total energy use</td><td>5,033,000</td><td>18,119</td><td>4,761,000</td><td>17,410</td><td>4,492,000</td><td>16,171</td></tr></tbody></table>
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  parent_id: ab5a06ab3f755f7b4dcdce9f07f5f7f9
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf

Table Specific Info:
Table HTML: <table><thead><tr><th rowspan="2">Energy'</th><th colspan="2">2022</th><th colspan="2">2021</th><th colspan="2">2020</th></tr><tr><th>Megawatt hours</th><th>Terajoules</th><th>Megawatt hours</th><th>Terajoules</th><th>Megawatt hours</th><th>Terajoules</th></tr></thead><tbody><tr><td>Natural gas</td><td>1,928,000</td><td>6,941</td><td>1,712,000</td><td>6,163</td><td>1,686,000</td><td>6,070</td></tr><tr><td>Jet kerosene</td><td>861,000</td><td>3,100</td><td>804,000</td><td>2,894</td><td>544,000</td><td>1,958</td></tr><tr><td>Fuel oil #2</td><td>127,000</td><td>457</td><td>153,000</td><td>551</td><td>149,000</td><td>536</td></tr><tr><td>Motor gasoline</td><td>24,000</td><td>86</td><td>21,000</td><td>76</td><td>21,000</td><td>76</td></tr><tr><td>Propane</td><td>11,000</td><td>40</td><td>10,000</td><td>36</td><td>12,000</td><td>43</td></tr><tr><td>Liquefied petroleum gas</td><td>2,000</td><td>7</td><td>1,000</td><td>4</td><td>-</td><td>-</td></tr><tr><td>Total nonrenewable fuels</td><td>2,953,000</td><td>10,631</td><td>2,701,000</td><td>9,724</td><td>2,412,000</td><td>8,683</td></tr><tr><td>Sustainable aviation fuel</td><td>9,000</td><td>32</td><td>4,000</td><td>14</td><td>2,000</td><td>7</td></tr><tr><td>Total renewable fuels</td><td>9,000</td><td>32</td><td>4,000</td><td>14</td><td>2,000</td><td>7</td></tr><tr><td>Purchased nonrenewable electricity</td><td>1,350,000</td><td>4,860</td><td>1,482,000</td><td>5,335</td><td>1,686,000</td><td>6,070</td></tr><tr><td>Purchased renewable electricity”</td><td>720,000</td><td>2,592</td><td>574,000</td><td>2,066</td><td>392,000</td><td>1,411</td></tr><tr><td>Total purchased electricity</td><td>2,070,000</td><td>7,452</td><td>2,056,000</td><td>7,402</td><td>2,078,000</td><td>7,481</td></tr><tr><td>Total energy use</td><td>5,033,000</td><td>18,119</td><td>4,761,000</td><td>17,410</td><td>4,492,000</td><td>16,171</td></tr></tbody></table>
==================================================

==================================================
Element Type: ListItem
Element Content: 1. Data represents 100% of the company. 2. Renewable electricity data excludes any renewable energy that is part of the grid by default, in alignment with SASB and other frameworks. Notably, Boeing operates in a number of grids that rely significantly on renewable sources. • Boeing did not sell any electricity, heating or cooling energy.

Metadata:
  coordinates: CoordinatesMetadata(points=((566.0, 973.7663888888889), (566.0, 1036.1052777777777), (2506.7305555555577, 1036.1052777777777), (2506.7305555555577, 973.7663888888889)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c5060>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  parent_id: ab5a06ab3f755f7b4dcdce9f07f5f7f9
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Table
Element Content: Emissions1 Tons CO2e Metric tons CO2e Tons CO2e Metric tons CO2e Tons CO2e Metric tons CO2e Scope 1 GHG2,3 708,000 642,000 675,000 612,000 611,000 554,000 Scope 2 GHG — location-based2,3 859,000 779,000 830,000 753,000 840,000 762,000 Scope 2 GHG — market-based2,3 442,000 401,000 493,000 447,000 580,000 526,000 Scope 3 GHG — business travel 205,000 186,000 97,000 88,000 101,000 92,000 Scope 3 GHG — use of sold products (Commercial Airplanes)3,6 400,000,000 363,000,000 306,000,000 278,000,000 246,000,000 223,000,000 Scope 3 GHG — use of sold products (Defense, Space & Security)3,6 24,000,000 22,000,000 24,000,000 22,000,000 22,000,000 20,000,000 Total calculated GHG excluding sold products 1,355,000 1,229,000 1,264,000 1,147,000 1,292,000 1,172,000 Core metrics sites GHG — location-based4 724,000 657,000 702,000 637,000 713,000 647,000 Core metrics sites GHG — market-based4 323,000 293,000 376,000 341,000 452,000 410,000 GHG Intensity5 $0.00002 $0.00002 $0.00002

Metadata:
  detection_class_prob: 0.8924164175987244
  coordinates: CoordinatesMetadata(points=((565.6591796875, 1046.9613037109375), (565.6591796875, 1502.2486572265625), (2696.497314453125, 1502.2486572265625), (2696.497314453125, 1046.9613037109375)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c4f10>)
  last_modified: 2024-11-28T10:49:00
  text_as_html: <table><thead><tr><th>Emissions’</th><th>Tons CO,e</th><th>Metric tons CO,e</th><th></th><th>Metric tons C!</th><th>Tons CO,e</th><th>Metric tons CO,e</th></tr></thead><tbody><tr><td>Scope 1 GHG?*</td><td>708,000</td><td>642,000</td><td>675,000</td><td>612,000</td><td>611,000</td><td>554,000</td></tr><tr><td>Scope 2 GHG — location-based?*</td><td>859,000</td><td>779,000</td><td>830,000</td><td>753,000</td><td>840,000</td><td>762,000</td></tr><tr><td>Scope 2 GHG — market-based”?</td><td>442,000</td><td>401,000</td><td>493,000</td><td>447,000</td><td>580,000</td><td>526,000</td></tr><tr><td>Scope 3 GHG — business travel</td><td>205,000</td><td>186,000</td><td>97,000</td><td>88,000</td><td>101,000</td><td>92,000</td></tr><tr><td>Scope 3 GHG — use of sold products (Commercial Airplanes)**</td><td>400,000,000</td><td>363,000,000</td><td>306,000,000</td><td>278,000,000</td><td>246,000,000</td><td>223,000,000</td></tr><tr><td>Scope 3 GHG — use of sold products (Defense, Space &amp; Security)**</td><td>24,000,000</td><td>22,000,000</td><td>24,000,000</td><td>22,000,000</td><td>22,000,000</td><td>20,000,000</td></tr><tr><td>Total calculated GHG excluding sold products</td><td>1,355,000</td><td>1,229,000</td><td>1,264,000</td><td>1,147,000</td><td>1,292,000</td><td>1,172,000</td></tr><tr><td>Core metrics sites GHG — location-based*</td><td>724,000</td><td>657,000</td><td>702,000</td><td>637,000</td><td>713,000</td><td>647,000</td></tr><tr><td>Core metrics sites GHG — market-based* GHG Intensity®</td><td>323,000</td><td>293,000 $0.00002</td><td colspan="2">376,000 341,000 $0.00002</td><td colspan="2">452,000 410,000 $0.00002</td></tr></tbody></table>
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  parent_id: ab5a06ab3f755f7b4dcdce9f07f5f7f9
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf

Table Specific Info:
Table HTML: <table><thead><tr><th>Emissions’</th><th>Tons CO,e</th><th>Metric tons CO,e</th><th></th><th>Metric tons C!</th><th>Tons CO,e</th><th>Metric tons CO,e</th></tr></thead><tbody><tr><td>Scope 1 GHG?*</td><td>708,000</td><td>642,000</td><td>675,000</td><td>612,000</td><td>611,000</td><td>554,000</td></tr><tr><td>Scope 2 GHG — location-based?*</td><td>859,000</td><td>779,000</td><td>830,000</td><td>753,000</td><td>840,000</td><td>762,000</td></tr><tr><td>Scope 2 GHG — market-based”?</td><td>442,000</td><td>401,000</td><td>493,000</td><td>447,000</td><td>580,000</td><td>526,000</td></tr><tr><td>Scope 3 GHG — business travel</td><td>205,000</td><td>186,000</td><td>97,000</td><td>88,000</td><td>101,000</td><td>92,000</td></tr><tr><td>Scope 3 GHG — use of sold products (Commercial Airplanes)**</td><td>400,000,000</td><td>363,000,000</td><td>306,000,000</td><td>278,000,000</td><td>246,000,000</td><td>223,000,000</td></tr><tr><td>Scope 3 GHG — use of sold products (Defense, Space &amp; Security)**</td><td>24,000,000</td><td>22,000,000</td><td>24,000,000</td><td>22,000,000</td><td>22,000,000</td><td>20,000,000</td></tr><tr><td>Total calculated GHG excluding sold products</td><td>1,355,000</td><td>1,229,000</td><td>1,264,000</td><td>1,147,000</td><td>1,292,000</td><td>1,172,000</td></tr><tr><td>Core metrics sites GHG — location-based*</td><td>724,000</td><td>657,000</td><td>702,000</td><td>637,000</td><td>713,000</td><td>647,000</td></tr><tr><td>Core metrics sites GHG — market-based* GHG Intensity®</td><td>323,000</td><td>293,000 $0.00002</td><td colspan="2">376,000 341,000 $0.00002</td><td colspan="2">452,000 410,000 $0.00002</td></tr></tbody></table>
==================================================

==================================================
Element Type: ListItem
Element Content: 1. Emissions (Enterprise Scope 1, Scope 2, and Scope 3 Categories 6 and 11) data is verified by an accredited independent third party to the level of limited assurance, see assurance statements. 2. Scope 1 and Scope 2 data represents 100% of the company. 3. For Scopes 1, 2 and 3, we calculate emissions from CO2, CH4, N2O, HFCs, PFCs, SF6 and NF3 for this data set. 4. Core metrics sites data represents emissions of CO2, CH4 and N2O where we track a subset of emissions from natural gas combustion and purchased electricity associated with sites that represent the majority (70%) of Boeing operations. 5. GHG intensity includes Scope 1 and Scope 2 (market-based) GHG (CO2, CH4, N2O, HFCs, PFCs, SF6 and NF3). 6. Use of sold products emissions are based on estimated lifetime emissions of Boeing Commercial Airplanes and Boeing Defense Services product deliveries in 2022, including direct emissions from combustion of fuel (335M tonnes) and indirect emissions

Metadata:
  coordinates: CoordinatesMetadata(points=((567.9913888888889, 1510.5997222222222), (567.9913888888889, 1647.2719444444444), (2701.9025000000024, 1647.2719444444444), (2701.9025000000024, 1510.5997222222222)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c45e0>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  parent_id: ab5a06ab3f755f7b4dcdce9f07f5f7f9
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: NarrativeText
Element Content: from production of fuel (50M tonnes).

Metadata:
  detection_class_prob: 0.5250699520111084
  coordinates: CoordinatesMetadata(points=((550.9948120117188, 1605.1129150390625), (550.9948120117188, 1681.2330322265625), (2651.37744140625, 1681.2330322265625), (2651.37744140625, 1605.1129150390625)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f649e7c42e0>)
  last_modified: 2024-11-28T10:49:00
  _known_field_names: frozenset({'is_continuation', 'detection_origin', 'text_as_html', 'attached_to_filename', 'languages', 'data_source', 'header_footer_type', 'table_as_cells', 'signature', 'link_texts', 'bcc_recipient', 'cc_recipient', 'file_directory', 'email_message_id', 'filename', 'image_path', 'link_urls', 'subject', 'emphasized_text_tags', 'sent_to', 'link_start_indexes', 'sent_from', 'regex_metadata', 'last_modified', 'detection_class_prob', 'page_number', 'image_base64', 'category_depth', 'coordinates', 'emphasized_text_contents', 'orig_elements', 'image_mime_type', 'url', 'key_value_pairs', 'page_name', 'links', 'filetype', 'parent_id'})
  filetype: application/pdf
  languages: ['eng']
  page_number: 1
  parent_id: ab5a06ab3f755f7b4dcdce9f07f5f7f9
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

用OCR拿到的结果会有所不同


==================================================
Element Type: Title
Element Content: 2023 Boeing ATU Ellateleyiam ats) lela 9

Metadata:
  coordinates: CoordinatesMetadata(points=((101.0, 99.0), (101.0, 161.0), (362.0, 161.0), (362.0, 99.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: (Ofoyn aN)

Metadata:
  coordinates: CoordinatesMetadata(points=((97.0, 279.0), (97.0, 312.0), (206.0, 312.0), (206.0, 279.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: Introduction

Metadata:
  coordinates: CoordinatesMetadata(points=((102.0, 377.0), (102.0, 396.0), (240.0, 396.0), (240.0, 377.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: Approach & Governance

Metadata:
  coordinates: CoordinatesMetadata(points=((100.0, 471.0), (100.0, 495.0), (384.0, 495.0), (384.0, 471.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: People

Metadata:
  coordinates: CoordinatesMetadata(points=((102.0, 565.0), (102.0, 589.0), (181.0, 589.0), (181.0, 565.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: Products & Services

Metadata:
  coordinates: CoordinatesMetadata(points=((102.0, 659.0), (102.0, 678.0), (337.0, 678.0), (337.0, 659.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: Operations

Metadata:
  coordinates: CoordinatesMetadata(points=((101.0, 753.0), (101.0, 777.0), (229.0, 777.0), (229.0, 753.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: Communities

Metadata:
  coordinates: CoordinatesMetadata(points=((101.0, 847.0), (101.0, 866.0), (255.0, 866.0), (255.0, 847.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: Reporting

Metadata:
  coordinates: CoordinatesMetadata(points=((102.0, 941.0), (102.0, 965.0), (223.0, 965.0), (223.0, 941.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: Key ESG Data GRI Index SASB Index TCFD Index

Metadata:
  coordinates: CoordinatesMetadata(points=((100.0, 991.0), (100.0, 1154.0), (254.0, 1154.0), (254.0, 991.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: U.N. Sustainable Development Goals

Metadata:
  coordinates: CoordinatesMetadata(points=((102.0, 1185.0), (102.0, 1240.0), (313.0, 1240.0), (313.0, 1185.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: Awards and Recognition

Metadata:
  coordinates: CoordinatesMetadata(points=((100.0, 1266.0), (100.0, 1288.0), (364.0, 1288.0), (364.0, 1266.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: Select Memberships and Partnerships

Metadata:
  coordinates: CoordinatesMetadata(points=((101.0, 1313.0), (101.0, 1368.0), (322.0, 1368.0), (322.0, 1313.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: Forward-Looking Statements

Metadata:
  coordinates: CoordinatesMetadata(points=((101.0, 1395.0), (101.0, 1444.0), (284.0, 1444.0), (284.0, 1395.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: Key ESG Data

Metadata:
  coordinates: CoordinatesMetadata(points=((571.0, 220.0), (571.0, 297.0), (1079.0, 297.0), (1079.0, 220.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: Environmental Data

Metadata:
  coordinates: CoordinatesMetadata(points=((576.0, 314.0), (576.0, 330.0), (782.0, 330.0), (782.0, 314.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Text
Element Content: 2022 2021 2020 Energy' Megawatt hours Terajoules Megawatt hours Terajoules Megawatt hours Terajoules Natural gas 1,928,000 6,941 1,712,000 6,163 1,686,000 6,070 Jet kerosene 861,000 3,100 804,000 2,894 544,000 1,958 Fuel oil #2 127,000 457 153,000 551 149,000 536 Motor gasoline 24,000 86 21,000 76 21,000 76 Propane 11,000 40 10,000 36 12,000 43 Liquefied petroleum gas 2,000 7 1,000 4 - - Total nonrenewable fuels 2,953,000 10,631 2,701,000 9,724 2,412,000 8,683 Sustainable aviation fuel 9,000 32 4,000 14 2,000 7 Total renewable fuels 9,000 32 4,000 14 2,000 7 Purchased nonrenewable electricity 1,350,000 4,860 1,482,000 5,335 1,686,000 6,070 Purchased renewable electricity” 720,000 2,592 574,000 2,066 392,000 1,411 Total purchased electricity 2,070,000 7,452 2,056,000 7,402 2,078,000 7,481 Total energy use 5,033,000 18,119 4,761,000 17,410 4,492,000 16,171

Metadata:
  coordinates: CoordinatesMetadata(points=((582.0, 355.0), (582.0, 954.0), (2571.0, 954.0), (2571.0, 355.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  parent_id: b1148db182a46f545d50e17a2ed4cd2c
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: ListItem
Element Content: 1. Data represents 100% of the company.

Metadata:
  coordinates: CoordinatesMetadata(points=((568.0, 975.0), (568.0, 993.0), (913.0, 993.0), (913.0, 975.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  parent_id: b1148db182a46f545d50e17a2ed4cd2c
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: ListItem
Element Content: 2. Renewable electricity data excludes any renewable energy that is part of the grid by default, in alignment with SASB and other frameworks. Notably, Boeing operates in a number of grids that rely significantly on renewable sources.

Metadata:
  coordinates: CoordinatesMetadata(points=((567.0, 998.0), (567.0, 1016.0), (2495.0, 1016.0), (2495.0, 998.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  parent_id: b1148db182a46f545d50e17a2ed4cd2c
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: ListItem
Element Content: Boeing did not sell any electricity, heating or cooling energy.

Metadata:
  coordinates: CoordinatesMetadata(points=((573.0, 1017.0), (573.0, 1035.0), (1081.0, 1035.0), (1081.0, 1017.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  parent_id: b1148db182a46f545d50e17a2ed4cd2c
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Text
Element Content: Emissions’ Tons CO,e Metric tons CO,e Tons CO,e Metric tons CO,e Tons CO,e Metric tons CO,e Scope 1 GHG?* 708,000 642,000 675,000 612,000 611,000 554,000 Scope 2 GHG — location-based?* 859,000 779,000 830,000 753,000 840,000 762,000 Scope 2 GHG — market-based?* 442,000 401,000 493,000 447,000 580,000 526,000 Scope 3 GHG — business travel 205,000 186,000 97,000 88,000 101,000 92,000 Scope 3 GHG — use of sold products (Commercial Airplanes)** 400,000,000 363,000,000 306,000,000 278,000,000 246,000,000 223,000,000 Scope 3 GHG — use of sold products (Defense, Space & Security)** 24,000,000 22,000,000 24,000,000 22,000,000 22,000,000 20,000,000 Total calculated GHG excluding sold products 1,355,000 1,229,000 1,264,000 1,147,000 1,292,000 1,172,000 Core metrics sites GHG — location-based* 724,000 657,000 702,000 637,000 713,000 647,000 Core metrics sites GHG — market-based* 323,000 293,000 376,000 341,000 452,000 410,000 GHG Intensity* $0.00002 $0.00002 $0.00002

Metadata:
  coordinates: CoordinatesMetadata(points=((583.0, 1065.0), (583.0, 1497.0), (2639.0, 1497.0), (2639.0, 1065.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  parent_id: b1148db182a46f545d50e17a2ed4cd2c
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: ListItem
Element Content: 1. Emissions (Enterprise Scope 1, Scope 2, and Scope 3 Categories 6 and 11) data is verified by an accredited independent third party to the level of limited assurance, see assurance statements.

Metadata:
  coordinates: CoordinatesMetadata(points=((570.0, 1511.0), (570.0, 1529.0), (2198.0, 1529.0), (2198.0, 1511.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  parent_id: b1148db182a46f545d50e17a2ed4cd2c
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: ListItem
Element Content: 2. Scope 1 and Scope 2 data represents 100% of the company.

Metadata:
  coordinates: CoordinatesMetadata(points=((569.0, 1535.0), (569.0, 1553.0), (1102.0, 1553.0), (1102.0, 1535.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  parent_id: b1148db182a46f545d50e17a2ed4cd2c
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: ListItem
Element Content: 3. For Scopes 1, 2 and 3, we calculate emissions from CO,, CH,, N,O, HFCs, PFCs, SF, and NF, for this data set. 4. Core metrics sites data represents emissions of CO,, CH, and N,O where we track a subset of emissions from natural gas combustion and purchased electricity associated with sites that represent the majority (70%) of Boeing operations 5. GHG intensity includes Scope 1 and Scope 2 (market-based) GHG (CO,, CH,, NO, HFCs, PFCs, SF, and NF.) 6. Use of sold products emissions are based on estimated lifetime emissions of Boeing Commercial Airplanes and Boeing Defense Services product deliveries in 2022, including direct emissions from combustion of fuel (835M tonnes) and indirect emissions

Metadata:
  coordinates: CoordinatesMetadata(points=((569.0, 1558.0), (569.0, 1647.0), (2696.0, 1647.0), (2696.0, 1558.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  parent_id: b1148db182a46f545d50e17a2ed4cd2c
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================

==================================================
Element Type: Title
Element Content: from production of fuel (50M tonnes).

Metadata:
  coordinates: CoordinatesMetadata(points=((590.0, 1648.0), (590.0, 1666.0), (897.0, 1666.0), (897.0, 1648.0)), system=<unstructured.documents.coordinates.PixelSpace object at 0x7f8c1e547070>)
  filetype: application/pdf
  languages: ['eng']
  last_modified: 2024-11-28T10:49:00
  page_number: 1
  _known_field_names: frozenset({'regex_metadata', 'is_continuation', 'header_footer_type', 'image_mime_type', 'image_base64', 'orig_elements', 'filename', 'page_name', 'signature', 'text_as_html', 'attached_to_filename', 'data_source', 'subject', 'coordinates', 'filetype', 'email_message_id', 'links', 'cc_recipient', 'url', 'link_texts', 'table_as_cells', 'link_start_indexes', 'bcc_recipient', 'link_urls', 'page_number', 'emphasized_text_contents', 'last_modified', 'sent_from', 'detection_origin', 'sent_to', 'key_value_pairs', 'file_directory', 'category_depth', 'image_path', 'languages', 'parent_id', 'emphasized_text_tags', 'detection_class_prob'})
  file_directory: data/Boeing-Sustainability-Report/1_page
  filename: Boeing-Sustainability-Report_074.pdf
==================================================



'''