import fitz
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from unstructured.partition.pdf import partition_pdf
from typing import List, Dict, Any
import numpy as np

class PDFVisualizer:
    def __init__(self):
        self.category_to_color = {
            "Title": "orchid",
            "Text": "deepskyblue",
            "NarrativeText": "lightgreen",
            "Header": "gold",
            "Footer": "lightcoral",
            "ListItem": "cornflowerblue",
            "Image": "forestgreen",
            "Table": "tomato",
            "FigureCaption": "purple",
            "PageBreak": "gray"
        }
    
    def get_element_category(self, element: Any) -> str:
        """获取元素类型名称"""
        return type(element).__name__
    
    def get_element_coordinates(self, element: Any) -> Dict:
        """获取元素坐标信息"""
        if hasattr(element, 'coordinates'):
            return {
                'points': element.coordinates.points,
                'layout_width': element.coordinates.layout_width,
                'layout_height': element.coordinates.layout_height
            }
        return None

    def plot_pdf_with_boxes(self, pdf_page, elements: List[Any]) -> None:
        """在PDF页面上绘制边界框"""
        # 获取页面图像
        pix = pdf_page.get_pixmap()
        pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # 创建图形
        fig, ax = plt.subplots(1, figsize=(15, 20))
        ax.imshow(pil_image)
        
        categories = set()
        
        # 绘制每个元素的边界框
        for element in elements:
            coords = self.get_element_coordinates(element)
            if coords:
                points = coords['points']
                layout_width = coords['layout_width']
                layout_height = coords['layout_height']
                
                # 缩放坐标点
                scaled_points = [
                    (x * pix.width / layout_width, y * pix.height / layout_height)
                    for x, y in points
                ]
                
                category = self.get_element_category(element)
                categories.add(category)
                box_color = self.category_to_color.get(category, "deepskyblue")
                
                # 创建并添加多边形
                rect = patches.Polygon(
                    scaled_points,
                    linewidth=1,
                    edgecolor=box_color,
                    facecolor="none",
                    alpha=0.7
                )
                ax.add_patch(rect)
                
                # 添加文本标签
                centroid = np.mean(scaled_points, axis=0)
                ax.text(centroid[0], centroid[1], category, 
                       fontsize=8, color=box_color,
                       ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # 创建图例
        legend_handles = []
        for category in categories:
            if category in self.category_to_color:
                legend_handles.append(
                    patches.Patch(
                        color=self.category_to_color[category],
                        label=category
                    )
                )
        
        ax.axis("off")
        ax.legend(handles=legend_handles, loc="upper right")
        plt.tight_layout()
        plt.show()

    def render_page(self, file_path: str, page_number: int, print_text: bool = True) -> None:
        """
        渲染PDF页面并显示元素边界框
        
        Args:
            file_path: PDF文件路径
            page_number: 页码（从1开始）
            print_text: 是否打印文本内容
        """
        try:
            # 打开PDF页面
            pdf = fitz.open(file_path)
            pdf_page = pdf.load_page(page_number - 1)
            
            # 获取页面元素
            elements = partition_pdf(
                filename=file_path,
                include_page_breaks=True,
                include_metadata=True,
                include_coords=True,
                start_page=page_number,
                end_page=page_number
            )
            
            # 绘制可视化
            self.plot_pdf_with_boxes(pdf_page, elements)
            
            # 打印文本内容
            if print_text:
                print("\n页面文本内容:")
                print("=" * 50)
                for element in elements:
                    if hasattr(element, 'text'):
                        category = self.get_element_category(element)
                        print(f"\n[{category}]:")
                        print("-" * 40)
                        print(element.text)
            
            pdf.close()
            
        except Exception as e:
            print(f"处理PDF页面时出错: {str(e)}")
            raise

if __name__ == "__main__":
    file_path = "山西-en.pdf"
    visualizer = PDFVisualizer()
    
    try:
        # 渲染第一页
        visualizer.render_page(file_path, page_number=1)
        
    except FileNotFoundError:
        print(f"找不到PDF文件: {file_path}")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")