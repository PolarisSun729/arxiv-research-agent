import pandas as pd
import asyncio
from pathlib import Path
import sys
import os

# 添加当前目录（backend-241123）到系统路径
current_dir = Path(__file__).parent.parent  # 回到 backend-241123 目录
sys.path.append(str(current_dir))

from services.search_service import SearchService

async def process_disclosures():
    # 使用相对于脚本的路径来找到 CSV 文件
    csv_path = Path(__file__).parent / "Disclosure_Requirements_Recommendations_Guidance.csv"
    df = pd.read_csv(csv_path)
    
    # 合并每行的所有列内容
    df['combined_text'] = df.apply(lambda row: ' '.join(str(val) for val in row if pd.notna(val)), axis=1)
    
    # 初始化SearchService
    search_service = SearchService()
    
    # 存储结果的列表
    all_results = []
    
    # 对每行进行搜索
    for index, row in df.iterrows():
        query = row['combined_text']
        results = await search_service.search(
            query=query,
            collection_id="Boeing_openai_20241127014109",
            top_k=10,
            threshold=0.0  # 设置为0以获取所有结果
        )
        
        # 提取页码和文本
        pages = [str(result['metadata']['page']) for result in results]
        texts = [result['text'] for result in results]
        
        # 添加到结果列表
        result_row = {
            'Disclosure': row['Disclosure'],
            'Pages': ','.join(pages),
        }
        # 添加文本字段
        for i, text in enumerate(texts, 1):
            result_row[f'Text_{i}'] = text
        
        all_results.append(result_row)
    
    # 创建结果DataFrame并保存
    result_df = pd.DataFrame(all_results)
    output_path = Path(__file__).parent / "disclosure_search_results.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(process_disclosures()) 