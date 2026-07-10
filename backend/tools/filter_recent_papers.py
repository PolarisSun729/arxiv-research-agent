import json
import os
from datetime import datetime
from pathlib import Path

def filter_papers_after_date(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    min_date: str = "2026-04-01",
):
    """
    从 arXiv 数据集中筛选出指定日期之后且分类包含 cs.AI 的论文
    
    Args:
        input_path (str | os.PathLike[str]): 输入的 JSON 数据集路径
        output_path (str | os.PathLike[str]): 输出的 JSON 文件路径
        min_date (str): 最小日期，格式为 YYYY-MM-DD，默认为 2026-04-01
    """
    print(f"Loading dataset from: {input_path}")
    print(f"Filtering papers after: {min_date}")
    print("Filtering papers with category: cs.AI")
    
    if not os.path.exists(input_path):
        print(f"Error: File not found - {input_path}")
        return
    
    min_date_obj = datetime.strptime(min_date, "%Y-%m-%d")
    filtered_papers = []
    total_papers = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    paper = json.loads(line)
                    total_papers += 1
                    
                    categories = paper.get('categories', '')
                    update_date = paper.get('update_date', '')
                    if update_date and 'cs.AI' in categories:
                        try:
                            date_obj = datetime.strptime(update_date, "%Y-%m-%d")
                            if date_obj >= min_date_obj:
                                filtered_papers.append(paper)
                        except ValueError:
                            pass
                    
                    if total_papers % 100000 == 0:
                        print(f"Processed {total_papers} papers...")
                        
                except json.JSONDecodeError:
                    continue
    
    print(f"\nTotal papers processed: {total_papers}")
    print(f"Papers after {min_date} found: {len(filtered_papers)}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for paper in filtered_papers:
            f.write(json.dumps(paper, ensure_ascii=False) + '\n')
    
    print(f"\nFiltered papers saved to: {output_path}")

if __name__ == "__main__":
    # 默认数据目录相对仓库根目录解析，确保脚本在不同开发机上行为一致。
    data_dir = Path(__file__).resolve().parents[2] / "07-local-arxiv"
    input_file = data_dir / "arxiv-2026-papers.json"
    output_file = data_dir / "arxiv-2026-04-papers.json"
    
    filter_papers_after_date(input_file, output_file, min_date="2026-04-01")
