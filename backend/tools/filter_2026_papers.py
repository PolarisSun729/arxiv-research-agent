import json
import os

def filter_2026_papers(input_path: str, output_path: str):
    """
    从 arXiv 数据集中筛选出 2026 年的论文
    
    Args:
        input_path (str): 输入的 JSON 数据集路径
        output_path (str): 输出的 JSON 文件路径
    """
    print(f"Loading dataset from: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: File not found - {input_path}")
        return
    
    papers_2026 = []
    total_papers = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    paper = json.loads(line)
                    total_papers += 1
                    
                    update_date = paper.get('update_date', '')
                    if update_date.startswith('2026'):
                        papers_2026.append(paper)
                    
                    if total_papers % 10000 == 0:
                        print(f"Processed {total_papers} papers...")
                        
                except json.JSONDecodeError:
                    continue
    
    print(f"\nTotal papers processed: {total_papers}")
    print(f"2026 papers found: {len(papers_2026)}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for paper in papers_2026:
            f.write(json.dumps(paper, ensure_ascii=False) + '\n')
    
    print(f"\n2026 papers saved to: {output_path}")

if __name__ == "__main__":
    input_file = r"D:\极客时间大模型RAG进阶实战营\rag-project01-framework\07-local-arxiv\arxiv-metadata-oai-snapshot.json"
    output_file = r"D:\极客时间大模型RAG进阶实战营\rag-project01-framework\07-local-arxiv\arxiv-2026-papers.json"
    
    filter_2026_papers(input_file, output_file)