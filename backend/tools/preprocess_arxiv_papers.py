import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from services.embedding.embedding_service import EmbeddingService
from services.storage.sqlite import StorageContainer
from services.storage.vector_store_service import VectorStoreService

PAPER_EMBEDDING_COLLECTION = "arxiv_paper_embeddings"

DATA_FILE_PATH = r"D:\极客时间大模型RAG进阶实战营\rag-project01-framework\07-local-arxiv\arxiv-2026-04-papers.json"
BATCH_SIZE = 10

def load_arxiv_papers(file_path: str) -> list:
    papers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    paper = json.loads(line)
                    papers.append(paper)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
    return papers

def get_published_date(paper: dict) -> str:
    versions = paper.get('versions', [])
    if versions:
        first_version = versions[0]
        created_str = first_version.get('created', '')
        try:
            dt = datetime.strptime(created_str, '%a, %d %b %Y %H:%M:%S GMT')
            return dt.strftime('%Y-%m-%d')
        except:
            pass
    return paper.get('update_date', '')

def main():
    logger.info("Starting arXiv papers preprocessing...")
    
    storage = StorageContainer()
    paper_catalog_store = storage.paper_catalog
    embedding_service = EmbeddingService()
    vector_store_service = VectorStoreService()
    
    logger.info(f"Loading papers from {DATA_FILE_PATH}")
    papers = load_arxiv_papers(DATA_FILE_PATH)
    logger.info(f"Loaded {len(papers)} papers")
    
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for i, paper in enumerate(papers, 1):
        arxiv_id = paper.get('id', '')
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        authors = paper.get('authors', '')
        categories = paper.get('categories', '')
        update_date = paper.get('update_date', '')
        
        if not arxiv_id or not title:
            skipped_count += 1
            continue
        
        existing_paper = paper_catalog_store.get_paper(arxiv_id)
        if existing_paper:
            logger.debug(f"Skipping existing paper: {arxiv_id}")
            skipped_count += 1
            continue
        
        try:
            text_to_embed = f"{title}\n\n摘要：{abstract}"
            embedding_config = embedding_service.get_default_embedding_config()
            embedding = embedding_service.create_single_embedding(
                text_to_embed,
                provider=embedding_config.provider,
                model=embedding_config.model_name,
                api_key=embedding_config.api_key,
                base_url=embedding_config.base_url,
                dimension=embedding_config.dimension,
            )
            
            published_date = get_published_date(paper)
            url = f"https://arxiv.org/abs/{arxiv_id}"
            
            metadata = {
                "content": abstract,
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "categories": categories,
                "published_date": published_date,
                "url": url,
                "embedding_model": embedding_config.model_name
            }
            
            embedding_id = vector_store_service.insert_single_embedding(
                PAPER_EMBEDDING_COLLECTION, 
                embedding, 
                metadata
            )
            
            success = paper_catalog_store.add_paper({
                'arxiv_id': arxiv_id,
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'categories': categories,
                'published_date': published_date,
                'url': url,
                'embedding_id': str(embedding_id),
                'embedding_model': embedding_config.model_name
            })
            
            if success:
                processed_count += 1
                logger.debug(f"Processed {i}/{len(papers)}: {arxiv_id} - {title[:50]}...")
            else:
                failed_count += 1
                
        except Exception as e:
            logger.error(f"Error processing paper {arxiv_id}: {str(e)}")
            failed_count += 1
        
        if i % BATCH_SIZE == 0:
            logger.debug(f"Progress: {i}/{len(papers)} papers processed")
    
    logger.info(f"Preprocessing complete!")
    logger.info(f"Processed: {processed_count}")
    logger.info(f"Skipped (existing): {skipped_count}")
    logger.info(f"Failed: {failed_count}")

if __name__ == "__main__":
    main()
