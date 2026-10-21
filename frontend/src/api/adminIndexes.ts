import request from './request'

// 管理端论文索引查看：只读展示已建索引论文的切分与解析产物，后端仅允许管理员访问。

export interface PaperIndexSummary {
  arxiv_id: string
  title: string | null
  status: string
  chunk_count: number | null
  loading_method: string | null
  chunking_strategy: string | null
  active_build_id: string | null
  indexed_at: string | null
  updated_at: string | null
  build_count: number
}

export interface PaperIndexBuild {
  build_id: string
  status: string
  is_active: boolean
  artifact_status: string | null
  chunk_count: number | null
  loading_method: string | null
  chunking_strategy: string | null
  failed_stage: string | null
  indexed_at: string | null
  created_at: string | null
  has_chunk_file: boolean
}

export interface ChunkingParams {
  chunk_size: number
  max_chunk_content_length: number
  chunk_overlap_length: number
  split_search_window: number
}

export interface IndexedChunk {
  content: string
  char_count: number
  has_image: boolean
  metadata: Record<string, unknown> & {
    chunk_id: number
    chunk_type?: string
    page_start?: number
    page_end?: number
    word_count?: number
    section_path?: string
    section_part_index?: number
    section_part_count?: number
    subchunk_index?: number
    subchunk_count?: number
  }
}

export interface LengthBucket {
  min: number
  max: number | null
  count: number
}

export interface ChunkStats {
  total: number
  by_type: Record<string, number>
  text_count: number
  section_count: number
  text_chars: { min: number; p50: number; p90: number; max: number; mean: number }
  short_text_threshold: number
  short_text_count: number
  split_by_size_count: number
  split_overlong_count: number
  length_histogram: LengthBucket[]
}

export interface ChunkArtifactOverview {
  loading_method: string | null
  chunking_strategy: string | null
  total_pages: number | null
  saved_at: string | null
  chunking_params: ChunkingParams | null
  parsing: {
    docling_item_count: number | null
    docling_picture_count: number | null
    docling_table_count: number | null
    markdown_length: number | null
  }
  table_structure: Record<string, unknown> | null
  retrieval_index: { index_count?: number; index_type_counts?: Record<string, number> } | null
}

export type ChunkArtifact =
  | { available: false; reason: 'not_recorded' | 'missing' | 'invalid' }
  | { available: true; overview: ChunkArtifactOverview; stats: ChunkStats; chunks: IndexedChunk[] }

export interface PaperIndexDetail {
  arxiv_id: string
  status: string
  active_build_id: string | null
  selected_build_id: string
  builds: PaperIndexBuild[]
  artifact: ChunkArtifact
}

export interface ParsedDocument {
  pages: Array<{ page_number: number; text: string }>
  markdown: string
}

const BASE = '/admin/paper-indexes'

export function listPaperIndexes(params: { q?: string; status?: string; offset?: number; limit?: number }) {
  return request.get(BASE, { params }) as Promise<{ items: PaperIndexSummary[]; total: number }>
}

export function getPaperIndex(arxivId: string, buildId = '') {
  return request.get(`${BASE}/${encodeURIComponent(arxivId)}`, { params: buildId ? { build_id: buildId } : {} }) as Promise<PaperIndexDetail>
}

export function getParsedDocument(arxivId: string, buildId = '') {
  return request.get(`${BASE}/${encodeURIComponent(arxivId)}/parsed`, { params: buildId ? { build_id: buildId } : {} }) as Promise<ParsedDocument>
}

// 图片走 AuthenticatedImage 的受保护请求，这里只拼地址。
export function figureAssetUrl(arxivId: string, chunkId: number, buildId = '') {
  const query = buildId ? `?build_id=${encodeURIComponent(buildId)}` : ''
  return `/api${BASE}/${encodeURIComponent(arxivId)}/assets/${chunkId}${query}`
}
