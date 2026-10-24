import request from './request'

// 管理端检索调试：对一篇已建索引的论文跑一次单轮检索，返回各路召回、融合、rerank 和最终上下文的逐 chunk 名次。

export interface RetrievalDebugRequest {
  arxiv_id: string
  question: string
  top_k?: number | null
  enable_query_rewrite?: boolean
  enable_hyde?: boolean
  enable_keyword_search?: boolean
  enable_table_structured_route?: boolean
  enable_llm_rerank?: boolean
  enable_context_expansion?: boolean
}

export interface RrfVote {
  rank: number | null
  weight: number | null
  multiplier: number | null
  vote: number | null
}

export interface MatchedIndex {
  type: string
  text: string
  score: number | null
  routes: string[]
}

export interface DebugChunkRow {
  key: string
  chunk_id: number | string | null
  chunk_type: string
  section: string
  page: number | string | null
  preview: string
  content: string
  route_ranks: Record<string, number>
  route_scores: Record<string, number | null>
  keyword_terms: string[]
  fused_rank: number | null
  fused_score: number | null
  rrf_votes: Record<string, RrfVote>
  matched_indexes: MatchedIndex[]
  source_queries: string[]
  rerank_rank: number | null
  rerank_score: number | null
  final_rank: number | null
  context_role: string
  final_reason: string
}

export interface RouteSummary {
  name: string
  status: string
  enabled: boolean
  latency_ms: number | null
  candidate_count: number
  weight: number | null
  error: string
  fallback_reason: string
}

export interface QueryCandidate {
  query: string
  source: string
  selected: boolean
  reason: string
}

export interface RetrievalDebugRun {
  arxiv_id: string
  title: string
  active_build_id: string
  question: string
  elapsed_ms: number
  query: {
    original: string
    // 原问向量路由实际检索用的英文译文；status 为 error 时 query 退回原问。
    original_translation: {
      query?: string
      status?: string
      error?: string
      latency_ms?: number
    }
    intent: {
      main_intent: string | null
      sub_intents: string[] | null
      confidence: number | null
      intent_summary: string | null
      source: string | null
      fallback_reason: string | null
    }
    rewrite_enabled: boolean
    rewrite_status: string
    rewrite_latency_ms: number | null
    rewrite_error: string
    selected_queries: string[]
    candidates: QueryCandidate[]
    keywords: string[]
    hyde_text: string
    rerank_query: string
  }
  config: {
    effective_top_k: number | null
    recall_candidate_limit: number | null
    rerank_candidate_limit: number | null
    context_expansion_anchor_limit: number | null
    enable_query_rewrite: boolean | null
    enable_hyde: boolean | null
    enable_keyword_search: boolean | null
    enable_table_structured_route: boolean | null
    enable_llm_rerank: boolean | null
    enable_context_expansion: boolean | null
  }
  fusion: { rrf_k: number | null; route_weights: Record<string, number> }
  routes: RouteSummary[]
  rerank: {
    enabled: boolean | null
    applied: boolean | null
    mode: string | null
    reason: string | null
    input_chunks: number | null
    output_chunks: number | null
    latency_ms: number | null
  }
  // 锚点数计算过程；页面暂不展示，随导出 JSON 一起带出供排查。
  anchor_budget?: AnchorBudget | null
  chunks: DebugChunkRow[]
}

export interface AnchorBudget {
  policy: string
  ratio: number
  chunk_count: number
  chunk_count_source: string
  paper_share: number
  raw: number | null
  min: number
  max: number
  anchor_limit: number
  reranked_count: number
  reject_tail_ratio: number
  reject_after_rank: number
  blocked: Array<{
    chunk_id: string
    section: string
    chunk_type: string
    rerank_rank: number
    relation: string
    from_anchor_rank: number
  }>
}

export function runRetrievalDebug(payload: RetrievalDebugRequest) {
  return request.post('/admin/retrieval-debug/runs', payload) as Promise<RetrievalDebugRun>
}
