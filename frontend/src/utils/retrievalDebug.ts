import type { DebugChunkRow, RetrievalDebugRequest, RetrievalDebugRun } from '@/api/adminRetrievalDebug'

// 检索调试页的派生计算：每个 chunk 走到了哪一步、各路召回的独有贡献与重叠。只读接口数据，不重新打分。

export const ROUTE_LABELS: Record<string, string> = {
  vector_original: '向量·原问',
  vector_rewrite: '向量·改写',
  vector_hyde: '向量·HyDE',
  keyword: '关键词 BM25',
  table_structured: '表格结构化',
  memory_context: '对话记忆'
}

export function routeLabel(name: string) {
  return ROUTE_LABELS[name] || name
}

export type ChunkOutcomeCode = 'final' | 'expansion' | 'fusion_cut' | 'rerank_dropped' | 'not_selected' | 'not_fused'

export interface ChunkOutcome {
  code: ChunkOutcomeCode
  label: string
}

// 「检索输出」是检索阶段交给下游的候选池（锚点 + 扩展块），最终进 prompt 的部分还要经过预算规划器裁剪。
export const OUTCOME_LABELS: Record<ChunkOutcomeCode, string> = {
  final: '进入检索输出',
  expansion: '扩展补入',
  fusion_cut: '融合截断',
  rerank_dropped: 'rerank 后落选',
  not_selected: '未入选输出',
  not_fused: '未参与融合'
}

// 融合阶段实际保留的条数：与后端 fused_limit 同义，取 rrf_candidate_limit 和 top_k 的较大者。
export function fusedKeepLimit(run: RetrievalDebugRun) {
  return Math.max(run.config.rrf_candidate_limit ?? 0, run.config.effective_top_k ?? 0)
}

// 一个 chunk 最终停在哪一步；没进检索输出的，按最早把它挡掉的阶段归因。
export function chunkOutcome(row: DebugChunkRow, run: RetrievalDebugRun): ChunkOutcome {
  const keepLimit = fusedKeepLimit(run)
  const fusionKept = row.fused_rank !== null && row.fused_rank <= keepLimit
  let code: ChunkOutcomeCode
  // 融合没保留（或根本没召回）却出现在输出里的，是上下文扩展按锚点补进来的图表、相邻段落。
  if (row.final_rank !== null) code = fusionKept ? 'final' : 'expansion'
  else if (row.fused_rank === null) code = 'not_fused'
  else if (!fusionKept) code = 'fusion_cut'
  else if (run.rerank.applied && row.rerank_rank !== null) code = 'rerank_dropped'
  else code = 'not_selected'
  return { code, label: OUTCOME_LABELS[code] }
}

// 每一路投票占这个 chunk 融合总分的比例，看名次主要由哪一路撑起来。
export function voteShares(row: DebugChunkRow) {
  const total = Object.values(row.rrf_votes).reduce((sum, vote) => sum + (vote.vote ?? 0), 0)
  return Object.entries(row.rrf_votes)
    .map(([route, vote]) => ({ route, ...vote, share: total > 0 ? (vote.vote ?? 0) / total : 0 }))
    .sort((a, b) => b.share - a.share)
}

export interface RouteStats {
  name: string
  // 只有这一路召回到的 chunk 数：接近 0 说明这一路只是在给别的路已有的结果重复投票。
  unique: number
  // 经融合进入检索输出的 chunk 里，这一路召回到的个数（扩展补入的不算，它们不是靠召回进来的）。
  finalHits: number
  // 上述 chunk 的融合票里，这一路所占的比例。
  finalVoteShare: number
}

export function routeStats(run: RetrievalDebugRun, routeNames: string[]): RouteStats[] {
  const finalRows = run.chunks.filter(row => chunkOutcome(row, run).code === 'final')
  const finalVoteTotal = finalRows.reduce((sum, row) => sum + Object.values(row.rrf_votes).reduce((acc, vote) => acc + (vote.vote ?? 0), 0), 0)
  return routeNames.map(name => {
    const unique = run.chunks.filter(row => {
      const routes = Object.keys(row.route_ranks)
      return routes.length === 1 && routes[0] === name
    }).length
    const finalHits = finalRows.filter(row => name in row.route_ranks).length
    const routeVotes = finalRows.reduce((sum, row) => sum + (row.rrf_votes[name]?.vote ?? 0), 0)
    return { name, unique, finalHits, finalVoteShare: finalVoteTotal > 0 ? routeVotes / finalVoteTotal : 0 }
  })
}

// 两两路由召回集合的 Jaccard 重叠度；对角线为 1，某路为空时与它相关的格子为 null。
export function routeOverlap(run: RetrievalDebugRun, routeNames: string[]) {
  const sets = new Map(routeNames.map(name => [name, new Set(run.chunks.filter(row => name in row.route_ranks).map(row => row.key))]))
  return routeNames.map(a => routeNames.map(b => {
    const setA = sets.get(a)!
    const setB = sets.get(b)!
    if (!setA.size || !setB.size) return null
    let intersection = 0
    setA.forEach(key => { if (setB.has(key)) intersection += 1 })
    return intersection / (setA.size + setB.size - intersection)
  }))
}

// 导出一次运行的完整结果，供离线排查：带上当次的请求开关，每个 chunk 附上页面判定的去向，
// 不随页面筛选变化，导出的始终是全部 chunk。
export function buildDebugExport(run: RetrievalDebugRun, request: RetrievalDebugRequest, exportedAt = new Date()) {
  return {
    exported_at: exportedAt.toISOString(),
    request,
    run: {
      ...run,
      chunks: run.chunks.map(row => {
        const outcome = chunkOutcome(row, run)
        return { ...row, outcome: outcome.code, outcome_label: outcome.label }
      })
    }
  }
}

// 旧式 arXiv 编号带斜杠（如 hep-th/9901001），不能直接放进文件名。
export function debugExportFilename(arxivId: string, exportedAt = new Date()) {
  const stamp = exportedAt.toISOString().slice(0, 19).replace(/[-:]/g, '').replace('T', '-')
  return `retrieval-debug-${arxivId.replace(/[^\w.-]+/g, '_')}-${stamp}.json`
}
