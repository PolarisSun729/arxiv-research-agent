import request from './request'
import type { ArxivSearchRequest, ArxivSearchResponse } from '@/types/agent'

export async function runArxivSearchAgent(payload: ArxivSearchRequest): Promise<ArxivSearchResponse> {
  return request.post('/agent/arxiv-search', payload)
}
