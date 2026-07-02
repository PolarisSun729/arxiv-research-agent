export type ArxivQueryCapabilityStatus = 'info' | 'warning' | 'error'

export interface BackendArxivQueryCapability {
  source?: string | null
  mode?: string | null
  supported_subset?: string[]
  precision_policy?: string | null
  full_arxiv_syntax_supported?: boolean | null
  unsupported_reason?: string | null
  suggested_action?: string | null
  fts5_available?: boolean | null
  search_index_status?: string | null
  [key: string]: unknown
}

export interface NormalizedArxivQueryCapability {
  status: ArxivQueryCapabilityStatus
  source: string
  sourceLabel: string
  mode: string
  modeLabel: string
  summary: string
  detailLines: string[]
  supportedFields: string[]
  supportedOperators: string[]
  unsupportedSyntax: string[]
  warnings: string[]
  unsupportedReason?: string | null
  suggestedAction?: string | null
  fullArxivSyntaxSupported: boolean
  raw?: BackendArxivQueryCapability | Record<string, unknown> | null
  errorCode?: string | null
  errorMessage?: string | null
}
