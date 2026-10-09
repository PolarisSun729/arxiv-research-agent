<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { fetchAgentGraph } from '@/api/agent'
import type { AgentGraphResponse } from '@/types/agent'

interface GraphNode {
  key: string
  label: string
  subtitle: string
  x: number
  y: number
  width: number
  height: number
  tone: string
}

interface GraphEdge {
  from: string
  to: string
  label?: string
}

// 这页只负责展示图结构，不参与 agent 运行逻辑，所以用固定布局会更直观也更稳定。
const fallbackGraph: AgentGraphResponse = {
  graph_name: 'arxiv_search_agent',
  render_source: 'fallback',
  node_names: [
    'parse_search_request',
    'run_agent_turn'
  ],
  mermaid: `graph TD;
    START([START]) --> parse_search_request;
    parse_search_request --> run_agent_turn;
    run_agent_turn --> END([END]);`,
  supports_png: false
}

const graph = ref<AgentGraphResponse>(fallbackGraph)
const loading = ref(false)
const errorMessage = ref('')

const nodeMeta: Record<string, { label: string; subtitle: string; tone: string }> = {
  parse_search_request: { label: 'Parse', subtitle: 'Intent and context', tone: 'tone-parse' },
  run_agent_turn: { label: 'Run Turn', subtitle: 'Plan, observe, interrupt or answer', tone: 'tone-final' }
}

const nodes = computed<GraphNode[]>(() => {
  const width = 220
  const height = 68
  return [
    { key: 'parse_search_request', x: 70, y: 150, width, height, ...nodeMeta.parse_search_request },
    { key: 'run_agent_turn', x: 420, y: 150, width, height, ...nodeMeta.run_agent_turn }
  ]
})

// 这里直接按照后端图的边来画，避免额外引入 Mermaid 运行时。
const edges: GraphEdge[] = [
  { from: 'START', to: 'parse_search_request' },
  { from: 'parse_search_request', to: 'run_agent_turn', label: 'parsed state' },
  { from: 'run_agent_turn', to: 'END' }
]

const nodeByKey = computed(() => {
  return nodes.value.reduce<Record<string, GraphNode>>((acc, node) => {
    acc[node.key] = node
    return acc
  }, {})
})

const diagramWidth = 720
const diagramHeight = 360

const nodeCount = computed(() => graph.value.node_names.length)
const mermaidSource = computed(() => graph.value.mermaid || fallbackGraph.mermaid)

function getAnchorPoint(node: GraphNode, position: 'left' | 'right' | 'top' | 'bottom') {
  if (position === 'left') return { x: node.x, y: node.y + node.height / 2 }
  if (position === 'right') return { x: node.x + node.width, y: node.y + node.height / 2 }
  if (position === 'top') return { x: node.x + node.width / 2, y: node.y }
  return { x: node.x + node.width / 2, y: node.y + node.height }
}

function edgePoints(edge: GraphEdge) {
  const from = edge.from === 'START' ? null : nodeByKey.value[edge.from]
  const to = edge.to === 'END' ? null : nodeByKey.value[edge.to]
  if (!from || !to) return null

  const fromPoint = getAnchorPoint(from, from.x <= to.x ? 'right' : 'bottom')
  const toPoint = getAnchorPoint(to, from.x <= to.x ? 'left' : 'top')
  return {
    fromPoint,
    toPoint,
    midPoint: {
      x: (fromPoint.x + toPoint.x) / 2,
      y: (fromPoint.y + toPoint.y) / 2
    }
  }
}

const svgEdges = computed(() => {
  return edges
    .map(edge => {
      if (edge.from === 'START') {
        const target = nodeByKey.value[edge.to]
        if (!target) return null
        const y = target.y + target.height / 2
        return { key: `${edge.from}-${edge.to}`, x1: 16, y1: y, x2: target.x, y2: y, label: null }
      }

      if (edge.to === 'END') {
        const source = nodeByKey.value[edge.from]
        if (!source) return null
        const y = source.y + source.height / 2
        return { key: `${edge.from}-${edge.to}`, x1: source.x + source.width, y1: y, x2: diagramWidth - 18, y2: y, label: null }
      }

      const points = edgePoints(edge)
      if (!points) return null

      return {
        key: `${edge.from}-${edge.to}`,
        x1: points.fromPoint.x,
        y1: points.fromPoint.y,
        x2: points.toPoint.x,
        y2: points.toPoint.y,
        label: edge.label || null,
        labelX: points.midPoint.x,
        labelY: points.midPoint.y
      }
    })
    .filter(Boolean) as Array<{
    key: string
    x1: number
    y1: number
    x2: number
    y2: number
    label: string | null
    labelX?: number
    labelY?: number
  }>
})

async function loadGraph() {
  loading.value = true
  errorMessage.value = ''
  try {
    graph.value = await fetchAgentGraph()
  } catch (error) {
    graph.value = fallbackGraph
    errorMessage.value = error instanceof Error ? error.message : 'Failed to load graph preview'
    ElMessage.warning('Backend graph endpoint is unavailable, showing local preview.')
  } finally {
    loading.value = false
  }
}

async function copyMermaid() {
  try {
    await navigator.clipboard.writeText(mermaidSource.value)
    ElMessage.success('Mermaid copied')
  } catch {
    ElMessage.error('Copy failed, please select the text manually.')
  }
}

onMounted(() => {
  loadGraph()
})
</script>

<template>
  <div class="agent-graph-page">
    <section class="hero-card">
      <div class="hero-copy">
        <p class="eyebrow">Agent Graph</p>
        <h1 class="title">arXiv Agent flow preview</h1>
        <p class="subtitle">
          This page shows the LangGraph structure at a glance. The left side is a compact flowchart, and the right
          side shows the Mermaid source returned by the backend.
        </p>
      </div>

      <div class="hero-actions">
        <el-button :loading="loading" @click="loadGraph">Refresh</el-button>
        <el-button type="primary" @click="copyMermaid">Copy Mermaid</el-button>
      </div>
    </section>

    <section class="summary-row">
      <div class="summary-card">
        <span class="summary-label">Graph</span>
        <strong class="summary-value">{{ graph.graph_name }}</strong>
      </div>
      <div class="summary-card">
        <span class="summary-label">Source</span>
        <strong class="summary-value">{{ graph.render_source }}</strong>
      </div>
      <div class="summary-card">
        <span class="summary-label">Nodes</span>
        <strong class="summary-value">{{ nodeCount }}</strong>
      </div>
      <div class="summary-card">
        <span class="summary-label">PNG support</span>
        <strong class="summary-value">{{ graph.supports_png ? 'Yes' : 'No' }}</strong>
      </div>
    </section>

    <el-alert
      v-if="errorMessage"
      class="graph-alert"
      :title="errorMessage"
      type="warning"
      show-icon
      :closable="false"
    />

    <section class="graph-grid">
      <div class="panel graph-panel">
        <div class="panel-head">
          <div>
            <h2>Flowchart</h2>
            <p>Conditional branches are annotated on the connecting lines.</p>
          </div>
          <el-tag effect="plain" type="success">Live Preview</el-tag>
        </div>

        <div class="graph-canvas">
          <svg :viewBox="`0 0 ${diagramWidth} ${diagramHeight}`" class="graph-svg" role="img" aria-label="Agent flow diagram">
            <defs>
              <marker id="arrow-head" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto">
                <path d="M0,0 L8,4 L0,8 Z" fill="currentColor" />
              </marker>
            </defs>

            <g class="edge-layer">
              <line
                v-for="edge in svgEdges"
                :key="edge.key"
                :x1="edge.x1"
                :y1="edge.y1"
                :x2="edge.x2"
                :y2="edge.y2"
                class="graph-edge"
                marker-end="url(#arrow-head)"
              />
              <text
                v-for="edge in svgEdges.filter(item => item.label)"
                :key="`${edge.key}-label`"
                :x="edge.labelX ?? 0"
                :y="edge.labelY ?? 0"
                class="edge-label"
                text-anchor="middle"
              >
                {{ edge.label }}
              </text>
            </g>

            <g class="node-layer">
              <g v-for="node in nodes" :key="node.key" :transform="`translate(${node.x}, ${node.y})`">
                <rect class="graph-node" :class="node.tone" :width="node.width" :height="node.height" rx="18" ry="18" />
                <text class="node-title" x="18" y="27">{{ node.label }}</text>
                <text class="node-subtitle" x="18" y="49">{{ node.subtitle }}</text>
              </g>
            </g>
          </svg>
        </div>
      </div>

      <div class="panel detail-panel">
        <div class="panel-head">
          <div>
            <h2>Mermaid source</h2>
            <p>You can paste this into Mermaid Live Editor or documentation tools.</p>
          </div>
        </div>

        <pre class="mermaid-source">{{ mermaidSource }}</pre>

        <div class="node-list">
          <div class="node-list__title">Node list</div>
          <div class="node-chip" v-for="name in graph.node_names" :key="name">{{ name }}</div>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.agent-graph-page {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.hero-card {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding: 22px 24px;
  border-radius: 22px;
  background:
    radial-gradient(circle at top right, rgba(52, 211, 153, 0.16), transparent 30%),
    linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.94));
  color: #fff;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
}

.hero-copy {
  min-width: 0;
}

.eyebrow {
  margin: 0 0 8px;
  color: #86efac;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.title {
  margin: 0;
  font-size: 30px;
  line-height: 1.15;
}

.subtitle {
  margin: 10px 0 0;
  max-width: 820px;
  color: rgba(226, 232, 240, 0.88);
}

.hero-actions {
  display: flex;
  gap: 10px;
  flex: none;
}

.summary-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}

.summary-card {
  padding: 14px 16px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.86);
  border: 1px solid rgba(148, 163, 184, 0.16);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}

.summary-label {
  display: block;
  color: #64748b;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.summary-value {
  display: block;
  margin-top: 6px;
  color: #0f172a;
  font-size: 15px;
}

.graph-alert {
  border-radius: 14px;
}

.graph-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(360px, 0.9fr);
  gap: 16px;
  align-items: start;
}

.panel {
  border-radius: 22px;
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(148, 163, 184, 0.16);
  box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
  overflow: hidden;
}

.panel-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding: 18px 20px 0;
}

.panel-head h2 {
  margin: 0;
  font-size: 18px;
  color: #0f172a;
}

.panel-head p {
  margin: 6px 0 0;
  color: #64748b;
  line-height: 1.55;
}

.graph-panel {
  min-width: 0;
}

.graph-canvas {
  padding: 14px 16px 18px;
  overflow: auto;
}

.graph-svg {
  width: 100%;
  min-width: 1200px;
  height: auto;
  display: block;
}

.edge-layer {
  color: rgba(37, 99, 235, 0.45);
}

.graph-edge {
  stroke: currentColor;
  stroke-width: 2.2;
  fill: none;
  marker-end: url(#arrow-head);
}

.edge-label {
  fill: #475569;
  font-size: 12px;
  font-weight: 700;
  paint-order: stroke;
  stroke: rgba(255, 255, 255, 0.92);
  stroke-width: 4px;
  stroke-linejoin: round;
}

.graph-node {
  stroke: rgba(15, 23, 42, 0.08);
  stroke-width: 1.2;
  filter: drop-shadow(0 12px 18px rgba(15, 23, 42, 0.12));
}

.tone-parse {
  fill: #dbeafe;
}

.tone-build {
  fill: #cffafe;
}

.tone-tool {
  fill: #fef3c7;
}

.tone-check {
  fill: #dcfce7;
}

.tone-retry {
  fill: #fee2e2;
}

.tone-rank {
  fill: #ede9fe;
}

.tone-pref {
  fill: #ffe4e6;
}

.tone-confirm {
  fill: #e2e8f0;
}

.tone-handle {
  fill: #e0f2fe;
}

.tone-read {
  fill: #d1fae5;
}

.tone-final {
  fill: #f3e8ff;
}

.node-title {
  fill: #0f172a;
  font-size: 16px;
  font-weight: 700;
}

.node-subtitle {
  fill: #475569;
  font-size: 12px;
}

.detail-panel {
  padding-bottom: 18px;
}

.mermaid-source {
  margin: 16px 20px 0;
  padding: 16px;
  max-height: 360px;
  overflow: auto;
  border-radius: 18px;
  background: #0f172a;
  color: #dbeafe;
  font-size: 12px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}

.node-list {
  padding: 16px 20px 0;
}

.node-list__title {
  margin-bottom: 10px;
  color: #0f172a;
  font-size: 14px;
  font-weight: 700;
}

.node-chip {
  display: inline-flex;
  align-items: center;
  margin: 0 10px 10px 0;
  padding: 8px 10px;
  border-radius: 999px;
  background: #eff6ff;
  color: #1d4ed8;
  font-size: 12px;
  font-weight: 600;
}

@media (max-width: 1180px) {
  .summary-row {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .graph-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 820px) {
  .hero-card {
    flex-direction: column;
    align-items: stretch;
  }

  .summary-row {
    grid-template-columns: 1fr;
  }

  .hero-actions {
    flex-wrap: wrap;
  }
}
</style>
