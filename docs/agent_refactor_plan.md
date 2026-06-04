# Agent refactor plan

## Stage 0 scope

Stage 0 only addresses baseline stability and observability for the arXiv Agent search path.

- fix recoverable failure handling in search and personalized rerank nodes
- keep degraded flows usable instead of escalating them into global runtime errors
- clarify the boundary between warnings, errors, debug, and steps
- remove obviously duplicated comments that reduce readability
- add minimal node-level tests for personalized rerank fallback behavior

## Stage 0 non-goals

Stage 0 does **not** introduce or restructure the larger agent architecture.

- no Planner is introduced
- no LangGraph main flow restructuring is performed
- no dynamic tool selection framework is added
- no tool registry redesign is performed
- no memory system schema or storage redesign is performed
- no frontend-backend end-to-end workflow rewrite is performed

## Testing strategy in Stage 0

Tests stay focused on node behavior and local downgrade semantics.

- use mocks or monkeypatch-style overrides for recommendation service failures
- do not depend on real arXiv network requests
- do not depend on real recommendation services
- do not depend on real database state
- prefer narrow tests that verify papers are preserved, steps are recorded, and warnings remain user-friendly

## Stage 1 scope

Stage 1 introduces a lightweight planning layer after intent parsing and before the existing intent-specific execution branches.

- add `Goal` and `ExecutionPlanStep` as structured planning data models
- add `goal` and `execution_plan` to `AgentState` and response payloads
- add a dedicated `plan_task` node after `parse_search_request`
- keep planning deterministic and template-based for now; LLM use is optional and not required in Stage 1
- expose planning state in sync responses and stream state snapshots for debugging and frontend display

## Stage 1 flow shape

The high-level graph shape in Stage 1 is:

- `START -> parse_search_request -> plan_task -> original intent routing`

This means every request gets a structured planning pass before entering the existing search, paper reading, preference, recommendation, or fallback branches.

## Stage 1 non-goals

Stage 1 intentionally does **not** introduce a full planner-executor architecture.

- no dynamic tool selection
- no runtime replan / reflect / observe loop
- no tool registry redesign
- no tool execution takeover by `execution_plan`
- no memory manager redesign
- no `memory_read` / `memory_write` refactor
- no frontend workflow rewrite tied to planner semantics

## Stage 1 testing strategy

Tests remain local, deterministic, and independent from external systems.

- verify `plan_task` generates `goal` and `execution_plan` for supported intents
- verify `unclear` and `unsupported` still produce safe non-tool execution plans
- verify pending confirmation routes still work after inserting `plan_task`
- verify graph compilation and Mermaid export include `plan_task`
- verify sync response and compact stream state include planning fields
- do not depend on real arXiv requests
- do not depend on real recommendation services
- do not depend on real LLM responses

## Stage 2 scope

Stage 2 introduces an internal tool-call protocol layer between planning-oriented state and existing execution nodes.

- add `ToolCallRequest` as the internal representation of an intended tool call
- add `ToolObservation` as the internal representation of a tool execution outcome
- add a generic `execute_tool` node that reuses the existing `tool_registry`
- keep legacy `tool_name` / `tool_args` / `tool_result` / `tool_calls` fields for compatibility during migration
- migrate the `arxiv_search` search path first as the pilot flow for the new internal tool layer
- preserve downstream compatibility for `check_search_result`, personalized rerank, sync responses, and stream snapshots

## Stage 2 boundaries and non-goals

Stage 2 is intentionally **MCP-like in shape but not standard MCP**.

- this stage only defines an internal tool layer for the agent runtime
- this stage does **not** implement standard MCP
- this stage does **not** introduce an MCP server
- this stage does **not** introduce an MCP client
- this stage does **not** introduce external protocol-based tool communication
- this stage does **not** allow the LLM to freely choose arbitrary tools at runtime
- this stage does **not** add dynamic tool discovery or external tool attachment
- this stage does **not** add Observe / Reflect / Replan loops
- this stage does **not** redesign `tool_registry`; it reuses the existing registry and validation flow
- this stage does **not** refactor `memory_read` / `memory_write`
- this stage does **not** restructure the memory subsystem or storage contracts

## Stage 2 testing strategy

Stage 2 tests should stay local, deterministic, and fully isolated from external services.

- verify `ToolCallRequest` can be constructed, dumped, and validated
- verify `execute_tool` returns `skipped` when `tool_call_request` is missing
- verify `execute_tool` produces failed observations for unknown tool names
- verify `execute_tool` produces failed observations for invalid tool arguments
- verify `execute_tool` produces success observations for successful tool execution
- verify the `arxiv_search` flow writes `tool_call_request` before execution
- verify the `arxiv_search` flow produces `tool_observations` after execution
- verify papers extracted from `tool_result` still drive `check_search_result` and downstream ranking logic
- verify tool failures stay as structured observations/warnings instead of escalating into a global runtime error
- use mocks or monkeypatch-style overrides for tool execution
- do not depend on real arXiv network requests
- do not depend on real recommendation services
- do not depend on real database state
