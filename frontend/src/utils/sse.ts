export interface SseBlock {
  event: string | null
  dataText: string
}

// 只处理 SSE 协议层：拆出 event 名和拼接后的 data 文本；JSON 解析与容错策略由各调用方决定。
export function parseSseBlock(raw: string): SseBlock {
  let event: string | null = null
  const dataLines: string[] = []

  for (const line of raw.split(/\r?\n/)) {
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim()
    } else if (line.startsWith('data:')) {
      // 按 SSE 规范只去掉一个前导空格，保留 data 内容本身的空白。
      dataLines.push(line.slice('data:'.length).replace(/^ /, ''))
    }
  }

  return { event, dataText: dataLines.join('\n') }
}

// 逐个产出以空行分隔的原始事件块；流结束时连同 decoder 残留字节一起产出尾块。
// 消费方提前 return/throw 时生成器的 finally 会释放 reader 锁。
export async function* readSseBlocks(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      const parts = buffer.split(/\r?\n\r?\n/)
      buffer = parts.pop() || ''
      yield* parts
    }

    const tail = `${buffer}${decoder.decode()}`
    if (tail.trim()) yield tail
  } finally {
    reader.releaseLock()
  }
}
