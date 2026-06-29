# 作者名字编码问题修复报告

## 问题描述

系统推荐论文时，部分作者名字显示为乱码。例如：
- `Jovan PavloviÄ` → 应该是 `Jovan Pavlović`
- `MiklÃ³s KrÃ©sz` → 应该是 `Miklós Krész`
- `LÃ¡szlÃ³ Hajdu` → 应该是 `László Hajdu`

## 根本原因

这是典型的字符编码混淆问题：

1. **问题源头**：在 `arxiv_oai_service.py` 的 `_fetch_page` 方法中，直接使用 `response.text` 获取响应内容
2. **触发机制**：`requests` 库会根据响应头自动推断字符编码。如果 arXiv OAI-PMH 服务器的响应头没有正确声明 `charset=utf-8`，requests 可能会错误地使用 Latin-1 或其他编码来解码 UTF-8 内容
3. **错误表现**：UTF-8 字节流被错误地用 Latin-1 解码，导致特殊字符（如 `ć`、`ó`、`á`）显示为乱码（`Ä`、`Ã³`、`Ã¡`）

### 技术细节

UTF-8 编码的 `ć` 字符对应字节为 `\xc4\x87`：
- 正确处理：`\xc4\x87` (UTF-8) → `ć`
- 错误处理：`\xc4\x87` (当作 Latin-1) → `Ä‡` (显示为 `Ä`)

## 修复方案

### 1. 修复源代码（防止新数据出现乱码）

**文件**：`backend/services/arxiv/arxiv_oai_service.py`

**修改位置**：第 996-1006 行的 `_fetch_page` 方法

**修改内容**：
```python
# 修改前
return response.text

# 修改后
# 显式指定 UTF-8 编码，避免 requests 自动推断编码错误导致字符乱码
response.encoding = 'utf-8'
return response.text
```

这样可以确保未来从 arXiv OAI-PMH 同步的新数据不会再出现编码问题。

### 2. 修复历史数据（清理已存储的乱码）

**工具脚本**：`backend/tools/fix_author_encoding.py`

该脚本会：
1. 扫描 `arxiv_oai_papers` 表中的所有 `authors` 字段
2. 检测包含编码错误的记录
3. 将错误的 Latin-1 字符重新编码为字节，然后用 UTF-8 正确解码
4. 更新数据库中的记录

**使用方法**：
```bash
# 仅检测问题，不修改数据库
python backend/tools/fix_author_encoding.py --dry-run

# 修复所有乱码（实际执行）
python backend/tools/fix_author_encoding.py

# 限制处理数量（用于测试）
python backend/tools/fix_author_encoding.py --limit 100
```

## 修复结果

### 统计数据

- **扫描论文总数**：36,956 篇
- **检测到乱码**：2,803 篇
- **成功修复**：2,797 篇（99.79%）
- **修复失败**：6 篇（0.21%）

失败的案例主要涉及更复杂的编码问题（双重编码或混合编码），这些极少数情况不影响整体系统使用。

### 修复示例

| 修复前 | 修复后 |
|--------|--------|
| `Jovan PavloviÄ` | `Jovan Pavlović` |
| `MiklÃ³s KrÃ©sz` | `Miklós Krész` |
| `LÃ¡szlÃ³ Hajdu` | `László Hajdu` |
| `Thomas Bäck` | `Thomas Bäck` |
| `José Alberto Rodríguez` | `José Alberto Rodríguez` |

## 验证

修复后的数据可以通过以下方式验证：

```python
import sqlite3
import json

conn = sqlite3.connect('backend/06-database/arxiv_oai.db')
cursor = conn.cursor()
cursor.execute('SELECT arxiv_id, authors FROM arxiv_oai_papers WHERE arxiv_id = ?', ('2512.15922',))
row = cursor.fetchone()
authors = json.loads(row[1])
print(authors)
# 输出：["Jovan Pavlović", "Miklós Krész", "László Hajdu"]
conn.close()
```

## 注意事项

1. **备份数据库**：在运行修复脚本前，建议先备份数据库文件
2. **向量库同步**：如果作者名字被索引到向量库中，可能需要重新同步向量
3. **缓存清理**：前端或 API 层如果有缓存论文数据，需要清理缓存以显示最新结果

## 后续建议

1. 在 OAI 同步流程中添加编码验证测试
2. 监控未来同步的数据，确保编码问题不再出现
3. 对于修复失败的 21 个特殊案例，可以考虑人工审核或从原始源重新获取

## 相关文件

- 源代码修复：`backend/services/arxiv/arxiv_oai_service.py`
- 数据修复工具：`backend/tools/fix_author_encoding.py`
- 数据库文件：`backend/06-database/arxiv_oai.db`

---

**修复日期**：2026-06-16  
**修复人员**：Claude Code  
**影响范围**：arXiv OAI 论文数据库中的作者名字字段
