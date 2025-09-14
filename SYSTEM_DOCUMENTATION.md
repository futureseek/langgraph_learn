# LangGraph多智能体学习项目

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3.6+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> 基于LangGraph框架构建的多智能体（Multi-Agent）对话系统，集成RAG智能问答、文档处理、搜索引擎等功能。

## 🎯 项目概述

本系统是一个基于 LangGraph 框架构建的多智能体对话系统，集成了检索增强生成(RAG)、文档处理、搜索引擎等多种功能。系统采用模块化设计，通过三个专门化的智能体协同工作，实现了复杂的任务规划、执行和评估流程。

### ✨ 核心特性

- 🤖 **多智能体协作**：任务规划、执行、评估三大智能体协同工作
- 🧠 **RAG智能问答**：基于向量数据库的文档检索增强生成
- 🛠️ **丰富工具集**：文档处理、搜索引擎、路径管理等实用工具
- 🔄 **状态管理**：完整的对话状态和上下文维护机制
- 📱 **易于扩展**：模块化设计，支持自定义工具和智能体

### 🏗️ 系统架构

```
用户输入 → 任务规划 → 任务执行 → 工具调用 → 结果评估 → 智能回复
    ↓         ↓         ↓         ↓         ↓         ↓
  需求分析   制定计划   执行步骤   调用工具   质量评估   用户反馈
```

#### 三大智能体

- **🎯 TaskPlanner**：任务拆解专家，负责分析需求和制定执行计划
- **⚡ TaskExecutor**：任务执行专家，负责调用工具完成具体操作
- **🔍 TaskEvaluator**：结果评估专家，负责评估质量和提供改进建议

## 🚀 快速开始

### 环境要求

- Python 3.12+
- uv 包管理器
- 4GB+ 内存
- 2GB+ 存储空间

### 安装步骤

1. **克隆项目**
```shell
git clone <项目地址>
cd langgraph_learn
```

2. **检查uv版本**
```shell
uv --version
```

3. **创建虚拟环境**
```shell
uv venv
```

4. **安装依赖**
```shell
uv sync
```

5. **运行项目**
```shell
uv run langgraph_chat.py
```

### 首次使用

启动后您将看到：
```
🤖 多智能体协作系统已启动！
📋 系统包含三个专门化智能体：
   🎯 TaskPlanner - 任务拆解专家
   ⚡ TaskExecutor - 任务执行专家
   🔍 TaskEvaluator - 结果评估专家

🛠️ 可用工具：
   🔍 搜索工具 - 网络信息检索
   📄 文档工具 - 文件读取和导出
   📁 路径工具 - 文件路径获取
   🧠 RAG工具 - 智能文档问答系统
```

## 🧠 RAG智能问答系统

### 核心功能

RAG（检索增强生成）系统是本项目的重要特性，支持基于文档的智能问答。

#### 主要组件

- **📚 VectorDatabase**：基于ChromaDB的向量数据库
- **📝 DocumentProcessor**：支持20+种文档格式的处理器
- **🔍 RAGRetriever**：智能检索和问答生成器

#### 支持的文件格式

- **文本文件**：.txt, .md, .log, .cfg, .ini, .yaml, .yml
- **代码文件**：.py, .js, .java, .cpp, .c, .h, .hpp
- **文档文件**：.pdf, .docx
- **数据文件**：.json, .xml, .csv
- **网页文件**：.html

### RAG工具使用

#### 1. 添加文档到知识库
```
用户: add_document_to_rag ./README.md
```

#### 2. 批量添加目录文档
```
用户: add_directory_to_rag ./docs/
```

#### 3. 智能问答
```
用户: rag_question_answer 这个项目如何安装？
```

#### 4. 查看知识库状态
```
用户: get_rag_stats
```

#### 5. 管理知识库
```
用户: delete_rag_document ./old_doc.md
用户: clear_rag_knowledge_base
```

## 🛠️ 工具系统

### 文档工具

#### DocumentReader（文档读取）
- **功能**：读取各种格式的文本文件
- **特性**：智能编码检测、批量处理、通配符支持
- **使用**：`document_reader("*.py")` 或 `document_reader("./src/")`

#### DocumentExporter（文档导出）
- **功能**：将内容导出为多种格式
- **支持**：TXT, MD, JSON, CSV, HTML
- **使用**：`export_document(content, "output.md", "markdown")`

### 搜索工具

#### TavilySearcher（网络搜索）
- **功能**：集成Tavily API进行网络信息检索
- **场景**：实时信息查询、市场研究、技术资料搜索
- **使用**：`tavily_search_results_json("AI技术发展")`

### 路径工具

#### Path_Acquire（路径管理）
- **功能**：获取和管理文件系统路径
- **特性**：当前目录获取、路径解析、目录结构分析

## 📋 使用示例

### 基础对话
```
用户: 请读取README.md文件
系统: [TaskPlanner] 分析任务 → [TaskExecutor] 调用document_reader → [TaskEvaluator] 评估结果
```

### RAG问答流程
```
用户: 将项目文档添加到知识库
系统: [TaskPlanner] 识别add_directory_to_rag → [TaskExecutor] 处理文档 → [TaskEvaluator] 确认完成

用户: 根据知识库回答：这个系统的架构是什么？
系统: [TaskPlanner] 识别rag_question_answer → [TaskExecutor] 检索相关文档 → [RAG] 生成回答 → [TaskEvaluator] 评估质量
```

### 复合任务
```
用户: 搜索AI新闻并生成报告
系统: [TaskPlanner] 制定计划 → [TaskExecutor] 调用搜索工具 → 整理内容 → 调用导出工具 → [TaskEvaluator] 验证结果
```

## 🔧 配置说明

### API配置

在 `langgraph_chat.py` 中配置模型API：
```python
model = ChatOpenAI(
    model='Qwen/Qwen3-1.7B',
    base_url='https://api-inference.modelscope.cn/v1',
    api_key='your-api-key-here'
)
```

### RAG配置

默认RAG配置参数：
```python
# 文档处理配置
chunk_size = 1000          # 文档块大小
chunk_overlap = 200        # 块重叠大小

# 检索配置
retrieval_k = 4            # 检索文档数量
score_threshold = 0.3      # 相似度阈值
max_context_length = 4000  # 最大上下文长度
```

## 📂 项目结构

```
langgraph_learn/
├── agents/                    # 智能体模块
│   ├── BaseAgent.py          # 基础智能体类
│   ├── MessageManager.py     # 消息管理器
│   ├── MultiAgentState.py    # 状态定义
│   └── __init__.py
├── tools/                     # 工具模块
│   ├── DocumentReader.py     # 文档读取
│   ├── document_exporter.py  # 文档导出
│   ├── TavilySearcher.py     # 搜索工具
│   ├── Path_Acquire.py       # 路径工具
│   ├── VectorDatabase.py     # 向量数据库
│   ├── DocumentProcessor.py  # 文档处理
│   ├── RAGRetriever.py       # RAG检索器
│   └── __init__.py
├── doc/                       # 文档目录
│   ├── RAG_INTEGRATION_REPORT.md    # RAG集成报告
│   ├── langgraph_project_documentation.md  # 详细技术文档
│   ├── rag_usage_guide.md     # RAG使用指南
│   ├── test_document.md       # 测试文档
│   └── 中华人民共和国证券法(2019修订).pdf  # 测试PDF文档
├── rag_data/                  # RAG数据目录
│   └── vector_db/            # 向量数据库存储
├── langgraph_chat.py         # 主程序入口
├── test_rag.py               # RAG功能测试脚本
├── pyproject.toml            # 项目配置
├── README.md                 # 项目说明
└── TODO.md                   # 功能扩展计划
```

## 🚧 扩展开发

### 添加新工具

1. **创建工具文件**
```python
# tools/NewTool.py
from langchain.tools import Tool

def new_tool_function(input_text: str) -> str:
    # 工具逻辑实现
    return result

def create_new_tool() -> Tool:
    return Tool(
        name="new_tool",
        description="工具描述",
        func=new_tool_function
    )
```

2. **注册工具**
```python
# tools/__init__.py
from .NewTool import create_new_tool
__all__.append("create_new_tool")
```

3. **集成到主程序**
```python
# langgraph_chat.py
from tools.NewTool import create_new_tool
new_tool = create_new_tool()
tools = [existing_tools..., new_tool]
```

### 自定义智能体

```python
class CustomAgent(BaseAgent):
    def __init__(self, model, tools):
        super().__init__(model, tools, "CustomAgent", "自定义智能体")
    
    def process(self, state):
        # 自定义处理逻辑
        return result
```

## 📊 技术栈
### 🛠️ 技术栈

- **核心框架**：LangGraph
- **语言模型**：Qwen/Qwen3-1.7B（可替换为其他模型）
- **向量数据库**：ChromaDB
- **文档处理**：Meta-Chunking 文本分块技术

### 核心依赖
- **LangGraph** 0.3.6+ - 多智能体工作流框架
- **LangChain** 0.3.27+ - AI应用开发框架
- **ChromaDB** 1.0.20+ - 向量数据库

### 完整依赖列表
```toml
dependencies = [
    "langchain-community>=0.3.27",
    "langchain-mcp-adapters>=0.1.9",
    "langchain-openai>=0.3.29",
    "langchain-tavily>=0.2.11",
    "langgraph-cli[inmem]>=0.3.6",
    "openai>=1.99.6",
    "chromadb>=0.4.22",
    "sentence-transformers>=2.2.2",
    "langchain-chroma>=0.1.2",
    "pypdf>=4.0.1",
    "python-docx>=1.1.0",
]
```

## 🧪 测试功能

### RAG功能测试

项目包含完整的RAG功能测试脚本 `test_rag.py`，支持自动化测试：

```bash
# 运行RAG功能测试
uv run test_rag.py
```

测试内容包括：
- ✅ 知识库初始状态检查
- ✅ 文档添加功能测试
- ✅ 文档检索功能测试
- ✅ 智能问答功能测试
- ✅ 知识库状态统计

### 测试文档

`doc/` 目录包含丰富的测试文档：
- **test_document.md**：基础功能测试文档
- **中华人民共和国证券法(2019修订).pdf**：PDF文档处理测试
- **rag_usage_guide.md**：RAG使用指南
- **RAG_INTEGRATION_REPORT.md**：RAG集成技术报告

## 🐛 故障排除

### 常见问题

1. **模型API调用失败**
   - 检查API密钥配置
   - 验证模型名称正确性（推荐：Qwen/Qwen3-1.7B）
   - 确认网络连接正常

2. **RAG功能异常**
   - 确保文档格式受支持
   - 检查向量数据库权限
   - 验证ChromaDB依赖安装
   - 运行 `test_rag.py` 诊断问题

3. **依赖安装失败**
   - 检查Python版本（需要3.12+）
   - 确保uv工具正常工作
   - 尝试清理缓存重新安装：`uv cache clean`

4. **文档处理问题**
   - 确保文件路径正确
   - 检查文件权限和编码
   - 验证文件格式支持

### 调试技巧

- 使用 `get_rag_stats` 查看知识库状态
- 运行测试脚本排查问题：`uv run test_rag.py`
- 检查工具执行日志输出
- 单独测试工具功能
- 查看 `rag_data/vector_db/` 目录权限

- 要开关 uv 虚拟环境，你可以使用以下命令：
- 激活虚拟环境
 在 PowerShell 中激活：`. .venv\Scripts\Activate.ps1`
在 CMD 中激活：`.venv\Scripts\activate.bat`
- 退出虚拟环境
无论你使用的是 PowerShell 还是 CMD，都可以通过以下命令退出虚拟环境：`deactivate`

## 🤝 贡献指南

1. Fork 本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) - 多智能体工作流框架
- [LangChain](https://github.com/langchain-ai/langchain) - AI应用开发框架
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量数据库
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) - 文本嵌入模型

---

## 📚 更多资源

- 📖 **[详细技术文档](./doc/langgraph_project_documentation.md)** - 完整的技术实现和架构说明
- 🛠️ **[RAG使用指南](./doc/rag_usage_guide.md)** - RAG功能详细使用教程
- 📊 **[RAG集成报告](./doc/RAG_INTEGRATION_REPORT.md)** - RAG功能集成技术报告
- 📋 **[功能扩展计划](./TODO.md)** - 未来功能开发路线图
- 🧪 **[测试脚本](./test_rag.py)** - 自动化功能测试

**💡 快速上手提示**：首次使用建议先运行 `uv run test_rag.py` 验证系统功能完整性。