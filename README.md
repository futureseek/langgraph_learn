# LangGraph多智能体学习项目

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3.6+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> 基于LangGraph框架构建的多智能体（Multi-Agent）对话系统，集成RAG智能问答、文档处理、搜索引擎等功能。
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
2. **配置ollama**
- 下载并安装 Ollama: 
访问 [Ollama 官网](https://ollama.com/) 下载并安装适用于你操作系统的 Ollama 软件。
- 配置模型: 
运行以下命令拉取所需模型：
```shell
ollama pull qwen3:1.7b
ollama pull embeddinggemma:300m
```
- 设置ollama并发：
编辑账号的环境变量--环境变量 打开环境变量设置，在用户环境变量中我们增加以上2个属性
```shell
OLLAMA_NUM_PARALLEL = 2
OLLAMA_MAX_LOADED_MODELS = 2
```
- 运行ollama服务：
```shell
ollama run qwen3:1.7b
ollama run embeddinggemma:300m
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
        • add_document_to_rag ./doc/中华人民共和国证券法(2019修订).pdf - 添加文档到知识库
        • add_directory_to_rag ./docs/ - 批量添加目录文档
        • rag_question_answer 您的问题 - 基于知识库问答
        • get_rag_stats - 查看知识库统计
        • delete_rag_document ./path/to/file.md - 删除指定文档
        • clear_rag_knowledge_base - 清空整个知识库
```
## 1. 系统概述

本系统是一个基于 LangGraph 框架构建的多智能体对话系统，集成了检索增强生成(RAG)、文档处理、搜索引擎等多种功能。系统采用模块化设计，通过三个专门化的智能体协同工作，实现了复杂的任务规划、执行和评估流程。

### 1.1 核心架构

系统采用三层智能体架构：
1. **TaskPlanner（任务规划器）**：负责接收用户请求，分析任务需求，制定执行计划
2. **TaskExecutor（任务执行器）**：根据计划执行具体任务，调用各种工具完成工作
3. **TaskEvaluator（任务评估器）**：评估任务执行结果，决定是否需要重新规划或调整策略

### 1.2 技术栈
- **核心框架**：LangGraph
- **语言模型**：Qwen/Qwen3-1.7B（可替换为其他模型）
- **向量数据库**：ChromaDB
- **文档处理**：Meta-Chunking 文本分块技术
- **依赖管理**：Python 3.12+

## 2. 功能模块详解

### 2.1 多智能体系统

#### 2.1.1 TaskPlanner（任务规划器）
负责分析用户请求，将复杂任务分解为可执行的子任务，并制定执行策略。规划器能够：
- 理解用户意图
- 识别所需工具和资源
- 生成任务执行计划
- 动态调整策略

#### 2.1.2 TaskExecutor（任务执行器）
负责执行具体的任务操作，包括：
- 调用RAG系统进行文档检索和问答
- 使用搜索引擎获取外部信息
- 处理文档和文本数据
- 执行文件操作和路径管理

#### 2.1.3 TaskEvaluator（任务评估器）
负责评估任务执行效果，包括：
- 检查任务完成度
- 评估结果质量
- 决定是否需要重新规划
- 提供改进建议

### 2.2 RAG（检索增强生成）系统

RAG系统是本系统的核心功能之一，提供了完整的文档管理和智能问答能力。

#### 2.2.1 核心功能
- **文档处理**：支持多种格式文档的解析和处理
- **向量存储**：使用ChromaDB存储文档向量表示
- **智能检索**：根据用户问题检索相关文档片段
- **问答生成**：基于检索结果生成准确答案

#### 2.2.2 技术实现
- 使用Meta-Chunking技术进行智能文本分块
- 集成多种文本分块策略（困惑度分块、概率差分分块、语义分块）
- 基于ChromaDB的向量数据库管理
- 支持增量式文档添加和更新

### 2.3 文档处理模块

#### 2.3.1 支持的文档格式
- Markdown (.md)
- PDF (.pdf)
- Word文档 (.docx)
- 纯文本 (.txt)
- Python (.py)
- JavaScript (.js)
- Java (.java)
- C/C++ (.c, .cpp, .h, .hpp)
- HTML (.html)
- JSON (.json)
- CSV (.csv)
- 配置文件 (.cfg, .ini, .yaml, .yml)
- 日志文件 (.log)

#### 2.3.2 Meta-Chunking技术
系统采用了先进的Meta-Chunking文本分块技术，包括：
- **Perplexity Chunking（困惑度分块）**：基于语言模型困惑度评估文本连贯性
- **Prob Subtraction Chunking（概率差分分块）**：通过计算句子间概率差分识别分块边界
- **Semantic Chunking（语义分块）**：基于语义相似度进行分块

### 2.4 工具系统

系统集成了丰富的工具集，支持各种常见任务：

#### 2.4.1 文档工具
- **DocumentReaderTool**：读取多种格式文档
- **DocumentExporterTool**：导出处理后的文档

#### 2.4.2 搜索工具
- **SearchTool**：网络搜索功能
- **PathTool**：文件路径管理

#### 2.4.3 其他工具
- **CalculatorTool**：数学计算
- **DateTimeTool**：日期时间处理

## 3. 系统工作流程

### 3.1 基本交互流程
1. 用户向系统提出问题或任务请求
2. TaskPlanner分析请求，制定执行计划
3. TaskExecutor根据计划调用相应工具执行任务
4. TaskEvaluator评估执行结果
5. 系统返回最终结果给用户

### 3.2 RAG问答流程
1. 用户提出问题
2. 系统检索向量数据库中相关文档片段
3. 结合检索结果和用户问题生成答案
4. 返回答案给用户

## 4. 系统特色功能

### 4.1 智能文本分块
系统采用Meta-Chunking技术，能够根据文本内容自动选择最优的分块策略，提高检索和问答的准确性。

### 4.2 多智能体协作
通过三个专门化智能体的协作，系统能够处理复杂的多步骤任务，提供更智能的服务。

### 4.3 可扩展架构
系统采用模块化设计，易于扩展新的功能模块和工具。

## 5. 使用方法

### 5.1 环境准备
1. 安装Python 3.12+
2. 安装依赖包：`pip install -r requirements.txt`
3. 下载语言模型（如Qwen/Qwen3-1.7B）

### 5.2 启动系统
运行主程序：
```bash
python langgraph_chat.py
```

### 5.3 使用示例
系统支持多种类型的查询，包括：
- 文档问答："请解释RAG系统的原理"
- 信息检索："查找关于Meta-Chunking的相关资料"
- 任务执行："帮我总结这个文档的主要内容"

### 5.4 RAG工具使用

#### 5.4.1 添加文档到知识库
```
用户: add_document_to_rag ./README.md
```

#### 5.4.2 批量添加目录文档
```
用户: add_directory_to_rag ./docs/
```

#### 5.4.3 智能问答
```
用户: rag_question_answer 这个项目如何安装？
```

#### 5.4.4 查看知识库状态
```
用户: get_rag_stats
```

#### 5.4.5 管理知识库
```
用户: delete_rag_document ./old_doc.md
用户: clear_rag_knowledge_base
```

## 6. 系统配置

### 6.1 API配置
在 `langgraph_chat.py` 中配置模型API：
```python
model = ChatOpenAI(
    model='Qwen/Qwen3-1.7B',
    base_url='https://api-inference.modelscope.cn/v1',
    api_key='your-api-key-here'
)
```

### 6.2 RAG配置
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

## 7. 项目结构

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
│   ├── text_normalizer.py    # 文本规范化工具
│   ├── clean_think.py        # 屏蔽<think>块内容
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
├── pyproject.toml            # 项目配置
├── README.md                 # 项目说明
└── TODO.md                   # 功能扩展计划
```

## 8. 扩展开发

### 8.1 添加新工具

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

### 8.2 自定义智能体

```
class CustomAgent(BaseAgent):
    def __init__(self, model, tools):
        super().__init__(model, tools, "CustomAgent", "自定义智能体")
    
    def process(self, state):
        # 自定义处理逻辑
        return result
```

## 9. 未来发展规划

根据TODO.md文件，系统未来计划扩展以下功能：
- 支持更多文档格式
- 增强多语言处理能力
- 优化智能体协作机制
- 扩展工具集功能
- 改进用户交互界面

## 10. RAG功能详解

### 10.1 什么是RAG？
RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合了信息检索和自然语言生成的AI技术。它通过以下步骤工作：
1. **文档索引**：将文档分块并转换为向量存储
2. **检索相关内容**：根据用户问题检索最相关的文档片段
3. **生成回答**：基于检索到的内容生成准确回答

### 10.2 RAG核心组件

#### 10.2.1 VectorDatabase.py
- 基于ChromaDB的向量数据库管理
- 支持文档向量化存储和检索
- 自动处理文档更新和重复检测
- 提供完整的CRUD操作接口

#### 10.2.2 DocumentProcessor.py
- 支持多种文档格式处理
- 智能文本分块处理
- 支持目录批量处理

#### 10.2.3 RAGRetriever.py
- 完整的RAG问答流程
- 语义相似性检索
- 上下文增强生成
- 支持重新排序和置信度评估

### 10.3 RAG技术特性

#### 向量化技术
- 使用sentence-transformers的all-MiniLM-L6-v2模型
- 支持中英文混合文档处理
- 自动文档去重和增量更新

#### 检索策略
- 语义相似性搜索
- 可配置的相似度阈值
- 智能重新排序机制
- 支持关键词和语义混合检索

#### 问答增强
- 上下文长度智能控制
- 来源信息自动追踪
- 置信度评估
- 支持模型和非模型两种回答模式

## 11. Meta-Chunking技术详解

### 11.1 Meta-Chunking概述
Meta-Chunking是一种利用大语言模型(LLM)能力灵活将文档分割成逻辑连贯、独立块的文本分块技术。其核心原理是：允许块大小的可变性，以更有效地捕捉和维护内容的逻辑完整性。

### 11.2 核心技术原理

#### 11.2.1 困惑度分块(PPL Chunking)
- 基于语言模型的困惑度评估文本连贯性
- 在确定性点进行分割，在不确定性点保持完整
- 利用语言模型的"幻觉"来感知文本边界
- 适用于长短文档的快速准确分块

#### 11.2.2 概率差分分块(Margin Sampling Chunking)
- 对连续句子是否需要分割进行二元分类判断
- 基于边缘采样的概率进行决策
- 通过动态阈值调整分块策略
- 支持本地小型模型的使用

#### 11.2.3 语义分块(Semantic Chunking)
- 基于语义相似度进行分块
- 保持语义完整性
- 适用于复杂文档结构

### 11.3 MoC架构
MoC(Mixtures of Text Chunking Learners)是一种混合分块专家架构：
- 通过多粒度感知路由网络动态调度轻量化分块专家
- 融合正则表达式引导的分块方法
- 基于稀疏激活的多粒度分块机制
- 编辑距离驱动的校正算法

### 11.4 评估指标
- **Boundary Clarity（边界清晰度）**：量化分块边界的清晰程度
- **Chunk Stickiness（块粘性）**：评估文本块的语义粘性

## 12. 系统依赖和配置

### 12.1 项目依赖
系统依赖的Python包包括：
- langchain-community>=0.3.27
- langchain-mcp-adapters>=0.1.9
- langchain-openai>=0.3.29
- langchain-tavily>=0.2.11
- langgraph-cli[inmem]>=0.3.6
- openai>=1.99.6
- chromadb>=0.4.22
- sentence-transformers>=2.2.2
- langchain-chroma>=0.1.2
- pypdf>=4.0.1
- python-docx>=1.1.0
- jieba==0.42.1
- accelerate>=1.10.1
- pymupdf>=1.26.4

### 12.2 环境配置
系统需要Python 3.12或更高版本，并使用uv包管理器进行依赖管理。

## 13. 测试和验证

### 13.1 RAG功能测试
系统包含完整的RAG功能测试脚本[test_rag.py](test_rag.py)，用于验证：
- 知识库初始状态检查
- 文档添加功能测试
- 文档检索功能测试
- 智能问答功能测试
- 知识库状态统计

### 13.2 测试文档
系统提供了丰富的测试文档，包括：
- [test_document.md](doc/test_document.md)：系统用户手册
- [中华人民共和国证券法(2019修订).pdf](doc/中华人民共和国证券法(2019修订).pdf)：PDF文档处理测试
- [rag_usage_guide.md](doc/rag_usage_guide.md)：RAG使用指南
- [RAG_INTEGRATION_REPORT.md](doc/RAG_INTEGRATION_REPORT.md)：RAG集成技术报告

## 14. 分块技术实现细节

### 14.1 困惑度分块(PPL Chunking)实现
在[Meta-Chunking-main/tools/ppl_chunking.py](Meta-Chunking-main/tools/ppl_chunking.py)中实现了基于困惑度的分块算法：
- 使用语言模型计算文本序列的困惑度
- 通过识别困惑度局部极小值点来确定分块边界
- 支持批量处理长文本以提高效率
- 提供动态阈值调整机制

### 14.2 概率差分分块(Margin Sampling Chunking)实现
在[Meta-Chunking-main/tools/margin_sampling_chunking.py](Meta-Chunking-main/tools/margin_sampling_chunking.py)中实现了基于概率差分的分块算法：
- 通过语言模型判断相邻句子是否应该分块
- 使用边缘采样技术计算分割概率
- 支持动态阈值调整以适应不同文本特征

### 14.3 LumberChunker实现
在[Meta-Chunking-main/tools/lumberchunker.py](Meta-Chunking-main/tools/lumberchunker.py)中实现了LumberChunker分块算法：
- 基于内容变化检测的分块方法
- 使用大语言模型识别段落间的语义变化点
- 支持中英文文本处理

## 15. 总结

本系统通过结合LangGraph框架和RAG技术，构建了一个功能强大的多智能体对话系统。系统具有良好的可扩展性和智能化水平，能够处理复杂的任务请求，为用户提供准确的信息服务。通过采用先进的Meta-Chunking文本分块技术，系统在文档处理和问答方面表现出色，具有广泛的应用前景。

## 16. 故障排除

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

### 17.调试技巧

- 使用 `get_rag_stats` 查看知识库状态
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


# 更新：系统核心模块文档

本文档详细描述了系统中两个核心模块的功能和实现：向量数据库管理器 (`VectorDatabase`) 和 RAG 检索器 (`RAGRetriever`)。

## 1. 向量数据库管理器 (VectorDatabase)

### 1.1 概述
`VectorDatabase` 类是系统的核心组件之一，负责管理向量数据库，实现 RAG (Retrieval-Augmented Generation) 功能。它基于 ChromaDB 实现，提供文档的存储、检索和管理功能。

### 1.2 主要功能
- **文档存储**: 将处理后的文档块存储到向量数据库中。
- **相似性搜索**: 根据查询文本，在向量数据库中搜索最相关的文档块。
- **文档管理**: 支持添加、删除、清空文档，以及获取数据库统计信息。
- **元数据管理**: 维护文档的元数据，如文件哈希值、块数量等。

### 1.3 核心方法
- `__init__`: 初始化向量数据库，包括创建或连接到 ChromaDB 集合，初始化嵌入模型等。
- `add_documents`: 将文档列表添加到向量数据库。
- `similarity_search`: 执行相似性搜索，返回最相关的文档。
- `similarity_search_with_score`: 执行相似性搜索，并返回每个文档的相似度分数。
- `get_stats`: 获取数据库的统计信息。
- `list_documents`: 列出所有已索引的文档。
- `delete_document`: 从数据库中删除指定文档。
- `clear_all`: 清空数据库中的所有数据。

### 1.4 技术细节
- **嵌入模型**: 使用 `OllamaEmbeddings` 生成文档和查询的向量表示。
- **数据库**: 使用 ChromaDB 作为向量数据库，存储文档向量和元数据。
- **文件监控**: 通过计算文件哈希值来监控文件是否发生变化，避免重复处理。

## 2. RAG 检索器 (RAGRetriever)

### 2.1 概述
`RAGRetriever` 类实现了完整的 RAG 问答功能。它利用向量数据库管理器检索相关文档，并结合语言模型生成回答。

### 2.2 主要功能
- **文档添加**: 支持添加单个文档或整个目录到知识库。
- **文档检索**: 根据用户问题，在知识库中检索相关文档。
- **问答生成**: 基于检索到的文档和语言模型，生成回答。
- **知识库管理**: 提供获取统计信息、删除文档、清空知识库等功能。

### 2.3 核心方法
- `__init__`: 初始化 RAG 检索器，包括向量数据库和文档处理器。
- `add_document_to_knowledge_base`: 将单个文档添加到知识库。
- `add_directory_to_knowledge_base`: 将目录下的所有文档添加到知识库。
- `retrieve_relevant_documents`: 检索与查询相关的文档。
- `answer_question`: 基于知识库回答问题。
- `get_knowledge_base_stats`: 获取知识库统计信息。
- `delete_document_from_knowledge_base`: 从知识库删除文档。
- `clear_knowledge_base`: 清空知识库。

### 2.4 技术细节
- **文档处理**: 使用 `DocumentProcessor` 处理不同格式的文档，并将其分块。
- **向量检索**: 利用 `VectorDatabase` 进行相似性搜索。
- **语言模型**: 可选地使用语言模型生成更智能的回答。
- **重新排序**: 对检索到的文档进行重新排序，以提高相关性。

## 3. 总结

`VectorDatabase` 和 `RAGRetriever` 是系统中两个紧密相关的模块。`VectorDatabase` 负责底层的向量存储和检索，而 `RAGRetriever` 则在此基础上构建了更高级的问答功能。这两个模块共同构成了系统的核心，为用户提供强大的文档检索和问答能力。


