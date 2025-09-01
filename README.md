# 项目配置
```shell
git clone `````  # 拉取项目
cd langgraph_learn
# 使用uv管理虚拟环境,先确保已经有uv 了
uv --version
# 输出版本
uv venv # 创建虚拟环境
uv sync # 根据pyproject.toml文件安装依赖

uv run langgraph_chat.py # 运行

```


# 项目说明

## Tools
- 文档输出工具(document_exporter.py)
- 文档读取工具(DocumnetReader.py)
- 上下文管理器(MessageManager.py)
- 路径获取工具(Path_Acquire.py)
- 搜索引擎工具(TavilySearcher.py)

