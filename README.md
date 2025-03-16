# langchain

是什么：为了简化使用大模型语言，模型构建端到端应用程序的过程，是LLM大模型与AI应用的粘合剂

能做什么：

- LLMs & Prompt：提供了目前市面上几乎所有LLM的通用接口，同时还提供了提示词的管理和优化能力，同时也提供
  了非常多的相关适用工具，以方便开发人员利用LangChain与LLMs进行交互
- Chains：LangChain把提示词、大语言模型、结果解析封装成Chain，并提供标准的接口，以便允许不同的Chain
  形成交互序列，为Al原生应用提供了端到端的Chain
- RAG：检索增强生成（Retrieval Augemented Generation）是为了预训练语料数据无法及时更新，而带来的回答内容陈旧的解决方式。LangChain
  提供了支持检索增强生成式的Chain。在使用时，这些Chain会首先与外部数据源进行交互以获得对应数据，然后再利用获得的数据与LLMS进行交互。典型的应用场景如：基于特定数据源的问答机器人。
- Agent：对于一个任务，代理主要涉及让LLMS来对任务进行拆分、执行该行动、并观察执行结果。代理会重复执行这个过程，直到该任务完成为止。LangChain为代理提供了标准接口，可供选择的代理，以及一些端到端的代理的示例。
- Memory：指的是chain或agent调用之间的状态持久化，LangChain为Memory提供了标准接口,并提供了一系列的内存实现。
- Evaluation：LangChain还提供了非常多的评估能力以允许我们可以更方便的对LLMS进行评估

优劣势

环境搭建

```sh
# langchain 安装
pip install --upgrade langchain -i https://pypi.org/simple

# openai api 安装
pip install openai=v0.28.1 -i https://pypi.org/simple
```

设置key到环境变量

```python
# 设置环境变量
# linux
export OPENAI_API_KEY="your_api_key_here"
# windows
setx OPENAI_API_KEY "your_api_key_here"
# python
import os
os.environ["OPENAI_API_KEY"] = "sk-9npZpELUzM0IxRU1B79bF5B9FfC04bDeA730F4090dF782A3"
os.environ["OPENAI_PROXY"] = "https://ai-yyds.com/v1"

# 读取环境变量
import os
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_PROXY")

```



## prompts模板

优秀的提示词：

- 【立角色】：引导AI进入具体场景，赋予其行家身份
- 【述问题】：告诉AI你的困惑和问题，以及背景信息
- 【定目标】：告诉AI你的需求，希望达成的目标
- 【补要求】：告诉AI回答时注意什么，或者如何回复

提示词模板：

1. 将提示词提炼成模板
2. 实现提示词复用、版本管理、动态变化等

### 示例器选择

### 流式调用

### 输出格式控制



## 核心组件Chain

### 四种内置链

- LLMChain

  - 最常使用的链

  - 提示词模板 + （LLM/chatModes）+ 输出格式化器

  - 支持多种调用方式
- SequentialChain
  - 顺序执行
  - 将前一个LLM的输入作用下一个LLM的输入
- RouterChain
  - 路由链支持创建一个非确定性链，由LLM来选择下一步
  - 链内的多个prompts模板描述了不同的提示请求
- Transformation
  - 支持对传递部件的转换
  - 如将一个超长文本过滤转换为仅包含前三个段落，然后提交给LLM

### 内置文档链

- Stuff：最常见的文档链，将文档直接塞进prompt中，为LLM回答问题提供上下文资料，适合小文档场景
- Refine：通过循环输入文档并迭代更新答案来构建响应，一次只传递给LLM一个文档，将文档不断投喂，并产生各种中间答案，适合逻辑有上下文关联的文档，不适合交叉引用的文档
- Map reduce：先将每个文档或文档块分别投喂给LLM，并得到结果集（Map步骤），然后通过一个文档合并链，获得一个输出结果（Reduce步骤）
- Map re-mark：先将每个文档或文档块投喂给LLM,并对每个文档或文档块生成问题的答案进行打分，然后将打分最高的文档或文档块作为最终答案返回

### Memory

解决大模型无状态问题

- 利用内存实现短时记忆
- 利用Entity memory构建实体记忆
- 利用知识图谱来构建记忆
- 利用对话摘要来兼容内存中的长对话
- 使用token来刷新内存缓冲区
- 使用向量数据库实现长时记忆



## RAG

为LLM提供来自外部知识源的额外信息的概念，这允许它们生成更准确和有上下文的答案

1. 检索:外部相似搜索
2. 增强:提示词更新
3. 生成:更详细的提示词输入LLM

### loader

多种文档格式读取解析数据

### 文档转换分割

将文档按照Token完整性进行拆分、关键词提取、文档摘要

### 如何控制长文本切割精准度低问题

### 文本向量化

使用Embedding模型，对分割后文本进行向量化，便于后续向量检索



## Agent

AlAgents是基于LLM的能够自主理解、自主规划决策、执行复杂任务的智能体，Agents不是chatGPT的升级版，它不仅告诉你"如何做"，更会帮你去做，如各种Copilot是副驾驶，那么Agents就是主驾使

Agents=LLM+规划技能+记忆+工具使用

本质上Agents是一个LLM的编排与执行系统

### 常见类型

- OPENAI_FUNCTIONS：openai函数调用型
- ZERO_SHOT_REACT_DESCRIPTION：零样本增强生成型
- CHAT_ZERO_SHOT_REACT_DESCRIPTION：零样本增强生成型(对话)
- CONVERSATIONAL_REACT_DESCRIPTION：对话增强生成型
- STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION：结构化对话生成增强型

### 工具使用

### 记忆实现

### 语言表达式



## 示例

- 服务器端：接口 -> langchain -> openai\ollama。
- 客户端：电报机器人、微信机器人、website。
- 接口：http,https,websocket

### 服务器：

1. 接口访问，python选型fastapi
2. /chat的接口，post请求
3. /add_ursl 从url中学习知识
4. /add_pdfs 从pdf里学习知识
5. /add_texts 从txt文本里学习

### 人性化

1. 用户输入-> AI判断一下当前问题的情绪倾向？-> 判断 -> 反馈 -> agent判断
2. 工具调用： 用户发起请求 -> agent判断使用哪个工具 -> 带着相关的参数去请求工具 -> 得到观察结果

### 示例能力

1. api
2. angent框架
3. tools:搜索、查询信息、专业知识库
4. 记忆，长时记忆
5. 学习能力

### 从url来学习，实现增强

1. 输入URL
2. 地址的HTML变成文本
3. 向量化
4. 检索 -> 相关文本块
5. LLM回答
