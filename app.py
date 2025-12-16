import streamlit as st
import pandas as pd
import os
import re
import zipfile
import io
import time
import requests
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential

# ====================
# Core Classes for AI & Data
# ====================

class LLMClient:
    """处理与大模型的交互 (Embedding 和 Chat)"""
    def __init__(self, config):
        self.base_url = config.get('base_url', '').rstrip('/')
        self.api_key = config.get('api_key')
        self.model = config.get('model')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_embedding(self, text):
        """获取文本的向量表示 (默认尝试兼容 OpenAI 格式的 embedding 接口)"""
        # 注意：不同的模型商 Embedding URL 可能不同，这里默认使用 OpenAI 兼容路径
        # 对于 Gemini，通常是 models/embedding-001
        # 为了兼容性，这里做一个简单的路径适配，或者由用户指定 Embedding Model
        
        # 简化处理：尝试使用 text-embedding-004 或用户指定的通用 embedding 模型
        embedding_model = "text-embedding-004" # 默认一个较新的模型
        
        payload = {
            "input": text.replace("\n", " "),
            "model": embedding_model
        }
        
        # 尝试标准 OpenAI 路径
        url = f"{self.base_url}/embeddings"
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()['data'][0]['embedding']
            else:
                # 如果失败，对于 Gemini 可能是不同的路径，这里暂不做极其复杂的自动探测
                # 实际生产中应增加更多的 endpoint 适配
                print(f"Embedding failed: {response.text}")
                return None
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def chat_completion(self, system_prompt, user_prompt):
        """调用 Chat 接口"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload, timeout=60)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            # 清理可能的 markdown 标记
            content = content.replace("```json", "").replace("```", "")
            return json.loads(content)
        else:
            raise Exception(f"API Error {response.status_code}: {response.text}")

class VectorStore:
    """简单的内存向量数据库"""
    def __init__(self):
        self.documents = [] # 存储原文片段: {'id': int, 'text': str, 'source': str}
        self.vectors = []   # 存储对应的 numpy 向量
        self.llm_client = None

    def set_client(self, client):
        self.llm_client = client

    def add_documents(self, file_corpus):
        """
        处理并入库
        file_corpus: [{'name': 'filename', 'content': 'full text'}, ...]
        """
        # 1. Chunking (切片)
        chunk_size = 500 # 字符数
        overlap = 50 
        
        self.documents = []
        texts_to_embed = []
        
        doc_id = 0
        for file in file_corpus:
            text = file['content']
            name = file['name']
            
            # 简单的滑动窗口切片
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if len(chunk) < 50: continue # 跳过太短的
                
                self.documents.append({
                    'id': doc_id,
                    'text': chunk,
                    'source': name
                })
                texts_to_embed.append(chunk)
                doc_id += 1
        
        # 2. Embedding (批量或逐个)
        # 实际生产中应该 Batch API，这里简化为逐个但用 ThreadPool 加速
        if not self.llm_client:
            return
            
        vectors = []
        with st.status("正在对制度文档进行量化处理 (Embedding)...") as status:
            total = len(texts_to_embed)
            completed = 0
            
            # 使用并发加速 Embedding
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_idx = {executor.submit(self.llm_client.get_embedding, t): i for i, t in enumerate(texts_to_embed)}
                
                # 初始化一个定长列表
                results = [None] * total
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    vec = future.result()
                    results[idx] = vec
                    
                    completed += 1
                    if completed % 10 == 0:
                        status.update(label=f"正在量化文档... ({completed}/{total})")
            
            # 过滤掉失败的 Embedding (None) 并同步移除 document
            valid_vectors = []
            valid_docs = []
            for i, vec in enumerate(results):
                if vec is not None:
                    valid_vectors.append(vec)
                    valid_docs.append(self.documents[i])
            
            self.vectors = np.array(valid_vectors)
            self.documents = valid_docs
            status.update(label="文档量化完成！", state="complete")

    def search(self, query_text, top_k=3):
        """语义检索"""
        if self.llm_client is None or len(self.vectors) == 0:
            return []

        query_vec = self.llm_client.get_embedding(query_text)
        if query_vec is None:
            return []
            
        query_vec = np.array(query_vec)
        
        # 计算余弦相似度: (A . B) / (|A| * |B|)
        # 假设向量已经是归一化的（OpenAI embedding 通常是），则 dot product 即可
        # 为保险，手动计算归一化余弦相似度
        norm_vectors = np.linalg.norm(self.vectors, axis=1)
        norm_query = np.linalg.norm(query_vec)
        
        if norm_query == 0: return []
        
        similarities = np.dot(self.vectors, query_vec) / (norm_vectors * norm_query)
        
        # 获取 Top K
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            doc = self.documents[idx]
            results.append({
                'source': doc['source'],
                'content': doc['text'],
                'score': float(score)
            })
            
        return results

# ====================
# Helper Functions
# ====================

def process_uploaded_files(uploaded_files):
    """处理上传的文件"""
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.zip'):
            try:
                with zipfile.ZipFile(uploaded_file) as z:
                    for filename in z.namelist():
                        if filename.endswith('/') or filename.startswith('__MACOSX') or filename.startswith('._'):
                            continue
                        if filename.endswith(('.docx', '.xlsx')):
                            with z.open(filename) as f:
                                yield filename, f.read()
            except Exception as e:
                st.error(f"解压文件 {uploaded_file.name} 失败: {str(e)}")
        else:
            yield uploaded_file.name, uploaded_file.getvalue()

import docx

def extract_text_from_content(filename, content):
    """提取纯文本 (使用 robust 库)"""
    text = ""
    try:
        file_stream = io.BytesIO(content)
        if filename.endswith('.docx'):
            doc = docx.Document(file_stream)
            text = '\n'.join([para.text for para in doc.paragraphs])
        elif filename.endswith('.xlsx'):
            df_dict = pd.read_excel(file_stream, sheet_name=None, header=None)
            text_parts = []
            for sheet_name, df in df_dict.items():
                # 将每一行转换为字符串，用空格连接
                sheet_text = df.astype(str).apply(lambda x: ' '.join(x), axis=1)
                text_parts.append('\n'.join(sheet_text))
            text = '\n'.join(text_parts)
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
    return text

def parse_regulation_clauses(text):
    """解析法规条款 (优化版)"""
    pattern = r'(第\s*[\d零一二三四五六七八九十百]+\s*条|Article\s+\d+)'
    parts = re.split(pattern, text)
    
    clauses = []
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            title = parts[i].strip()
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            
            # 简单的适用性预判
            applicability = "适用"
            gov_keywords = ["国务院", "县级以上", "监察机关", "人民政府", "主管部门", "行政机关"]
            corp_keywords = ["生产经营单位", "企业", "用人单位", "建设单位", "公司"]
            
            content_lower = content.lower()
            is_gov = any(k in content_lower for k in gov_keywords)
            is_corp = any(k in content_lower for k in corp_keywords)
            
            if is_gov and not is_corp:
                applicability = "不适用(政府职责)"
            
            clauses.append({
                "条款号": title,
                "法规正文": title + " " + content,
                "适用性": applicability
            })
    return clauses

def evaluate_single_clause(clause, vector_store, llm_client):
    """
    单个条款的分析逻辑 (设计为并发调用)
    """
    row = {
        "条款号": clause['条款号'],
        "法规正文": clause['法规正文'],
        "评价结论": "❌缺失/不符合",
        "支撑证据": "未检索到相关制度",
        "匹配度": 0.0
    }
    
    if clause['适用性'] != "适用":
        row['评价结论'] = "❗不适用"
        row['支撑证据'] = "条款主体非企业"
        return row

    # 1. 语义检索 (Retrieval)
    # 阈值设定：如果相似度低于 0.35，认为根本没有相关制度，直接跳过 LLM
    search_results = vector_store.search(clause['法规正文'], top_k=3)
    
    if not search_results:
        return row
        
    top_score = search_results[0]['score']
    row['匹配度'] = top_score
    
    # 阈值过滤 (Pre-filtering)
    if top_score < 0.35:
        row['评价结论'] = "❌缺失/不符合"
        row['支撑证据'] = f"未找到匹配制度 (最高相似度 {top_score:.2f} 低于阈值)"
        return row
        
    # 2. LLM 评估 (Evaluation)
    evidence_text = ""
    for i, res in enumerate(search_results):
        evidence_text += f"参考片段 {i+1} (来源: {res['source']}, 相似度: {res['score']:.2f}):\n{res['content']}\n---\n"
    
    system_prompt = "你是一个EHS合规专家。请对比法规条款和企业制度，判断是否合规。"
    user_prompt = f"""
【法规条款】
{clause['法规正文']}

【企业制度参考片段】
{evidence_text}

请严格基于上述参考片段进行判断。如果不符合或片段不相关，请直说。
返回JSON格式:
{{
    "status": "✅完全符合" 或 "⚠️部分符合/需完善" 或 "❌缺失/不符合",
    "evidence": "简要引用的制度内容",
    "reason": "一句话判定理由"
}}
"""
    try:
        result = llm_client.chat_completion(system_prompt, user_prompt)
        row['评价结论'] = result.get('status', '❌缺失/不符合')
        row['支撑证据'] = f"{result.get('evidence', '')}\n(AI理由: {result.get('reason', '')})"
    except Exception as e:
        row['支撑证据'] = f"LLM分析失败: {str(e)}"
        
    return row

# ====================
# Streamlit UI
# ====================

st.set_page_config(page_title="EHS智能合规引擎 (Pro版)", layout="wide")

if 'results' not in st.session_state:
    st.session_state.results = None

st.title("🛡️ EHS法规合规性智能评价引擎 (Pro版)")
st.markdown("🚀 **核心升级**：采用 `Embedding语义向量化` + `并发加速`，大幅提升准确率与分析速度。")

with st.sidebar:
    st.header("1. 配置与上传")
    llm_base_url = st.text_input("API Base URL", value="https://generativelanguage.googleapis.com/v1beta/openai", help="OpenAI 兼容接口地址")
    llm_api_key = st.text_input("API Key", type="password")
    llm_model_name = st.text_input("Model Name", value="gemini-2.0-flash")
    
    st.divider()
    reg_files = st.file_uploader("上传法规 (docx/zip)", type=['docx', 'zip'], accept_multiple_files=True, key="reg")
    policy_files = st.file_uploader("上传制度 (docx/xlsx/zip)", type=['docx', 'xlsx', 'zip'], accept_multiple_files=True, key="pol")

if st.button("🚀 开始极速分析", type="primary"):
    if not (reg_files and policy_files and llm_api_key):
        st.error("请确保文件已上传且 API Key 已填写。" )
    else:
        # 初始化组件
        llm_config = {"base_url": llm_base_url, "api_key": llm_api_key, "model": llm_model_name}
        client = LLMClient(llm_config)
        vector_store = VectorStore()
        vector_store.set_client(client)
        
        # 1. 处理制度库 (构建向量索引)
        policy_corpus = []
        for name, content in process_uploaded_files(policy_files):
            text = extract_text_from_content(name, content)
            if text: policy_corpus.append({'name': name, 'content': text})
            
        if not policy_corpus:
            st.error("无法从制度文件中提取文本。" )
            st.stop()
            
        vector_store.add_documents(policy_corpus)
        
        # 2. 解析法规
        all_clauses = []
        for name, content in process_uploaded_files(reg_files):
            text = extract_text_from_content(name, content)
            clauses = parse_regulation_clauses(text)
            for c in clauses:
                c['source_file'] = name # 记录来源
                all_clauses.append(c)
                
        st.info(f"共解析出 {len(all_clauses)} 条法规条款，正在并发分析中...")
        
        # 3. 并发分析 (Map-Reduce)
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_list = []
        
        total_tasks = len(all_clauses)
        completed_tasks = 0
        
        # 开启线程池 (IO密集型任务，适合多线程)
        with ThreadPoolExecutor(max_workers=10) as executor:
            # 提交所有任务
            future_to_clause = {
                executor.submit(evaluate_single_clause, clause, vector_store, client): clause 
                for clause in all_clauses
            }
            
            for future in as_completed(future_to_clause):
                try:
                    res = future.result()
                    res['法规文件'] = future_to_clause[future]['source_file']
                    results_list.append(res)
                except Exception as exc:
                    st.warning(f"某条款分析异常: {exc}")
                
                completed_tasks += 1
                progress_bar.progress(completed_tasks / total_tasks)
                status_text.text(f"已完成: {completed_tasks}/{total_tasks} ...")
                
        st.success("分析完成！")
        st.session_state.results = pd.DataFrame(results_list)

# --- 结果展示 (即使刷新页面，只要 session_state 在就能显示) ---
if st.session_state.results is not None:
    df = st.session_state.results
    
    st.divider()
    st.subheader("📊 分析结果看板")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("完全符合", len(df[df['评价结论']=="✅完全符合"]))
    col2.metric("需完善", len(df[df['评价结论']=="⚠️部分符合/需完善"]))
    col3.metric("缺失/不符合", len(df[df['评价结论'].str.contains("缺失|不符合")]))
    
    # 筛选器
    status_filter = st.multiselect("筛选结论", df['评价结论'].unique(), default=df['评价结论'].unique())
    show_df = df[df['评价结论'].isin(status_filter)]
    
    st.dataframe(
        show_df,
        column_config={
            "法规正文": st.column_config.TextColumn("法规要求", width="medium"),
            "支撑证据": st.column_config.TextColumn("制度证据 & AI理由", width="large"),
            "匹配度": st.column_config.ProgressColumn("语义相似度", min_value=0, max_value=1, format="%.2f")
        },
        use_container_width=True,
        height=600
    )
    
    # 下载
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 下载详细报表 (CSV)", csv, "ehs_compliance_report.csv", "text/csv")