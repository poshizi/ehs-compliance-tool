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
import docx 
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
        self.embedding_model = config.get('embedding_model', 'text-embedding-004')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.embedding_failed = False 

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_embedding(self, text):
        """获取文本的向量表示"""
        if self.embedding_failed: return None
        
        payload = {
            "input": text.replace("\n", " "),
            "model": self.embedding_model
        }
        
        # 尝试标准 OpenAI 路径
        url = f"{self.base_url}/embeddings"
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    return data['data'][0]['embedding']
                else:
                    return None
            else:
                print(f"Embedding failed: {response.status_code} - {response.text}")
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
        
        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload, timeout=60)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                content = content.replace("```json", "").replace("```", "")
                return json.loads(content)
            else:
                raise Exception(f"API Error {response.status_code}: {response.text}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
            
    def test_connection(self):
        """测试连接状态"""
        results = {"chat": False, "embedding": False, "msg": ""}
        
        # 1. 测试 Chat
        try:
            self.chat_completion("You are a test bot.", "Reply JSON: {'status': 'ok'}")
            results['chat'] = True
        except Exception as e:
            results['msg'] += f"Chat Error: {str(e)}\n"
            
        # 2. 测试 Embedding
        try:
            emb = self.get_embedding("test")
            if emb:
                results['embedding'] = True
            else:
                results['msg'] += "Embedding Error: Returned None (Check model name or API support)\n"
        except Exception as e:
             results['msg'] += f"Embedding Exception: {str(e)}\n"
             
        return results

class VectorStore:
    """混合检索数据库 (向量 + 关键词)"""
    def __init__(self):
        self.documents = [] 
        self.vectors = []   
        self.llm_client = None

    def set_client(self, client):
        self.llm_client = client

    def add_documents(self, file_corpus):
        """处理并入库"""
        chunk_size = 500
        overlap = 50 
        
        self.documents = []
        texts_to_embed = []
        
        doc_id = 0
        for file in file_corpus:
            text = file['content']
            name = file['name']
            
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if len(chunk) < 50: continue 
                
                # 关键词提取
                keywords = set(re.split(r'[，。；：\s]', chunk))
                keywords = [k for k in keywords if len(k) > 1]

                self.documents.append({
                    'id': doc_id,
                    'text': chunk,
                    'source': name,
                    'keywords': keywords 
                })
                texts_to_embed.append(chunk)
                doc_id += 1
        
        # 尝试向量化
        if self.llm_client:
            with st.status("正在构建索引 (尝试向量化 + 关键词库)...") as status:
                valid_vectors = []
                # 使用并发
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_idx = {executor.submit(self.llm_client.get_embedding, t): i for i, t in enumerate(texts_to_embed)}
                    
                    results = [None] * len(texts_to_embed)
                    success_count = 0
                    
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        vec = future.result()
                        results[idx] = vec
                        if vec is not None: success_count += 1
                
                self.vectors = results 
                
                if success_count == 0:
                    status.update(label="⚠️ 向量化全部失败 (将降级使用关键词检索)", state="error")
                    st.warning("提示：Embedding API 调用失败，已自动切换为 **关键词匹配模式**。请检查 Embedding Model 配置。" )
                else:
                    status.update(label=f"索引构建完成 (向量化成功率: {success_count}/{len(texts_to_embed)})", state="complete")

    def search(self, query_text, top_k=3):
        """混合检索"""
        vec_results = []
        
        # 1. 向量检索
        has_vectors = any(v is not None for v in self.vectors)
        if has_vectors and self.llm_client:
            query_vec = self.llm_client.get_embedding(query_text)
            if query_vec is not None:
                q_v = np.array(query_vec)
                norm_q = np.linalg.norm(q_v)
                if norm_q > 0:
                    scores = []
                    for i, doc_vec in enumerate(self.vectors):
                        if doc_vec is None: 
                            scores.append(-1)
                            continue
                        d_v = np.array(doc_vec)
                        norm_d = np.linalg.norm(d_v)
                        if norm_d == 0: scores.append(0)
                        else: scores.append(np.dot(d_v, q_v) / (norm_d * norm_q))
                    
                    top_indices = np.argsort(scores)[-top_k:][::-1]
                    for idx in top_indices:
                        if scores[idx] > 0:
                            vec_results.append({'doc': self.documents[idx], 'score': float(scores[idx]), 'method': 'vector'})

        # 2. 关键词检索 (兜底)
        kw_results = []
        query_keywords = [k for k in re.split(r'[，。；：\s]', query_text) if len(k) > 1]
        
        for doc in self.documents:
            overlap = sum(1 for k in query_keywords if k in doc['text'])
            if overlap > 0:
                score = overlap / (len(query_keywords) + 1) * 0.8 
                kw_results.append({'doc': doc, 'score': score, 'method': 'keyword'})
        
        kw_results.sort(key=lambda x: x['score'], reverse=True)
        kw_results = kw_results[:top_k]

        combined = vec_results + kw_results
        seen_ids = set()
        final_results = []
        combined.sort(key=lambda x: x['score'], reverse=True)
        
        for res in combined:
            did = res['doc']['id']
            if did not in seen_ids:
                final_results.append({'source': res['doc']['source'], 'content': res['doc']['text'], 'score': res['score']})
                seen_ids.add(did)
            if len(final_results) >= top_k: break
                
        return final_results

# ====================
# Helper Functions
# ====================

def process_uploaded_files(uploaded_files):
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.zip'):
            try:
                with zipfile.ZipFile(uploaded_file) as z:
                    for filename in z.namelist():
                        if filename.endswith('/') or filename.startswith('__MACOSX') or filename.startswith('._'): continue
                        if filename.endswith(('.docx', '.xlsx')):
                            with z.open(filename) as f: yield filename, f.read()
            except Exception as e: st.error(f"解压失败: {e}")
        else: yield uploaded_file.name, uploaded_file.getvalue()

def extract_text_from_content(filename, content):
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
                sheet_text = df.astype(str).apply(lambda x: ' '.join(x), axis=1)
                text_parts.append('\n'.join(sheet_text))
            text = '\n'.join(text_parts)
    except Exception as e: print(f"Error parsing {filename}: {e}")
    return text

def parse_regulation_clauses(text):
    pattern = r'(第\s*[\d零一二三四五六七八九十百]+\s*条|Article\s+\d+)'
    parts = re.split(pattern, text)
    clauses = []
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            title = parts[i].strip()
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            applicability = "适用"
            gov_keywords = ["国务院", "县级以上", "监察机关", "人民政府", "主管部门", "行政机关"]
            corp_keywords = ["生产经营单位", "企业", "用人单位", "建设单位", "公司"]
            content_lower = content.lower()
            is_gov = any(k in content_lower for k in gov_keywords)
            is_corp = any(k in content_lower for k in corp_keywords)
            if is_gov and not is_corp: applicability = "不适用(政府职责)"
            clauses.append({"条款号": title, "法规正文": title + " " + content, "适用性": applicability})
    return clauses

def evaluate_single_clause(clause, vector_store, llm_client):
    row = {"条款号": clause['条款号'], "法规正文": clause['法规正文'], "评价结论": "❌缺失/不符合", "支撑证据": "未检索到相关制度", "匹配度": 0.0}
    if clause['适用性'] != "适用":
        row['评价结论'] = "❗不适用"
        row['支撑证据'] = "条款主体非企业"
        return row

    search_results = vector_store.search(clause['法规正文'], top_k=3)
    if not search_results: return row
    
    top_score = search_results[0]['score']
    row['匹配度'] = top_score
    if top_score < 0.15:
        row['评价结论'] = "❌缺失/不符合"
        row['支撑证据'] = f"未找到匹配制度 (最高匹配度 {top_score:.2f} 低于阈值)"
        return row
        
    evidence_text = ""
    for i, res in enumerate(search_results):
        evidence_text += f"片段 {i+1} (相似度: {res['score']:.2f}):\n{res['content']}\n---\n"
    
    system_prompt = "你是一个EHS合规专家。请对比法规条款和企业制度，判断是否合规。"
    user_prompt = f"【法规条款】\n{clause['法规正文']}\n\n【企业制度参考片段】\n{evidence_text}\n\n请基于上述片段判断。若符合，引用原文。\n返回JSON:\n{{\n    "status": "✅完全符合" 或 "⚠️部分符合/需完善" 或 "❌缺失/不符合",\n    "evidence": "制度原文引用",\n    "reason": "判定理由"\n}}"
    
    try:
        result = llm_client.chat_completion(system_prompt, user_prompt)
        row['评价结论'] = result.get('status', '❌缺失/不符合')
        row['支撑证据'] = f"{result.get('evidence', '')}\n(AI理由: {result.get('reason', '')})"
    except Exception as e: row['支撑证据'] = f"LLM分析失败: {str(e)}"
    return row

# ====================
# Streamlit UI
# ====================

st.set_page_config(page_title="EHS智能合规引擎 (Hybrid版)", layout="wide")

if 'results' not in st.session_state: st.session_state.results = None

st.title("🛡️ EHS法规合规性智能评价引擎 (Hybrid Pro)")
st.markdown("🚀 **技术栈**: `Embedding` + `Hybrid Search` + `Concurrency`")

with st.sidebar:
    st.header("1. API 配置")
    llm_base_url = st.text_input("API Base URL", value="https://generativelanguage.googleapis.com/v1beta/openai", help="例如 https://api.openai.com/v1")
    llm_api_key = st.text_input("API Key", type="password")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        llm_model_name = st.text_input("Chat Model", value="gemini-2.0-flash")
    with col_m2:
        # 新增 Embedding Model 选择
        embedding_model_name = st.text_input("Embedding Model", value="text-embedding-004", help="例如 text-embedding-3-small")

    # 新增测试按钮
    if st.button("🔌 测试 API 连接", use_container_width=True):
        if not llm_api_key:
            st.error("请先填写 API Key")
        else:
            with st.spinner("正在测试连接..."):
                test_config = {
                    "base_url": llm_base_url,
                    "api_key": llm_api_key,
                    "model": llm_model_name,
                    "embedding_model": embedding_model_name
                }
                client = LLMClient(test_config)
                res = client.test_connection()
                
                if res['chat']: st.success(f"✅ Chat Model ({llm_model_name}): 连接成功")
                else: st.error(f"❌ Chat Model 连接失败")
                
                if res['embedding']: st.success(f"✅ Embedding Model ({embedding_model_name}): 连接成功")
                else: st.error(f"❌ Embedding Model 连接失败 (系统将自动降级为关键词检索)")
                
                if res['msg']: st.code(res['msg'], language="text")

    st.divider()
    st.header("2. 文件上传")
    reg_files = st.file_uploader("上传法规 (docx/zip)", type=['docx', 'zip'], accept_multiple_files=True, key="reg")
    policy_files = st.file_uploader("上传制度 (docx/xlsx/zip)", type=['docx', 'xlsx', 'zip'], accept_multiple_files=True, key="pol")

if st.button("🚀 开始极速分析", type="primary"):
    if not (reg_files and policy_files and llm_api_key):
        st.error("请确保文件已上传且 API Key 已填写。" )
    else:
        llm_config = {
            "base_url": llm_base_url, 
            "api_key": llm_api_key, 
            "model": llm_model_name,
            "embedding_model": embedding_model_name
        }
        client = LLMClient(llm_config)
        vector_store = VectorStore()
        vector_store.set_client(client)
        
        policy_corpus = []
        for name, content in process_uploaded_files(policy_files):
            text = extract_text_from_content(name, content)
            if text and len(text.strip()) > 0: policy_corpus.append({'name': name, 'content': text})
            else: st.warning(f"文件 {name} 内容为空，已跳过。" )
            
        if not policy_corpus:
            st.error("有效制度内容为空。" )
            st.stop()
            
        vector_store.add_documents(policy_corpus)
        
        all_clauses = []
        for name, content in process_uploaded_files(reg_files):
            text = extract_text_from_content(name, content)
            clauses = parse_regulation_clauses(text)
            for c in clauses:
                c['source_file'] = name 
                all_clauses.append(c)
        
        if not all_clauses:
             st.error("未解析出任何法规条款。" )
             st.stop()

        st.info(f"共 {len(all_clauses)} 条条款，开始分析...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_list = []
        total_tasks = len(all_clauses)
        completed_tasks = 0
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_clause = {executor.submit(evaluate_single_clause, clause, vector_store, client): clause for clause in all_clauses}
            for future in as_completed(future_to_clause):
                try:
                    res = future.result()
                    res['法规文件'] = future_to_clause[future]['source_file']
                    results_list.append(res)
                except Exception as exc: st.warning(f"分析异常: {exc}")
                completed_tasks += 1
                progress_bar.progress(completed_tasks / total_tasks)
                status_text.text(f"已完成: {completed_tasks}/{total_tasks} ...")
                
        st.success("分析完成！")
        st.session_state.results = pd.DataFrame(results_list)

if st.session_state.results is not None:
    df = st.session_state.results
    st.divider()
    st.subheader("📊 结果看板")
    col1, col2, col3 = st.columns(3)
    col1.metric("完全符合", len(df[df['评价结论']=="✅完全符合"]))
    col2.metric("需完善", len(df[df['评价结论']=="⚠️部分符合/需完善"]))
    col3.metric("缺失/不符合", len(df[df['评价结论'].str.contains("缺失|不符合")]))
    
    status_filter = st.multiselect("筛选结论", df['评价结论'].unique(), default=df['评价结论'].unique())
    show_df = df[df['评价结论'].isin(status_filter)]
    st.dataframe(show_df, column_config={"法规正文": st.column_config.TextColumn("法规要求", width="medium"), "支撑证据": st.column_config.TextColumn("证据", width="large"), "匹配度": st.column_config.ProgressColumn("匹配度", min_value=0, max_value=1, format="%.2f")}, use_container_width=True, height=600)
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 下载详细报表 (CSV)", csv, "ehs_compliance_report.csv", "text/csv")