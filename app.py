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
import pickle
import hashlib
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
from webdavclient3.client import Client as WebDavClient

# ====================
# Configuration & Constants
# ====================
CACHE_DIR = "vector_store_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

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
        url = f"{self.base_url}/embeddings"
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    return data['data'][0]['embedding']
            else:
                print(f"Embedding failed: {response.status_code}")
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
        results = {"chat": False, "embedding": False, "msg": ""}
        try:
            self.chat_completion("Test", "Reply JSON: {'status': 'ok'}")
            results['chat'] = True
        except Exception as e: results['msg'] += f"Chat Error: {e}\n"
        try:
            if self.get_embedding("test"): results['embedding'] = True
            else: results['msg'] += "Embedding Error: None\n"
        except Exception as e: results['msg'] += f"Embedding Exception: {e}\n"
        return results

class VectorStore:
    """持久化向量数据库"""
    def __init__(self):
        self.documents = [] 
        self.vectors = []   
        self.llm_client = None
        self.index_name = "default_index"

    def set_client(self, client):
        self.llm_client = client

    def save_to_disk(self, name="default"):
        """保存索引到磁盘"""
        path = os.path.join(CACHE_DIR, f"{name}.pkl")
        data = {
            "documents": self.documents,
            "vectors": self.vectors
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return path

    def load_from_disk(self, name="default"):
        """从磁盘加载索引"""
        path = os.path.join(CACHE_DIR, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.vectors = data["vectors"]
            return True
        return False

    def add_documents(self, file_corpus):
        """处理并入库"""
        chunk_size = 500
        overlap = 50 
        
        new_docs = []
        texts_to_embed = []
        start_doc_id = len(self.documents)
        
        for file in file_corpus:
            text = file['content']
            name = file['name']
            
            # 简单的查重 (基于文件名)
            if any(d['source'] == name for d in self.documents):
                print(f"Skipping {name}, already exists.")
                continue

            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if len(chunk) < 50: continue 
                
                keywords = set(re.split(r'[，。；：\s]', chunk))
                keywords = [k for k in keywords if len(k) > 1]

                new_docs.append({
                    'id': start_doc_id,
                    'text': chunk,
                    'source': name,
                    'keywords': keywords 
                })
                texts_to_embed.append(chunk)
                start_doc_id += 1
        
        if not texts_to_embed:
            return

        # 向量化
        if self.llm_client:
            with st.status(f"正在向量化 {len(texts_to_embed)} 个新片段...") as status:
                new_vectors = [None] * len(texts_to_embed)
                success_count = 0
                
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_idx = {executor.submit(self.llm_client.get_embedding, t): i for i, t in enumerate(texts_to_embed)}
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        vec = future.result()
                        new_vectors[idx] = vec
                        if vec is not None: success_count += 1
                
                # 合并
                self.documents.extend(new_docs)
                if len(self.vectors) == 0:
                    self.vectors = new_vectors
                else:
                    self.vectors = list(self.vectors) + new_vectors # Convert back to list to extend
                
                status.update(label=f"入库完成 (成功率: {success_count}/{len(texts_to_embed)})", state="complete")

    def search(self, query_text, top_k=3):
        """混合检索"""
        vec_results = []
        
        # 1. 向量检索
        # 过滤 None 向量
        valid_indices = [i for i, v in enumerate(self.vectors) if v is not None]
        
        if valid_indices and self.llm_client:
            query_vec = self.llm_client.get_embedding(query_text)
            if query_vec is not None:
                q_v = np.array(query_vec)
                norm_q = np.linalg.norm(q_v)
                
                # 构建矩阵
                matrix = np.array([self.vectors[i] for i in valid_indices])
                norm_matrix = np.linalg.norm(matrix, axis=1)
                
                if norm_q > 0:
                    # Cosine Sim
                    scores = np.dot(matrix, q_v) / (norm_matrix * norm_q)
                    
                    # 获取 Top K
                    top_k_indices = np.argsort(scores)[-top_k:][::-1]
                    
                    for idx_in_valid in top_k_indices:
                        real_idx = valid_indices[idx_in_valid]
                        score = scores[idx_in_valid]
                        if score > 0:
                            vec_results.append({'doc': self.documents[real_idx], 'score': float(score), 'method': 'vector'})

        # 2. 关键词检索
        kw_results = []
        query_keywords = [k for k in re.split(r'[，。；：\s]', query_text) if len(k) > 1]
        
        for doc in self.documents:
            overlap = sum(1 for k in query_keywords if k in doc['keywords']) # 使用预存的keywords集合加速
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
            clauses.append({"条款号": title, "法规正文": title + " " + content, "适用性": "适用"})
    return clauses

def evaluate_single_clause(clause, vector_store, llm_client):
    row = {"条款号": clause['条款号'], "法规正文": clause['法规正文'], "评价结论": "❌缺失/不符合", "差距分析": "未检索到相关制度", "改进建议": "请补充相关管理规定", "支撑证据": "无", "匹配度": 0.0}

    search_results = vector_store.search(clause['法规正文'], top_k=3)
    top_score = search_results[0]['score'] if search_results else 0
    row['匹配度'] = top_score
        
    evidence_text = ""
    if search_results:
        for i, res in enumerate(search_results):
            evidence_text += f"参考制度片段 {i+1} (来源: {res['source']}):\n{res['content'][:800]}\n---\n"
    else:
        evidence_text = "未检索到任何相关的企业内部制度文档。"
    
    system_prompt = """你是一名具有20年经验的EHS管理专家，精通中国EHS法规标准，擅长仓储物流场景。
    请对给定的法规条款进行合规性评价。严格执行以下思维链：
    1. 解读：理解条款核心要求（人机料法环），判定是否适用于物流仓储企业。如果不适用，直接标记“不适用”。
    2. 比对：对比法规要求与提供的企业制度片段。是否覆盖所有要素？针对物流场景是否具体可执行？
    3. 判定：给出定性结论。
    """
    
    user_prompt = f"""
    【法规条款】
    {clause['法规正文']}

    【企业制度现状（检索到的最相关片段）】
    {evidence_text}

    【任务要求】
    请严格按照以下格式返回 JSON 结果：
    {{
        "applicability": "适用" 或 "不适用",
        "compliance_status": "完全符合" 或 "部分符合/需完善" 或 "缺失/不符合" 或 "不适用",
        "gap_analysis": "150字以内。若不适用填'不适用'。若适用，分析具体缺了什么或说明合规点。",
        "improvement_suggestion": "若完全符合填'无'。否则给出1-2条建议。",
        "evidence_summary": "列出最匹配的制度名称及关键句摘要"
    }}
    """
    try:
        result = llm_client.chat_completion(system_prompt, user_prompt)
        status = result.get('compliance_status', '缺失/不符合')
        if "完全符合" in status: status = "✅完全符合"
        elif "部分" in status or "需完善" in status: status = "⚠️部分符合/需完善"
        elif "不适用" in status: status = "❗不适用"
        else: status = "❌缺失/不符合"
        
        row['评价结论'] = status
        row['差距分析'] = result.get('gap_analysis', '无分析')
        row['改进建议'] = result.get('improvement_suggestion', '无建议')
        row['支撑证据'] = result.get('evidence_summary', '无')
    except Exception as e:
        row['差距分析'] = f"LLM分析失败: {str(e)}"
    return row

def generate_word_report(df_results, summary_stats):
    doc = Document()
    title = doc.add_heading('EHS法规合规性评价报告', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(f"评价日期: {time.strftime('%Y-%m-%d')}")
    
    doc.add_heading('第一部分：总体评价', level=1)
    p = doc.add_paragraph()
    p.add_run(f"本次共分析法规条款 {summary_stats['total']} 条。\n")
    p.add_run(f"✅ 完全符合: {summary_stats['compliant']} 条\n")
    p.add_run(f"⚠️ 部分符合/需完善: {summary_stats['partial']} 条\n")
    p.add_run(f"❌ 缺失/不符合: {summary_stats['non_compliant']} 条\n")
    
    doc.add_heading('第二部分：详细合规性评价矩阵', level=1)
    table = doc.add_table(rows=1, cols=6)
    table.style = 'Table Grid'
    headers = ["序号", "法规条款", "评价结论", "差距分析与论据", "改进建议", "支撑证据"]
    for i, header in enumerate(headers):
        table.rows[0].cells[i].text = header
        
    for idx, row in df_results.iterrows():
        row_cells = table.add_row().cells
        row_cells[0].text = str(row.get('序号', idx + 1))
        row_cells[1].text = str(row.get('法规正文', ''))
        row_cells[2].text = str(row.get('评价结论', ''))
        row_cells[3].text = str(row.get('差距分析', ''))
        row_cells[4].text = str(row.get('改进建议', ''))
        row_cells[5].text = str(row.get('支撑证据', ''))
        
    f = io.BytesIO()
    doc.save(f)
    f.seek(0)
    return f

# ====================
# Streamlit UI
# ====================

st.set_page_config(page_title="EHS专家合规系统 (Enterprise)", layout="wide")

if 'results' not in st.session_state: st.session_state.results = None
if 'vector_store' not in st.session_state: st.session_state.vector_store = VectorStore()

st.title("🛡️ EHS法规合规性智能评价系统 (Enterprise)")

with st.sidebar:
    st.header("1. API 配置")
    llm_base_url = st.text_input("API Base URL", value="https://generativelanguage.googleapis.com/v1beta/openai")
    llm_api_key = st.text_input("API Key", type="password")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1: llm_model_name = st.text_input("Chat Model", value="gemini-2.0-flash")
    with col_m2: embedding_model_name = st.text_input("Embedding Model", value="text-embedding-004")
    
    # 初始化 LLM
    if llm_api_key:
        llm_config = {"base_url": llm_base_url, "api_key": llm_api_key, "model": llm_model_name, "embedding_model": embedding_model_name}
        client = LLMClient(llm_config)
        st.session_state.vector_store.set_client(client)
    
    st.divider()
    
    # 向量库管理
    st.header("💾 向量库管理")
    db_name = st.text_input("索引名称", value="ehs_master_index")
    col_db1, col_db2 = st.columns(2)
    with col_db1:
        if st.button("保存索引"):
            path = st.session_state.vector_store.save_to_disk(db_name)
            st.success(f"已保存: {path}")
    with col_db2:
        if st.button("加载索引"):
            if st.session_state.vector_store.load_from_disk(db_name):
                st.success(f"已加载! ({len(st.session_state.vector_store.documents)} 片段)")
            else:
                st.error("索引文件不存在")

st.info(f"当前向量库状态: 包含 {len(st.session_state.vector_store.documents)} 个制度片段")

tab1, tab2, tab3 = st.tabs(["📂 本地文件上传", "☁️ WebDAV 远程库", "🚀 开始评估"])

with tab1:
    reg_files_local = st.file_uploader("上传法规 (docx/zip)", type=['docx', 'zip'], accept_multiple_files=True, key="reg_local")
    policy_files_local = st.file_uploader("上传制度 (docx/xlsx/zip)", type=['docx', 'xlsx', 'zip'], accept_multiple_files=True, key="pol_local")
    
    if st.button("📥 将本地制度加入向量库"):
        if policy_files_local:
            corpus = []
            for name, content in process_uploaded_files(policy_files_local):
                text = extract_text_from_content(name, content)
                if text: corpus.append({'name': name, 'content': text})
            st.session_state.vector_store.add_documents(corpus)
            st.success("入库完成！请点击侧边栏保存索引。")

with tab2:
    st.markdown("### 连接到 WebDAV 服务器 (如 Nextcloud/坚果云)")
    webdav_url = st.text_input("WebDAV URL", help="e.g. https://dav.jianguoyun.com/dav/")
    webdav_user = st.text_input("Username")
    webdav_pass = st.text_input("Password", type="password")
    
    if st.button("🔗 连接并获取文件列表"):
        try:
            options = {'webdav_hostname': webdav_url, 'webdav_login': webdav_user, 'webdav_password': webdav_pass}
            wd_client = WebDavClient(options)
            files = wd_client.list() # List root
            st.session_state.webdav_files = [f for f in files if f.endswith(('.docx', '.zip', '.xlsx'))]
            st.session_state.wd_client = wd_client
            st.success(f"成功连接！发现 {len(st.session_state.webdav_files)} 个支持的文件。")
        except Exception as e:
            st.error(f"连接失败: {e}")

    if 'webdav_files' in st.session_state:
        selected_files = st.multiselect("选择要分析的法规/制度文件", st.session_state.webdav_files)
        file_type = st.radio("这些文件是:", ["制度 (加入向量库)", "法规 (用于分析)"])
        
        if st.button("⬇️ 下载并处理选定文件"):
            downloaded_corpus = []
            for fname in selected_files:
                try:
                    # WebDAV download to memory
                    with st.spinner(f"正在下载 {fname}..."):
                        # webdavclient3 download_from returns None, writes to file. We need bytes.
                        # Using buffer
                        buff = io.BytesIO()
                        st.session_state.wd_client.download_from(fname, buff)
                        buff.seek(0)
                        content = buff.read()
                        
                        text = extract_text_from_content(fname, content)
                        if text: downloaded_corpus.append({'name': fname, 'content': text})
                except Exception as e:
                    st.error(f"下载 {fname} 失败: {e}")
            
            if file_type == "制度 (加入向量库)":
                st.session_state.vector_store.add_documents(downloaded_corpus)
                st.success("WebDAV 制度文件已入库！")
            else:
                st.session_state.webdav_reg_corpus = downloaded_corpus
                st.success("WebDAV 法规文件已准备就绪！")

with tab3:
    st.subheader("执行合规性分析")
    st.markdown("数据源: **已加载的向量库** (制度) vs **上传/选定的法规文件**")
    
    if st.button("🚀 开始专家级评估", type="primary"):
        # 准备法规
        reg_corpus = []
        if reg_files_local:
            for name, content in process_uploaded_files(reg_files_local):
                text = extract_text_from_content(name, content)
                reg_corpus.append({'name': name, 'content': text})
        
        if 'webdav_reg_corpus' in st.session_state:
            reg_corpus.extend(st.session_state.webdav_reg_corpus)
            
        if not reg_corpus:
            st.error("请先上传或选择法规文件！")
            st.stop()
            
        if len(st.session_state.vector_store.documents) == 0:
            st.error("向量库为空！请先上传制度文件并入库。" )
            st.stop()
            
        # 解析法规
        all_clauses = []
        for doc in reg_corpus:
            clauses = parse_regulation_clauses(doc['content'])
            for i, c in enumerate(clauses):
                c['source_file'] = doc['name']
                c['序号'] = i + 1
                all_clauses.append(c)
                
        st.info(f"共识别出 {len(all_clauses)} 条法规条款，开始分析...")
        
        # 并发执行
        results_list = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        completed = 0
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_clause = {executor.submit(evaluate_single_clause, clause, st.session_state.vector_store, client): clause for clause in all_clauses}
            for future in as_completed(future_to_clause):
                res = future.result()
                res['法规文件'] = future_to_clause[future]['source_file']
                res['序号'] = future_to_clause[future]['序号']
                results_list.append(res)
                completed += 1
                progress_bar.progress(completed / len(all_clauses))
                status_text.text(f"分析进度: {completed}/{len(all_clauses)}")
                
        st.success("分析完成！")
        results_list.sort(key=lambda x: x['序号'])
        st.session_state.results = pd.DataFrame(results_list)

    if st.session_state.results is not None:
        df = st.session_state.results
        summary_stats = {
            "total": len(df),
            "compliant": len(df[df['评价结论'].str.contains("完全符合")]) ,
            "partial": len(df[df['评价结论'].str.contains("部分")]) ,
            "non_compliant": len(df[df['评价结论'].str.contains("缺失|不符合")])
        }
        
        st.dataframe(df)
        word_file = generate_word_report(df, summary_stats)
        st.download_button("📥 下载 Word 报告", word_file, "EHS_Report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")