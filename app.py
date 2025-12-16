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
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
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
            # 基础预判，实际由LLM在思维链中最终决定
            clauses.append({"条款号": title, "法规正文": title + " " + content, "适用性": applicability})
    return clauses

def evaluate_single_clause(clause, vector_store, llm_client):
    """
    基于 EHS 专家思维链的深度评估
    """
    row = {
        "条款号": clause['条款号'],
        "法规正文": clause['法规正文'],
        "评价结论": "❌缺失/不符合",
        "差距分析": "未检索到相关制度",
        "改进建议": "请补充相关管理规定",
        "支撑证据": "无",
        "匹配度": 0.0
    }

    # 1. 混合检索
    search_results = vector_store.search(clause['法规正文'], top_k=3)
    
    # 无论是否检索到，都必须交给 LLM 判断（尤其是判断是否适用）
    # 如果没检索到，LLM 会基于“无相关制度”进行判定
    
    top_score = search_results[0]['score'] if search_results else 0
    row['匹配度'] = top_score
        
    evidence_text = ""
    if search_results:
        for i, res in enumerate(search_results):
            evidence_text += f"参考制度片段 {i+1} (来源: {res['source']}):\n{res['content'][:800]}\n---\n"
    else:
        evidence_text = "未检索到任何相关的企业内部制度文档。"
    
    # 2. 构造专家 Prompt
    system_prompt = """你是一名具有20年经验的EHS管理专家，精通中国EHS法规标准，擅长仓储物流场景。
    请对给定的法规条款进行合规性评价。严格执行以下思维链：
    1. 解读：理解条款核心要求（人机料法环），判定是否适用于物流仓储企业。如果不适用，直接标记“不适用”。
    2. 比对：对比法规要求与提供的企业制度片段。是否覆盖所有要素？针对物流场景是否具体可执行？
    3. 判定：给出定性结论。
    
    请保持客观、犀利、直接。"""
    
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
        "gap_analysis": "150字以内。若不适用填'不适用'。若适用，分析具体缺了什么（如责任人、频次、物流特定措施）或说明通过哪几条实现了合规。",
        "improvement_suggestion": "若完全符合填'无'。否则结合物流仓储特点给出1-2条具体建议。",
        "evidence_summary": "列出最匹配的制度名称及关键句摘要（若无则填'未检索到相关制度'）"
    }}
    """
    
    try:
        result = llm_client.chat_completion(system_prompt, user_prompt)
        
        # 解析结果
        status = result.get('compliance_status', '缺失/不符合')
        # 统一状态图标
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
    """生成 Word 格式的合规性评价报告"""
    doc = Document()
    
    # 标题
    title = doc.add_heading('EHS法规合规性评价报告', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph(f"评价日期: {time.strftime('%Y-%m-%d')}")
    
    # 第一部分：总体评价
    doc.add_heading('第一部分：总体评价', level=1)
    
    # 统计数据
    p = doc.add_paragraph()
    p.add_run(f"本次共分析法规条款 {summary_stats['total']} 条.\n")
    p.add_run(f"✅ 完全符合: {summary_stats['compliant']} 条\n")
    p.add_run(f"⚠️ 部分符合/需完善: {summary_stats['partial']} 条\n")
    p.add_run(f"❌ 缺失/不符合: {summary_stats['non_compliant']} 条\n")
    p.add_run(f"❗ 不适用: {summary_stats['na']} 条")
    
    # 专家综述 (模拟)
    doc.add_heading('专家综述与建议', level=2)
    overall_conclusion = "总体来看，企业建立了基本的EHS管理框架。但在物流仓储特定场景的落地执行细节上（如现场作业管控、隐患排查频次）仍有待完善。建议重点关注“缺失”和“部分符合”的条款，结合改进建议尽快落实整改。"
    if summary_stats['compliance_rate'] > 85:
        overall_conclusion = "企业EHS管理制度体系较为完备，与法规要求匹配度较高。建议继续保持，并关注新法规的发布与更新。"
    
    doc.add_paragraph(overall_conclusion)
    
    # 第二部分：详细评价矩阵
    doc.add_heading('第二部分：详细合规性评价矩阵', level=1)
    
    # 创建表格
    table = doc.add_table(rows=1, cols=6)
    table.style = 'Table Grid'
    
    # 表头
    hdr_cells = table.rows[0].cells
    headers = ["序号", "法规条款", "评价结论", "差距分析与论据", "改进建议", "支撑证据"]
    for i, header in enumerate(headers):
        hdr_cells[i].text = header
        hdr_cells[i].paragraphs[0].runs[0].font.bold = True
    
    # 填充数据
    for idx, row in df_results.iterrows():
        row_cells = table.add_row().cells
        row_cells[0].text = str(row.get('序号', idx + 1))
        row_cells[1].text = str(row.get('法规正文', ''))
        row_cells[2].text = str(row.get('评价结论', ''))
        row_cells[3].text = str(row.get('差距分析', ''))
        row_cells[4].text = str(row.get('改进建议', ''))
        row_cells[5].text = str(row.get('支撑证据', ''))
        
    # 保存到内存
    f = io.BytesIO()
    doc.save(f)
    f.seek(0)
    return f

# ====================
# Streamlit UI
# ====================

st.set_page_config(page_title="EHS专家合规系统 (Expert Edition)", layout="wide")

if 'results' not in st.session_state: st.session_state.results = None

st.title("🛡️ EHS法规合规性智能评价系统 (Expert Edition)")
st.markdown("👨‍🏫 **专家模式**: 基于20年EHS经验的深度思维链分析，自动生成 Gap Analysis 与改进建议。")

with st.sidebar:
    st.header("1. API 配置")
    llm_base_url = st.text_input("API Base URL", value="https://generativelanguage.googleapis.com/v1beta/openai", help="例如 https://api.openai.com/v1")
    llm_api_key = st.text_input("API Key", type="password")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        llm_model_name = st.text_input("Chat Model", value="gemini-2.0-flash")
    with col_m2:
        embedding_model_name = st.text_input("Embedding Model", value="text-embedding-004")

    if st.button("🔌 测试 API 连接", use_container_width=True):
        if not llm_api_key:
            st.error("请先填写 API Key")
        else:
            with st.spinner("正在测试连接..."):
                test_config = {"base_url": llm_base_url, "api_key": llm_api_key, "model": llm_model_name, "embedding_model": embedding_model_name}
                client = LLMClient(test_config)
                res = client.test_connection()
                if res['chat']: st.success(f"✅ Chat: {llm_model_name} OK")
                else: st.error("❌ Chat Failed")
                if res['embedding']: st.success(f"✅ Embedding: {embedding_model_name} OK")
                else: st.error("❌ Embedding Failed")

    st.divider()
    st.header("2. 文件上传")
    reg_files = st.file_uploader("上传法规 (docx/zip)", type=['docx', 'zip'], accept_multiple_files=True, key="reg")
    policy_files = st.file_uploader("上传制度 (docx/xlsx/zip)", type=['docx', 'xlsx', 'zip'], accept_multiple_files=True, key="pol")

if st.button("🚀 开始专家级评估", type="primary"):
    if not (reg_files and policy_files and llm_api_key):
        st.error("请确保文件已上传且 API Key 已填写。" )
    else:
        llm_config = {"base_url": llm_base_url, "api_key": llm_api_key, "model": llm_model_name, "embedding_model": embedding_model_name}
        client = LLMClient(llm_config)
        vector_store = VectorStore()
        vector_store.set_client(client)
        
        # 1. 构建制度库
        policy_corpus = []
        for name, content in process_uploaded_files(policy_files):
            text = extract_text_from_content(name, content)
            if text and len(text.strip()) > 0: policy_corpus.append({'name': name, 'content': text})
            
        if not policy_corpus:
            st.error("有效制度内容为空。" )
            st.stop()
            
        vector_store.add_documents(policy_corpus)
        
        # 2. 解析法规
        all_clauses = []
        for name, content in process_uploaded_files(reg_files):
            text = extract_text_from_content(name, content)
            clauses = parse_regulation_clauses(text)
            for i, c in enumerate(clauses):
                c['source_file'] = name 
                c['序号'] = i + 1
                all_clauses.append(c)
        
        if not all_clauses:
             st.error("未解析出任何法规条款。" )
             st.stop()

        st.info(f"共识别出 {len(all_clauses)} 条法规条款，正在执行专家级分析 (Deep Analysis)...")
        
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
                    res['序号'] = future_to_clause[future]['序号']
                    results_list.append(res)
                except Exception as exc: st.warning(f"分析异常: {exc}")
                completed_tasks += 1
                progress_bar.progress(completed_tasks / total_tasks)
                status_text.text(f"已完成: {completed_tasks}/{total_tasks} ...")
                
        st.success("分析完成！")
        # 按序号排序
        results_list.sort(key=lambda x: x['序号'])
        st.session_state.results = pd.DataFrame(results_list)

if st.session_state.results is not None:
    df = st.session_state.results
    
    st.divider()
    st.subheader("📊 专家评估看板")
    
    compliant_count = len(df[df['评价结论'].str.contains("完全符合")])
    partial_count = len(df[df['评价结论'].str.contains("部分符合")])
    non_compliant_count = len(df[df['评价结论'].str.contains("缺失|不符合")])
    na_count = len(df[df['评价结论'].str.contains("不适用")])
    total_valid = len(df) - na_count
    compliance_rate = (compliant_count / total_valid * 100) if total_valid > 0 else 0
    
    summary_stats = {
        "total": len(df),
        "compliant": compliant_count,
        "partial": partial_count,
        "non_compliant": non_compliant_count,
        "na": na_count,
        "compliance_rate": compliance_rate
    }

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("完全符合", compliant_count)
    col2.metric("需完善", partial_count)
    col3.metric("缺失/不符合", non_compliant_count)
    col4.metric("合规率 (适用项)", f"{compliance_rate:.1f}%")
    
    # 结果展示表格
    st.dataframe(
        df, 
        column_config={
            "法规正文": st.column_config.TextColumn("法规条款", width="medium"),
            "差距分析": st.column_config.TextColumn("差距分析与论据", width="large"),
            "改进建议": st.column_config.TextColumn("改进建议", width="medium"),
            "支撑证据": st.column_config.TextColumn("证据摘要", width="medium")
        }, 
        use_container_width=True, 
        height=600
    )
    
    # 导出报告
    st.subheader("📥 报告导出")
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        word_file = generate_word_report(df, summary_stats)
        st.download_button(
            label="📄 下载专家合规性评价报告 (.docx)",
            data=word_file,
            file_name=f"EHS_Expert_Report_{time.strftime('%Y%m%d')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    
    with col_d2:
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📊 下载详细数据表 (.csv)", csv, "ehs_compliance_data.csv", "text/csv")
