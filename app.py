import streamlit as st
import pandas as pd
import os
import re
import zipfile
import io
import time

# ====================
# 核心逻辑函数 (复用专家经验)
# ====================

def process_uploaded_files(uploaded_files):
    """
    生成器：处理上传的文件列表，自动解压zip包
    Yields: (filename, content_bytes)
    """
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.zip'):
            try:
                with zipfile.ZipFile(uploaded_file) as z:
                    for filename in z.namelist():
                        # 跳过文件夹和隐藏文件
                        if filename.endswith('/') or filename.startswith('__MACOSX') or filename.startswith('._'):
                            continue
                        if filename.endswith(('.docx', '.xlsx')):
                            with z.open(filename) as f:
                                yield filename, f.read()
            except Exception as e:
                st.error(f"解压文件 {uploaded_file.name} 失败: {str(e)}")
        else:
            yield uploaded_file.name, uploaded_file.getvalue()

def extract_text_from_content(filename, content):
    """从文件内容中提取纯文本"""
    text = ""
    try:
        if filename.endswith('.docx'):
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                xml = zf.read('word/document.xml').decode('utf-8')
                text = re.sub(r'<[^>]+>', '', xml)
        elif filename.endswith('.xlsx'):
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                if 'xl/sharedStrings.xml' in zf.namelist():
                    xml = zf.read('xl/sharedStrings.xml').decode('utf-8')
                    text = re.sub(r'<[^>]+>', '', xml)
    except Exception as e:
        st.error(f"解析文件 {filename} 失败: {str(e)}")
    return text

def parse_regulation_clauses(text):
    """将法规文本拆解为条款列表"""
    # 匹配 "第X条" 的模式，支持中文数字
    pattern = r'(第[零一二三四五六七八九十百]+条)'
    parts = re.split(pattern, text)
    
    clauses = []
    if len(parts) > 1:
        # parts[0] 是前言，parts[1]是"第一条", parts[2]是内容...
        for i in range(1, len(parts), 2):
            title = parts[i]
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            
            # 简单的适用性判断逻辑（排除纯政府职责）
            applicability = "适用"
            gov_keywords = ["国务院", "县级以上", "监察机关", "人民政府", "主管部门"]
            # 如果主要是在讲政府应该做什么，且没有提及“生产经营单位”
            if any(k in content[:20] for k in gov_keywords) and "生产经营单位" not in content[:50]:
                applicability = "不适用(政府职责)"
            
            full_text = title + " " + content
            clauses.append({
                "条款号": title,
                "法规正文": full_text,
                "适用性": applicability
            })
    return clauses

def calculate_match_score(clause_text, policy_text):
    """计算匹配度得分 (基于简单的关键词重叠)"""
    # 简单的分词：按标点符号分割
    keywords = re.split(r'[，。；：、“”]', clause_text)
    keywords = [k for k in keywords if len(k) > 2] # 仅保留有意义的词
    
    score = 0
    matched_words = []
    
    for k in keywords:
        if k in policy_text:
            score += len(k)
            matched_words.append(k)
            
    return score, list(set(matched_words))

def generate_markdown_report(summary_data, df_result):
    """生成 Markdown 格式的报告"""
    report = f"""# EHS法规合规性评价报告

**评价日期**: {time.strftime("%Y-%m-%d")}

## 第一部分：总体评价

**1. 评价概况**
*   **分析条款总数**: {summary_data['total']}
*   **完全符合条款数**: {summary_data['compliant']}
*   **部分符合/需完善条款数**: {summary_data['partial']}
*   **不适用/缺失条款数**: {summary_data['non_compliant']}
*   **总体合规率**: {summary_data['compliance_rate']:.1f}% (完全符合 + 部分符合)

**2. 评价结论综述**
本次评价针对上传的法规文件与企业内部制度进行了自动比对。
{ "总体合规情况良好。" if summary_data['compliance_rate'] > 80 else "存在一定合规风险，建议重点关注缺失和部分符合的条款。" }

---

## 第二部分：详细合规性评价矩阵

| 序号 | 法规文件 | 条款号 | 评价结论 | 支撑证据 |
| :--- | :--- | :--- | :--- | :--- |
"""
    for _, row in df_result.iterrows():
        # 清理换行符以免破坏表格格式
        evidence = str(row['支撑证据']).replace('\n', '<br>').replace('|', '\|')
        # 截断过长的证据
        if len(evidence) > 100:
            evidence = evidence[:100] + "..."
            
        report += f"| {row['序号']} | {row['法规文件']} | {row['条款号']} | {row['评价结论']} | {evidence} |\n"
        
    return report

def analyze_compliance(reg_files, policy_files, progress_bar, status_text):
    """执行合规性分析的主流程"""
    
    # 1. 预处理制度文件库
    status_text.text("正在构建制度知识库...")
    policy_corpus = []
    
    # 使用 process_uploaded_files 处理文件，可能包含解压后的多个文件
    processed_policies = list(process_uploaded_files(policy_files))
    total_policies = len(processed_policies)
    
    for idx, (p_name, p_content) in enumerate(processed_policies):
        p_text = extract_text_from_content(p_name, p_content)
        if p_text:
            policy_corpus.append({
                "name": p_name,
                "content": p_text
            })
        progress_bar.progress((idx + 1) / total_policies * 0.1) # 预处理占10%进度

    all_results = []
    
    # 2. 逐个分析法规文件
    processed_regs = list(process_uploaded_files(reg_files))
    
    for r_name, r_content in processed_regs:
        r_text = extract_text_from_content(r_name, r_content)
        clauses = parse_regulation_clauses(r_text)
        
        total_clauses = len(clauses)
        if total_clauses == 0:
            st.warning(f"文件 {r_name} 未识别到有效条款，请确认格式。")
            continue
            
        current_results = []
        
        for idx, clause in enumerate(clauses):
            # 更新进度条
            progress = 0.1 + ((idx + 1) / total_clauses * 0.9)
            progress_bar.progress(progress)
            status_text.text(f"正在分析 {r_name}: {clause['条款号']}...")
            
            row = {
                "序号": idx + 1,
                "法规文件": r_name,
                "条款号": clause['条款号'],
                "法规正文": clause['法规正文'],
                "评价结论": "❌缺失/不符合", # 默认
                "支撑证据": "未检索到相关制度",
                "匹配度": 0
            }
            
            if clause['适用性'] != "适用":
                row['评价结论'] = "❗不适用"
                row['支撑证据'] = "条款主体非企业"
            else:
                # 在制度库中寻找最佳匹配
                best_score = 0
                best_match = None
                best_keywords = []
                
                for policy in policy_corpus:
                    score, keywords = calculate_match_score(clause['法规正文'], policy['content'])
                    if score > best_score:
                        best_score = score
                        best_match = policy
                        best_keywords = keywords
                
                # 判定逻辑
                if best_score > 0:
                    row['匹配度'] = best_score
                    # 提取匹配片段
                    idx = best_match['content'].find(best_keywords[0])
                    start = max(0, idx - 20)
                    end = min(len(best_match['content']), idx + 100)
                    snippet = best_match['content'][start:end] + "..."
                    
                    row['支撑证据'] = f"[{best_match['name']}]\n相关内容: ...{snippet}"
                    
                    if best_score > 30: # 阈值可调
                        row['评价结论'] = "✅完全符合"
                    elif best_score > 10:
                        row['评价结论'] = "⚠️部分符合/需完善"
                
            current_results.append(row)
        
        all_results.extend(current_results)
        
    return pd.DataFrame(all_results)

# ====================
# Streamlit UI 界面
# ====================

st.set_page_config(page_title="EHS合规性智能评价助手", layout="wide")

st.title("🛡️ EHS法规合规性智能评价助手")
st.markdown("""
本工具用于自动比对 **外部法规** 与 **内部制度**，生成合规性评价矩阵。
请在左侧侧边栏上传相应的文件。
""")

# --- 侧边栏：文件上传 ---
with st.sidebar:
    st.header("📂 文件上传区")
    
    st.subheader("1. 上传法规文件 (标准)")
    reg_files = st.file_uploader("支持 .docx, .zip (如: 安全法.docx)", type=['docx', 'zip'], accept_multiple_files=True, key="reg")
    
    st.subheader("2. 上传制度文件 (依据)")
    policy_files = st.file_uploader("支持 .docx, .xlsx, .zip (如: 管理手册)", type=['docx', 'xlsx', 'zip'], accept_multiple_files=True, key="pol")
    
    st.info("提示：支持批量上传或ZIP压缩包。文件越多，分析时间越长，请耐心等待。")

# --- 主界面：分析控制与展示 ---

if reg_files and policy_files:
    if st.button("🚀 开始合规性匹配分析", type="primary"):
        # 进度条容器
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 执行分析
            df_result = analyze_compliance(reg_files, policy_files, progress_bar, status_text)
            
            status_text.text("✅ 分析完成！")
            progress_bar.progress(100)
            
            # 计算汇总数据
            total_clauses = len(df_result)
            compliant_count = len(df_result[df_result['评价结论'] == "✅完全符合"])
            partial_count = len(df_result[df_result['评价结论'] == "⚠️部分符合/需完善"])
            non_compliant_count = total_clauses - compliant_count - partial_count
            compliance_rate = ((compliant_count + partial_count) / total_clauses * 100) if total_clauses > 0 else 0
            
            summary_data = {
                "total": total_clauses,
                "compliant": compliant_count,
                "partial": partial_count,
                "non_compliant": non_compliant_count,
                "compliance_rate": compliance_rate
            }
            
            st.divider()
            
            # --- 第一部分：总体评价 ---
            st.header("第一部分：总体评价")
            
            # 1. 评价概况
            st.subheader("1. 评价概况")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("分析条款总数", total_clauses)
            with col2:
                st.metric("完全符合", compliant_count)
            with col3:
                st.metric("部分符合", partial_count)
            with col4:
                st.metric("总体合规率", f"{compliance_rate:.1f}%")
            
            # 2. 评价结论综述
            st.subheader("2. 评价结论综述")
            conclusion = "总体合规情况良好。" if compliance_rate > 80 else "存在一定合规风险，建议重点关注缺失和部分符合的条款。"
            st.info(f"本次评价针对上传的法规文件与企业内部制度进行了自动比对。\n{conclusion}")

            # --- 第二部分：详细评价矩阵 ---
            st.header("第二部分：详细合规性评价矩阵")
            
            # 增加筛选功能
            filter_status = st.multiselect(
                "筛选评价结论:",
                options=df_result['评价结论'].unique(),
                default=df_result['评价结论'].unique()
            )
            
            df_display = df_result[df_result['评价结论'].isin(filter_status)]
            st.dataframe(
                df_display, 
                use_container_width=True,
                height=600,
                column_config={
                    "法规正文": st.column_config.TextColumn("法规正文", width="medium"),
                    "支撑证据": st.column_config.TextColumn("支撑证据", width="large"),
                }
            )
            
            # 导出功能
            st.subheader("📥 导出报告")
            
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                # Markdown 报告下载
                md_report = generate_markdown_report(summary_data, df_result)
                st.download_button(
                    label="📄 下载评价报告 (Markdown/Word)",
                    data=md_report,
                    file_name=f"EHS合规性评价报告_{time.strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                )
            
            with col_d2:
                # CSV 下载
                csv = df_result.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📊 下载评价明细表 (CSV)",
                    data=csv,
                    file_name=f"EHS合规性评价明细_{time.strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
            
        except Exception as e:
            st.error(f"分析过程中发生错误: {str(e)}")
            st.exception(e)

else:
    st.info("👈 请先在左侧侧边栏上传至少一个法规文件和一个制度文件。")

st.divider()
st.caption("Powered by Gemini EHS Compliance Engine | 2025")
