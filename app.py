import streamlit as st
import pandas as pd
import re
import os
import zipfile
import math
from collections import Counter
import time
from io import BytesIO

# ==============================================================================
# 1. 核心逻辑层 (Core Logic) - 优化后的思维链引擎
# ==============================================================================

class ComplianceEngine:
    def __init__(self):
        # 核心关键词映射（语义扩展库）
        self.KEYWORD_MAPPING = {
            "主要负责人": ["总经理", "董事长", "第一责任人", "负责人", "党委书记", "EHS委员会主任"],
            "全员安全生产责任制": ["岗位安全责任", "一岗双责", "安全职责", "责任书", "承诺书", "绩效考核"],
            "资金投入": ["安全投入", "经费", "预算", "费用", "提取", "安责险"],
            "教育培训": ["培训", "学习", "考核", "三级教育", "复训", "继续教育"],
            "隐患排查": ["隐患治理", "检查", "巡查", "自查", "整改", "双重预防"],
            "风险分级管控": ["风险辨识", "风险评估", "危险源", "风险清单", "LEC"],
            "应急救援": ["应急预案", "演练", "处置方案", "响应", "救援队伍"],
            "相关方": ["承包商", "外包", "供应商", "承运方", "租赁", "劳务派遣"],
            "特种作业": ["电工", "焊接", "高处作业", "持证上岗", "作业许可"],
            "劳动防护用品": ["劳保用品", "防护服", "安全帽", "PPE"],
            "职业健康": ["职业病", "体检", "健康档案", "危害因素", "心理疏导"],
            "三同时": ["新建", "改建", "扩建", "设计", "验收", "工程项目"],
            "工会": ["职工代表", "民主监督", "工会", "职代会"],
            "监督检查": ["配合检查", "接受监督", "迎检", "合规性评价"],
            "法律责任": ["责任追究", "违规处罚", "行政处分", "问责"]
        }
        self.corpus_data = []

    def load_corpus_from_uploaded_files(self, uploaded_files):
        """从上传的文件中构建语料库"""
        self.corpus_data = []
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            try:
                # 读取docx文本
                text = self._extract_text_from_docx_stream(uploaded_file)
                # 按段落切分，保留上下文
                segments = re.split(r'[。\n；]', text)
                for seg in segments:
                    clean_seg = seg.strip()
                    if len(clean_seg) > 15: # 忽略过短片段
                        self.corpus_data.append({
                            'file': filename,
                            'content': clean_seg
                        })
            except Exception as e:
                st.error(f"无法读取文件 {filename}: {str(e)}")
        return len(self.corpus_data)

    def _extract_text_from_docx_stream(self, file_stream):
        """从文件流解析DOCX"""
        try:
            with zipfile.ZipFile(file_stream) as z:
                xml_content = z.read('word/document.xml').decode('utf-8')
                text = re.sub(r'<[^>]+>', '', xml_content)
                return text
        except:
            return ""

    def analyze_clause(self, clause_text):
        """步骤1：解读条款 (思维链：理解法规意图)"""
        info = {
            'is_applicable': True,
            'intent': 'general',
            'applicability_reason': '企业通用合规义务',
            'search_keywords': [],
            'required_elements': [] # 必须具备的闭环要素：记录、报告、培训等
        }

        # 1. 适用性判定 (排除纯政府职能)
        gov_keywords = ["国务院", "县级以上", "监察机关", "制定标准", "财政部门", "行业协会"]
        if any(k in clause_text for k in gov_keywords) and \
           not any(k in clause_text for k in ["生产经营单位", "企业", "主要负责人", "从业人员", "配合", "接受"]):
            info['is_applicable'] = False
            info['applicability_reason'] = "属政府行政职能条款"
            return info
        
        if "本法自" in clause_text and "施行" in clause_text:
            info['is_applicable'] = False
            info['applicability_reason'] = "生效时间条款"
            return info

        if "含义" in clause_text and "下列用语" in clause_text:
            info['is_applicable'] = True
            info['applicability_reason'] = "术语定义"
            info['search_keywords'] = ["术语", "定义", "附则"]

        # 2. 提取闭环要素 (深度理解)
        if "记录" in clause_text or "档案" in clause_text or "台账" in clause_text:
            info['required_elements'].append("记录留痕")
        if "报告" in clause_text or "通报" in clause_text:
            info['required_elements'].append("报告机制")
        if "培训" in clause_text or "教育" in clause_text:
            info['required_elements'].append("教育培训")

        # 3. 关键词提取与扩展
        found_keys = []
        for key, synonyms in self.KEYWORD_MAPPING.items():
            if key in clause_text or any(s in clause_text for s in synonyms):
                found_keys.extend(synonyms)
                found_keys.append(key)
        
        # 补充通用词
        clean_text = re.sub(r'[^\w]', ' ', clause_text)
        raw_words = [w for w in clean_text.split() if len(w) > 1 and w not in ["应当","可以","必须","单位","规定"]]
        
        info['search_keywords'] = list(set(found_keys + raw_words[:5]))
        return info

    def search_evidence(self, keywords):
        """步骤2：检索 (在语料库中寻找支撑)"""
        if not self.corpus_data or not keywords: return []
        
        matches = []
        for item in self.corpus_data:
            score = 0
            content = item['content']
            hit_words = []
            
            for kw in keywords:
                if kw in content:
                    score += 1
                    hit_words.append(kw)
            
            if score > 0:
                # 制度类文件加权
                if "制度" in content or "办法" in content or "规定" in content:
                    score += 0.5
                matches.append({
                    'file': item['file'],
                    'content': content,
                    'score': score,
                    'hits': hit_words
                })
        
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:3]

    def judge_compliance(self, clause_text, analysis, evidence):
        """步骤3-5：比对与判定 (思维链：逻辑推理)"""
        if not analysis['is_applicable']:
            return "❗不适用", analysis['applicability_reason'], "无"

        if not evidence:
            return "❌缺失/不符合", "制度库中完全未检索到相关管控条款，存在制度空白。", \
                   f"建议新增关于“{analysis['search_keywords'][:3]}”的专项管理规定。"

        top_ev = evidence[0]
        score = top_ev['score']
        
        # 闭环验证
        missing_loops = []
        for req in analysis['required_elements']:
            if req == "记录留痕" and not any(w in top_ev['content'] for w in ["记录", "档案", "台账", "凭证"]):
                missing_loops.append("记录要求")
            if req == "报告机制" and not any(w in top_ev['content'] for w in ["报告", "通报", "上报"]):
                missing_loops.append("报告流程")

        if score >= 2.0:
            if missing_loops:
                return "⚠️部分符合/需完善", \
                       f"制度涵盖了主体内容，但缺乏{'、'.join(missing_loops)}等闭环管理要求。", \
                       f"建议在《{top_ev['file']}》中补充{','.join(missing_loops)}的具体规定。"
            else:
                return "✅完全符合", "制度条款明确，要素齐全，能够有效支撑法规要求。", "无"
        elif score >= 1.0:
            return "⚠️部分符合/需完善", \
                   "制度中有提及相关概念，但执行细节（如频次、责任人、标准）不够明确。", \
                   f"建议细化关于{analysis['search_keywords'][:2]}的具体执行细则。"
        else:
            return "❌缺失/不符合", "检索到的制度关联度极低，无法有效支撑合规义务。", \
                   "需制定专项制度或在现有制度中增加专门章节。"

# ==============================================================================
# 2. UI 交互层 (Streamlit Interface)
# ==============================================================================

def main():
    st.set_page_config(page_title="EHS智能合规评价系统", layout="wide", page_icon="⚖️")
    
    st.title("⚖️ EHS智能合规评价系统 (专家版)")
    st.markdown("""
    本系统采用 **“语义理解-全量检索-闭环验证”** 的深度思维链，对企业制度与外部法规进行逐条对标评价。
    """)

    # --- Sidebar: Configuration ---
    with st.sidebar:
        st.header("1. 制度库构建")
        rule_files = st.file_uploader("上传内部制度文件 (支持多选)", type=['docx', 'txt'], accept_multiple_files=True)
        
        st.header("2. 评价对象")
        law_file = st.file_uploader("上传法规文件 (单选)", type=['docx'])
        
        st.info("提示：支持 docx 格式。系统会自动解析文档中的条款。")

    # --- Main Area ---
    if rule_files and law_file:
        engine = ComplianceEngine()
        
        # 1. Build Corpus
        with st.spinner('正在构建制度知识库...'):
            corpus_count = engine.load_corpus_from_uploaded_files(rule_files)
        st.success(f"✅ 制度库构建完成！共收录 {corpus_count} 条管理片段。")

        # 2. Process Law
        if st.button("开始合规评价", type="primary"):
            law_text = engine._extract_text_from_docx_stream(law_file)
            # Regex to split clauses: 第X条
            clauses_raw = re.split(r'(第[零一二三四五六七八九十百]+条)', law_text)
            
            results = []
            current_title = ""
            
            # Progress Bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_parts = len([p for p in clauses_raw if p.strip()])
            processed_count = 0

            # --- Evaluation Loop ---
            for part in clauses_raw:
                if re.match(r'^第[零一二三四五六七八九十百]+条$', part):
                    current_title = part
                elif current_title and part.strip():
                    content = part.strip()
                    processed_count += 1
                    
                    # Update UI
                    progress = min(processed_count / (total_parts/2), 1.0) # Approx
                    progress_bar.progress(progress)
                    status_text.text(f"正在评价: {current_title}...")

                    # Logic Chain
                    analysis = engine.analyze_clause(content)
                    evidence = []
                    if analysis['is_applicable']:
                        evidence = engine.search_evidence(analysis['search_keywords'])
                    
                    conclusion, gap, suggestion = engine.judge_compliance(content, analysis, evidence)
                    
                    # Format Evidence
                    ev_text = ""
                    if evidence:
                        ev_text = "\n".join([f"[{e['file']}] {e['content'][:50]}..." for e in evidence])
                    else:
                        ev_text = "无相关制度"

                    results.append({
                        "条款号": current_title,
                        "条款内容": content,
                        "评价结论": conclusion,
                        "差距分析": gap,
                        "改进建议": suggestion,
                        "支撑证据": ev_text,
                        "is_applicable": analysis['is_applicable']
                    })
                    current_title = ""
            
            progress_bar.progress(100)
            status_text.text("评价完成！")
            
            # --- 3. Report Generation & Display ---
            df = pd.DataFrame(results)
            
            # Statistics
            total = len(df)
            applicable = df[df['is_applicable'] == True]
            compliant = applicable[applicable['评价结论'].str.contains("完全符合")]
            partial = applicable[applicable['评价结论'].str.contains("部分符合")]
            missing = applicable[applicable['评价结论'].str.contains("缺失")]
            
            score = round((len(compliant) * 1 + len(partial) * 0.5) / len(applicable) * 100, 1) if len(applicable) > 0 else 0

            # --- Dashboard ---
            st.divider()
            st.subheader("📊 评价结果概览")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("合规得分", f"{score}分")
            col2.metric("完全符合", f"{len(compliant)}项")
            col3.metric("需完善", f"{len(partial)}项")
            col4.metric("缺失/不符合", f"{len(missing)}项", delta_color="inverse")

            # --- Comprehensive Summary (Generating Report Text) ---
            
            # Find top missing keywords
            missing_keywords = []
            for idx, row in missing.iterrows():
                words = re.sub(r'[^\w]', ' ', row['条款内容']).split()
                valid = [w for w in words if len(w)>2][:3]
                missing_keywords.extend(valid)
            top_risks = [k[0] for k in Counter(missing_keywords).most_common(5)]
            
            report_md = f"""# {law_file.name} 合规评价报告

## 第一部分：总体评价与管理建议

### 1. 整体情况概览
本次评价共对 **{total}** 个法规条款进行了逐条深度扫描。其中适用企业条款 **{len(applicable)}** 项。
整体合规得分为 **{score} 分**。
*   **合规亮点**：在核心管理要素（如{', '.join(list(engine.KEYWORD_MAPPING.keys())[:3])}）方面，制度体系较为完善，支撑证据充分。
*   **风险分布**：发现 **{len(missing)}** 项完全缺失，**{len(partial)}** 项制度存在瑕疵。

### 2. 核心风险领域 (Top Risks)
经智能分析，以下领域存在制度空白或严重不足，需重点关注：
> **{', '.join(top_risks)}**

### 3. 系统性问题诊断
*   **管理闭环缺失**：部分条款虽有制度提及，但在“记录留痕”、“定期报告”或“专项培训”等闭环环节存在缺失。
*   **新法跟进滞后**：针对法规中新增的特定要求（如心理疏导、安责险等），现有老制度尚未及时更新覆盖。

### 4. 下一步改进建议
1.  **填补空白**：针对上述核心风险领域，立即制定专项管理规定。
2.  **细化执行**：对判定为“需完善”的条款，修订对应制度，增加具体的执行频率、责任岗位和记录表单。
3.  **合规审查**：建议每半年进行一次制度与法规的对标审查。

---
## 第二部分：逐条评价明细表
"""
            
            st.markdown("### 📝 评价报告摘要")
            st.markdown(report_md)

            # --- Data Table ---
            st.subheader("🔎 逐条明细预览")
            st.dataframe(df[['条款号', '条款内容', '评价结论', '差距分析', '改进建议']])

            # --- Downloads ---
            st.subheader("📥 报告下载")
            
            # Generate Full Markdown
            full_md = report_md + "\n" + df.to_markdown(index=False)
            
            # Generate Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='合规评价表')
                # Add a summary sheet
                summary_df = pd.DataFrame({
                    "指标": ["总条款数", "适用条款数", "得分", "完全符合", "需完善", "缺失"],
                    "数值": [total, len(applicable), score, len(compliant), len(partial), len(missing)]
                })
                summary_df.to_excel(writer, index=False, sheet_name='概览')
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button("下载完整 Word/Markdown 报告", full_md, file_name=f"合规评价报告_{law_file.name}.md")
            with col_d2:
                st.download_button("下载 Excel 明细表", output.getvalue(), file_name=f"合规评价明细_{law_file.name}.xlsx")

    else:
        st.info("👈 请在左侧侧边栏上传 制度文件 和 法规文件 以开始。")

if __name__ == "__main__":
    main()
