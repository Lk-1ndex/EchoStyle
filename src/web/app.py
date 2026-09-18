import os
import tempfile
from pathlib import Path
import streamlit as st

from src.core.config import load_config
from src.core.models import DeepStyleProfile, StyleProfile, StatisticalMetrics, EvaluationReport
from src.memory.memory_manager import MemoryManager
from src.agents.coordinator import CoordinatorAgent
from src.agents.base import AgentContext

st.set_page_config(
    page_title="EchoStyle 2.0 - Multi-Agent 个人文风建模与创作系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 载入配置
if "config" not in st.session_state:
    st.session_state.config = load_config()

if "memory_mgr" not in st.session_state:
    st.session_state.memory_mgr = MemoryManager(
        embedding_config=st.session_state.config.embedding,
        llm_config=st.session_state.config.llm
    )

if "coordinator" not in st.session_state:
    st.session_state.coordinator = CoordinatorAgent(
        st.session_state.config,
        memory_manager=st.session_state.memory_mgr
    )

if "samples" not in st.session_state:
    st.session_state.samples = []

if "deep_profile" not in st.session_state:
    st.session_state.deep_profile = None

# ================= 侧边栏：大模型与 Agent 参数 =================
with st.sidebar:
    st.title("⚙️ 系统与 Agent 控制台")
    st.markdown("#### 1. 大模型 (LLM) 接口")

    api_key = st.text_input(
        "API Key",
        value=st.session_state.config.llm.api_key or os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="如 DeepSeek, OpenAI, Moonshot, SiliconFlow 等",
    )
    base_url = st.text_input(
        "Base URL",
        value=st.session_state.config.llm.base_url,
    )
    model_name = st.text_input(
        "Model 模型名",
        value=st.session_state.config.llm.model,
    )

    st.markdown("#### 2. Multi-Agent 反思参数")
    quality_threshold = st.slider(
        "Critic 质检合格分阈值",
        min_value=60.0,
        max_value=95.0,
        value=st.session_state.config.agent.quality_threshold,
        step=5.0,
        help="当文章得分低于该值或存在AI八股词时，自动触发反思重构",
    )
    max_reflections = st.slider(
        "最大反思迭代轮次",
        min_value=0,
        max_value=3,
        value=st.session_state.config.agent.max_reflections,
    )

    pdf_engine = st.selectbox(
        "PDF 解析路由策略",
        options=["auto", "markitdown", "mineru"],
        index=0,
        help="auto: 智能探测单双栏与扫描件；markitdown: 极速模式；mineru: 深度视觉模式",
    )

    # 同步状态
    st.session_state.config.llm.api_key = api_key
    st.session_state.config.llm.base_url = base_url
    st.session_state.config.llm.model = model_name
    st.session_state.config.agent.quality_threshold = quality_threshold
    st.session_state.config.agent.max_reflections = max_reflections
    st.session_state.config.extractor.pdf_engine = pdf_engine
    st.session_state.coordinator.critic_agent.quality_threshold = quality_threshold

    st.markdown("---")
    st.markdown("#### 💾 长期风格记忆库 (Style Memory)")
    active_profile_id = st.session_state.deep_profile.profile_id if st.session_state.deep_profile else None
    stats = st.session_state.memory_mgr.get_memory_stats(profile_id=active_profile_id) if active_profile_id else {"total_chunks": 0, "sources": []}
    st.write(f"• 已索引高光切片：**{stats['total_chunks']}** 个")
    st.write(f"• 历史文章来源：**{len(stats['sources'])}** 篇")
    if st.button("🗑️ 清空当前档案记忆库", disabled=not active_profile_id or stats["total_chunks"] == 0, help="仅重置当前档案的切片索引"):
        st.session_state.memory_mgr.clear_memory(profile_id=active_profile_id)
        st.success("当前档案记忆库已清空！")
        st.rerun()

# ================= 页面主标题 =================
st.title("🧬 EchoStyle 2.0: 基于 Multi-Agent 与风格记忆库的个人文风克隆系统")
st.caption("融合【统计语言学客观量化】+【长期风格记忆 (Style Memory RAG)】+【多智能体自审反思闭环】+【EchoEval 评测矩阵】")

tab1, tab2, tab3 = st.tabs([
    "📥 1. 样文感知与提取 (Extractor Agent)",
    "📊 2. 深度文风指纹与记忆 (Analyst Agent & Memory)",
    "✍️ 3. 智能创作与量化评估 (Writer & Critic Agents)"
])

# ================= Tab 1: 样文感知与提取 =================
with tab1:
    st.subheader("由 Extractor Agent 智能感知排版并高精抽取")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 来源 A：微信公众号文章链接")
        wechat_urls_text = st.text_area(
            "粘贴微信文章 URL（支持多行）",
            placeholder="https://mp.weixin.qq.com/s/xxxxxx",
            height=130,
        )

    with col2:
        st.markdown("##### 来源 B：本地 Word 或 PDF 文档")
        uploaded_files = st.file_uploader(
            "上传本地文档 (.docx, .pdf, .txt, .md)",
            type=["docx", "pdf", "txt", "md"],
            accept_multiple_files=True,
        )

    if st.button("🚀 启动 Extractor Agent 进行自适应解析", type="primary"):
        new_sources = []
        temp_paths = []
        uploaded_titles = {}
        ctx = AgentContext()

        try:
            # 收集输入
            if wechat_urls_text.strip():
                for line in wechat_urls_text.strip().split("\n"):
                    u = line.strip()
                    if u:
                        new_sources.append(u)

            if uploaded_files:
                for uf in uploaded_files:
                    suffix = Path(uf.name).suffix.lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        temp_paths.append(Path(tmp.name))
                        tmp.write(uf.getvalue())
                    new_sources.append(tmp.name)
                    uploaded_titles[Path(tmp.name).stem] = Path(uf.name).stem

            if not new_sources:
                st.warning("请至少提供一个公众号链接或上传一个文件！")
            else:
                with st.spinner("Extractor Agent 正在分析排版布局、去噪清洗并修补断行..."):
                    articles = st.session_state.coordinator.extract_sources(new_sources, state=ctx)
                    for article in articles:
                        article["title"] = uploaded_titles.get(article["title"], article["title"])
                    st.session_state.samples.extend(articles)
                    st.success(f"成功提取并清洗 {len(articles)} 篇样文！")
        except Exception as e:
            st.error(f"提取失败: {e}")
        finally:
            for temp_path in temp_paths:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError as e:
                    st.warning(f"上传临时文件清理失败: {e}")

        # 显示日志
        if ctx.execution_logs:
            with st.expander("🔍 查看 Extractor Agent 排版感知决策日志"):
                for log in ctx.execution_logs:
                    st.write(log)

    if st.session_state.samples:
        st.markdown("---")
        st.markdown(f"#### 已加载样文列表（共 {len(st.session_state.samples)} 篇）")
        for idx, s in enumerate(st.session_state.samples):
            with st.expander(f"📄 [{s.get('engine_used', 'text')}] {s['title']} ({s['char_count']} 字)"):
                st.text_area("清洗后正文", value=s["content"], height=180, key=f"s_{idx}")

# ================= Tab 2: 深度文风指纹与记忆 =================
with tab2:
    st.subheader("统计语言学客观量化 + LLM 质性特征 + 长期记忆库")

    if not st.session_state.samples:
        st.warning("请先在第一步中添加至少 1~3 篇样文！")
    else:
        profile_name = st.text_input("文风档案命名", value="我的深度文风档案")

        if st.button("🧬 启动 Analyst Agent 深度建模与记忆入库", type="primary"):
            if not st.session_state.config.llm.api_key:
                st.error("请在左侧边栏填写 API Key！")
            else:
                ctx = AgentContext()
                with st.spinner("Analyst Agent 正在计算句法统计指标、解构文风指纹并对语料向量化切片..."):
                    try:
                        deep_profile = st.session_state.coordinator.build_style(
                            st.session_state.samples,
                            profile_name=profile_name,
                            state=ctx
                        )
                        st.session_state.deep_profile = deep_profile
                        st.session_state.pop("final_art", None)
                        st.session_state.pop("eval_report", None)
                        st.session_state.pop("agent_ctx", None)
                        st.success("深度文风建模完成！语料已同步摄入长期记忆库。")
                    except Exception as e:
                        st.error(f"建模失败: {e}")

                if ctx.execution_logs:
                    with st.expander("🔍 查看 Analyst Agent 建模日志"):
                        for l in ctx.execution_logs:
                            st.write(l)

    # 展示深度档案
    if st.session_state.deep_profile:
        dp: DeepStyleProfile = st.session_state.deep_profile
        st.markdown("---")
        st.markdown(f"### 📋 深度文风建模档案：【{dp.name}】")

        # 1. 统计语言学客观指标
        if dp.quantitative:
            q = dp.quantitative
            st.markdown("#### 📐 1. 统计语言学客观指标 (Stylometrics)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("平均句长 (字符/句)", f"{q.avg_sentence_length:.1f}")
            m2.metric("句长标准差 (波长波动)", f"{q.sentence_length_std:.1f}")
            m3.metric("标点符号信息熵", f"{q.punctuation_entropy:.2f}")
            m4.metric("转折词密度 (次/千字)", f"{q.transition_density:.1f}")

            st.info(f"💡 **行文呼吸节奏评语**：{q.rhythm_pattern}")

            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.write("**主要标点占比分布**：")
                st.json(q.punctuation_distribution)
            with col_stat2:
                st.write("**标志性转折/逻辑词统计**：")
                st.json(q.transition_words)

        # 2. 质性风格特征
        st.markdown("#### 🎭 2. 质性语言指纹 (Linguistic DNA)")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**叙事视角**：{dp.qualitative.tone_persona.perspective}")
            st.write(f"**情感基调**：{dp.qualitative.tone_persona.emotional_tone}")
            st.write(f"**标志性口癖**：{'、'.join(dp.qualitative.lexicon_rhetoric.catchphrases) or '无'}")
            st.write(f"**比喻特色**：{dp.qualitative.lexicon_rhetoric.metaphor_style}")
        with c2:
            st.write(f"**句式节奏**：{dp.qualitative.cadence_syntax.sentence_style}")
            st.write(f"**开篇切入**：{dp.qualitative.discourse.opening_hook}")
            st.write(f"**正文推进**：{dp.qualitative.discourse.body_progression}")
            st.write(f"**结尾模式**：{dp.qualitative.discourse.ending_style}")

        st.warning(f"🚫 **去 AI 味禁令 (禁止出现的八股词)**：{'、'.join(dp.qualitative.anti_patterns.forbidden_words)}")

        # 3. 记忆库在线检索测试
        st.markdown("#### 🔍 3. 长期风格记忆库 (Style Memory) 动态检索测试")
        test_query = st.text_input("输入一个模拟创作主题，测试动态召回的历史片段", value="技术的本质与独立思考")
        if not dp.profile_id:
            st.warning("旧版文风档案缺少 profile_id，请重新建模后再检索或写作。")
        if st.button("执行检索测试", disabled=not dp.profile_id):
            matched_snippets = st.session_state.memory_mgr.retrieve_dynamic_few_shots(
                test_query, top_k=3, profile_id=dp.profile_id
            )
            if matched_snippets:
                for idx, snip in enumerate(matched_snippets, 1):
                    st.success(f"**动态召回高光段落 {idx}**：\n\n> {snip}")
            else:
                st.info("记忆库暂无匹配片段。")

# ================= Tab 3: 智能创作与量化评估 =================
with tab3:
    st.subheader("Writer Agent 创作 + Critic Agent 质检反思闭环")

    if not st.session_state.deep_profile or not st.session_state.deep_profile.profile_id:
        st.info("💡 请先在第二步重新建模深度文风档案！" if st.session_state.deep_profile else "💡 请先在第二步提炼深度文风档案！")
    else:
        topic = st.text_input("新文章主题", placeholder="例如：在被大模型充斥的世界，为什么独特的个人文风更稀缺？")
        key_points = st.text_area(
            "核心论点或大纲要点（可任意输入核心素材或几个论点）",
            placeholder="- 痛点：机器生成的内容越来越充斥千篇一律的‘八股翻译腔’\n- 本质：语言风格是人类长期思考与阅历沉淀的指纹，无法被平均化\n- 升华：做有温度的写作者，拒绝成为没有灵魂的套话复读机",
            height=110,
        )

        w1, w2 = st.columns(2)
        with w1:
            words = st.number_input("目标字数", min_value=300, max_value=8000, value=1500, step=100)
        with w2:
            audience = st.text_input("目标受众", value="热爱思考的创作者与科技读者")

        if st.button("✨ 启动 Multi-Agent 协作创作与闭环反思", type="primary"):
            if not topic.strip():
                st.error("请输入主题！")
            elif not st.session_state.config.llm.api_key:
                st.error("请配置 API Key！")
            else:
                st.session_state.pop("final_art", None)
                st.session_state.pop("eval_report", None)
                ctx = AgentContext()
                st.session_state.agent_ctx = ctx
                with st.spinner("Coordinator Agent 正在统筹 Writer 创作、向量记忆检索及 Critic 质检反思..."):
                    try:
                        final_art, report, ctx = st.session_state.coordinator.generate_article(
                            profile=st.session_state.deep_profile,
                            topic=topic,
                            key_points=key_points,
                            word_count=words,
                            target_audience=audience,
                            state=ctx
                        )
                        st.session_state.final_art = final_art
                        st.session_state.eval_report = report
                        st.session_state.agent_ctx = ctx
                    except Exception as e:
                        st.error(f"创作流程异常: {e}")

        # 展示 Agent 协作与反思日志
        if "agent_ctx" in st.session_state and st.session_state.agent_ctx.execution_logs:
            with st.expander("🤖 查看 Multi-Agent 协作调度与自省反思轨迹 (Audit Trail)", expanded=True):
                for log in st.session_state.agent_ctx.execution_logs:
                    if "未达到" in log or "触发" in log:
                        st.warning(log)
                    elif "优异" in log or "达成" in log:
                        st.success(log)
                    else:
                        st.write(log)

        # 展示成文与评测报告
        if "final_art" in st.session_state and "eval_report" in st.session_state:
            rep: EvaluationReport = st.session_state.eval_report

            st.markdown("---")
            st.markdown("### 🏆 生成质量综合评测报告 (EchoEval)")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("综合质量总分", f"{rep.overall_score:.1f} / 100")
            s2.metric("去 AI 味纯净度", f"{rep.anti_ai_score:.1f} / 100")
            s3.metric("句式统计拟合度", f"{rep.stylometric_similarity:.1f} / 100")
            s4.metric("LLM 专家仲裁分", f"{rep.llm_judge_score:.1f} / 100")

            if rep.radar_metrics:
                st.write("**五维雷达评估维度**：")
                st.json(rep.radar_metrics)

            if rep.detected_cliches:
                st.error(f"⚠️ 违规套话捕获：{', '.join(rep.detected_cliches)}")
            else:
                st.success("✅ 零违规套话：文章经受住了严格的去 AI 味八股审查！")

            st.info(f"📝 **总编辑审校评语**：{rep.feedback}")

            st.markdown("---")
            st.markdown("### 📄 终审成文 (Final Article)")
            st.markdown(st.session_state.final_art)

            st.download_button(
                label="📥 导出最终成文 Markdown",
                data=st.session_state.final_art,
                file_name=f"{topic[:20]}.md",
                mime="text/markdown",
            )
