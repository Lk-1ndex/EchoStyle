from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """风格记忆切片富元数据 (Rich Metadata)"""
    chunk_id: str
    source: str
    profile_id: Optional[str] = Field(None, description="所属文风档案 ID；None 表示无归属切片")
    position: str = Field("body", description="篇章位置: opening(开篇), body(主体论述), ending(结语收束)")
    position_pct: float = Field(0.5, description="在原文章中的相对位置百分比 (0.0-1.0)")
    function: str = Field("argument", description="功能分类: hook(吸引), argument(核心论点), example(生动案例), quote(警醒金句), conclusion(反思总结)")
    type: str = Field("argument", description="兼容旧版字段: hook, argument, quote, conclusion")
    topic: str = Field("通用", description="主题领域")
    emotion: str = Field("中性", description="情绪色彩")
    style: str = Field("分析", description="文体特色")
    char_length: int = 0


class StatisticalMetrics(BaseModel):
    """统计语言学特征模型 (扩展 STTR 与长短句破空比)"""
    total_chars: int = Field(0, description="总字符数")
    total_sentences: int = Field(0, description="总有效句子数")
    total_paragraphs: int = Field(0, description="总段落数")
    avg_sentence_length: float = Field(0.0, description="平均句长（字符/句）")
    sentence_length_std: float = Field(0.0, description="句长标准差（衡量节奏起伏与长短波长）")
    avg_paragraph_length: float = Field(0.0, description="平均段长（句子/段）")
    ttr: float = Field(0.0, description="原始词汇丰富度 Type-Token Ratio (0-1.0)")
    sttr: float = Field(0.0, description="标准化词汇丰富度 Standardized TTR (消除文本长度偏差)")
    unique_words_count: int = Field(0, description="独立不同词汇总量")
    short_sentence_ratio: float = Field(0.0, description="爆发力极短句占比 (<10字，如'别闹了。')")
    long_sentence_ratio: float = Field(0.0, description="复杂铺陈长句占比 (>40字)")
    punctuation_entropy: float = Field(0.0, description="标点符号使用多样性信息熵")
    punctuation_distribution: Dict[str, float] = Field(default_factory=dict, description="主要标点占比分布")
    transition_density: float = Field(0.0, description="转折与逻辑词密度（次/千字）")
    transition_words: Dict[str, int] = Field(default_factory=dict, description="高频出现的转折/连接词统计")
    rhythm_pattern: str = Field("", description="人类可读的句式节奏定性描述")


class TonePersona(BaseModel):
    """语气与人设特征"""
    perspective: str = Field(..., description="叙事视角，如第一人称'我'、'笔者'或第三人称")
    emotional_tone: str = Field(..., description="情感基调，如犀利幽默、冷静客观、温和治愈、批判反思等")
    persona_traits: List[str] = Field(default_factory=list, description="作者人设与个性标签")


class CadenceSyntax(BaseModel):
    """句式节奏与排版习惯"""
    sentence_style: str = Field(..., description="句子长短与节奏偏好")
    paragraph_habit: str = Field(..., description="段落长短与分段习惯")
    punctuation_habits: List[str] = Field(default_factory=list, description="标点符号特色")


class LexiconRhetoric(BaseModel):
    """用词偏好与修辞手法"""
    catchphrases: List[str] = Field(default_factory=list, description="标志性口头禅与高频词")
    metaphor_style: str = Field(..., description="比喻与类比风格")
    vocabulary_richness: str = Field(..., description="词汇风格")


class DiscourseArchitecture(BaseModel):
    """篇章逻辑与行文结构 (Discourse Style)"""
    opening_hook: str = Field(..., description="开篇习惯描述")
    body_progression: str = Field(..., description="论点推进逻辑描述")
    ending_style: str = Field(..., description="结尾模式描述")
    opening_pattern: str = Field("narrative_or_paradox", description="开篇模式: narrative_hook(故事), paradox_hook(反常识设问), quote_hook(金句直击), opinion_hook(暴论)")
    progression_pattern: str = Field("inductive", description="论证推进流: inductive(个案->本质->价值观), dialectical(破常规谬误->立新见解), narrative_interwoven(叙议交织)")
    ending_pattern: str = Field("aphorism_or_question", description="收尾模式: aphorism(金句警策), open_question(设问留白), call_to_action(呼吁行动)")



class AntiPatterns(BaseModel):
    """负向过滤规则与去 AI 味禁令"""
    forbidden_words: List[str] = Field(
        default_factory=lambda: ["总而言之", "不可否认", "值得一提的是", "宛如", "综上所述", "显而易见", "深入探讨"],
        description="原作者文章中绝对不出现的陈词滥调与典型 AI 套话"
    )
    forbidden_structures: List[str] = Field(
        default_factory=lambda: ["首先...其次...最后...三段论模板", "然而...但是...生硬转折"],
        description="禁止使用的死板句式模板"
    )


class StyleProfile(BaseModel):
    """文风指纹档案 (质性特征)"""
    name: str = Field("默认文风", description="文风档案名称")
    description: str = Field("", description="文风简要概述")
    tone_persona: TonePersona
    cadence_syntax: CadenceSyntax
    lexicon_rhetoric: LexiconRhetoric
    discourse: DiscourseArchitecture
    anti_patterns: AntiPatterns
    exemplar_snippets: List[str] = Field(default_factory=list, description="典型金句与原汁原味的段落范例")

    def to_system_prompt(self, dynamic_few_shots: Optional[List[str]] = None) -> str:
        catchphrases_str = "、".join(self.lexicon_rhetoric.catchphrases) if self.lexicon_rhetoric.catchphrases else "无特殊限制"
        punctuations_str = "；".join(self.cadence_syntax.punctuation_habits) if self.cadence_syntax.punctuation_habits else "无"
        forbidden_words_str = "、".join(self.anti_patterns.forbidden_words)
        forbidden_structs_str = "；".join(self.anti_patterns.forbidden_structures)

        prompt = f"""# 角色与文风复刻指令

你现在必须 100% 深度复刻并化身为目标作者进行创作。

## 一、 叙事视角与情感基调
- 视角：{self.tone_persona.perspective}
- 情感基调：{self.tone_persona.emotional_tone}
- 作者人设特征：{"，".join(self.tone_persona.persona_traits)}

## 二、 句式节奏与排版习惯
- 句子节奏：{self.cadence_syntax.sentence_style}
- 分段习惯：{self.cadence_syntax.paragraph_habit}
- 标点偏好：{punctuations_str}

## 三、 用词偏好与修辞手法
- 标志性口癖/高频特征词：{catchphrases_str}
- 比喻风格：{self.lexicon_rhetoric.metaphor_style}
- 词汇质感：{self.lexicon_rhetoric.vocabulary_richness}

## 四、 篇章逻辑与行文结构
- 开篇切入：{self.discourse.opening_hook}
- 正文推进：{self.discourse.body_progression}
- 结尾方式：{self.discourse.ending_style}

## 五、 严禁触犯的去 AI 味红线
1. 严禁使用以下套话词汇：【{forbidden_words_str}】。
2. 绝对不可采用的生硬过渡：{forbidden_structs_str}。
3. 杜绝翻译腔与说教式空洞升华，行文必须充满人类真实呼吸感。
"""
        snippets = dynamic_few_shots if dynamic_few_shots is not None else self.exemplar_snippets
        if snippets:
            prompt += "\n## 六、 真实高光范例 (Golden Exemplars)\n"
            for i, snippet in enumerate(snippets, 1):
                prompt += f"### 范例 {i}：\n> {snippet.strip()}\n\n"

        return prompt


class DeepStyleProfile(BaseModel):
    """
    EchoStyle 深度文风建模档案：
    深度融合【统计语言学客观指标 (Stylometrics)】与【LLM 质性解构指纹】。
    """
    name: str = Field("深度文风档案", description="文风名称")
    profile_id: Optional[str] = Field(None, description="新建模时生成的持久化档案 ID；旧档案缺失此字段")
    qualitative: StyleProfile = Field(..., description="质性风格特征")
    quantitative: Optional[StatisticalMetrics] = Field(None, description="统计语言学客观特征")

    def to_system_prompt(self, dynamic_few_shots: Optional[List[str]] = None) -> str:
        base_prompt = self.qualitative.to_system_prompt(dynamic_few_shots=dynamic_few_shots)

        if self.quantitative and self.quantitative.total_sentences > 0:
            q = self.quantitative
            quant_injection = f"""
### 量化句法统计指标精密约束：
- **目标平均句长**：严格控制在【{q.avg_sentence_length:.1f} 字/句】左右。
- **句长离散度(波长起伏)**：标准差保持在【{q.sentence_length_std:.1f}】。
- **标准化词汇丰富度 (STTR)**：维持在【{q.sttr:.2f}】左右。
- **短句破空比率**：短句(<10字)比例约【{q.short_sentence_ratio*100:.1f}%】，保持爆发力与呼吸停顿。
- **标点多样性信息熵**：{q.punctuation_entropy:.2f}。
- **篇章展开路径约束**：必须遵循作者【{self.qualitative.discourse.opening_hook} -> {self.qualitative.discourse.body_progression} -> {self.qualitative.discourse.ending_style}】推进。
- **排斥 AI 模板**：严禁采用【定义概念 -> 罗列阐释 -> 综上总结】的平均主义八股架构！
- **节奏动态**：{q.rhythm_pattern}
"""
            if "## 三、" in base_prompt:
                base_prompt = base_prompt.replace("## 三、", f"{quant_injection}\n## 三、")
            else:
                base_prompt += f"\n{quant_injection}"

        return base_prompt


class EvaluationReport(BaseModel):
    """文风生成质量量化评估报告 (EchoEval 标准化加权公式)"""
    overall_score: float = Field(0.0, description="综合总分 (加权合成 0-100)")
    style_fidelity: float = Field(0.0, description="文风神似度 (权重 0.35)")
    llm_judge_score: float = Field(0.0, description="人类读者主观好感/评委分 (权重 0.25)")
    stylometric_similarity: float = Field(0.0, description="句式统计拟合度 (权重 0.20)")
    logic_depth: float = Field(0.0, description="逻辑论述深度 (权重 0.20)")
    anti_ai_score: float = Field(0.0, description="去 AI 味纯净度 (扣分基准)")
    ai_penalty: float = Field(0.0, description="违规八股惩罚分")
    radar_metrics: Dict[str, float] = Field(default_factory=dict, description="五维雷达图数据")
    detected_cliches: List[str] = Field(default_factory=list, description="捕获到的违规套话")
    feedback: str = Field("", description="审校与反思修改建议")
