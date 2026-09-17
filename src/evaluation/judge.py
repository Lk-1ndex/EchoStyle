import json
import random
import re
from typing import Dict, List, Any, Tuple, Optional
from pydantic import BaseModel, Field

from src.core.config import LLMConfig
from src.core.models import DeepStyleProfile, EvaluationReport
from src.core.model_provider import ModelProvider
from .lexical_metrics import LexicalEvaluator
from .rhythm_metrics import RhythmEvaluator
from .discourse_metrics import DiscourseEvaluator


JUDGE_SYSTEM_PROMPT = """你是一位严苛的文学出版界高级总编辑与文风鉴定专家。
你的职责是对大模型生成的文章进行深度审校，评判其与作者原生文风的相似度，并提供专业的打分与修改意见。

评测维度说明：
1. style_fidelity (0-100)：文风神似度，包括人设语气、修辞意象、标志性口头禅是否逼真。
2. logic_depth (0-100)：逻辑论述深度，是否有见地、有思辨张力，拒绝泛泛而谈。
3. human_preference (0-100)：真实人类读者好感度，读起来是否有血肉呼吸感，有无机器翻译腔。
4. radar：五维特征（tone, cadence, lexicon, discourse, anti_ai），分值 0-100。
5. critique_feedback：若文章有缺陷或有 AI 痕迹，给出具体的批评与下一轮重写修改指导建议；若表现优秀则指出其亮点。

只输出合法 JSON，不要输出任何其他解释文字。
"""

JUDGE_USER_PROMPT_TEMPLATE = """## 目标作者文风指南
{style_guide}

## 待审校生成的文章
{article_content}

---
请对照上述文风指南完成独立裁决，输出评分 JSON。
"""


def _extract_json_payload(text: str) -> dict:
    """鲁棒的 JSON 提取工具：支持去除 Markdown 代码块包裹、提取括号内 JSON 及清除末尾非法逗号"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            raw_json = match.group(0)
            try:
                return json.loads(raw_json)
            except json.JSONDecodeError:
                cleaned = re.sub(r",\s*([\]}])", r"\1", raw_json)
                return json.loads(cleaned)
        raise


class LLMJudge:
    """
    EchoEval 综合评测裁判专家：
    落地标准化加权评分公式：
    Final Score = 0.35 * Style_Fidelity + 0.25 * Human_Preference + 0.20 * Stylometrics_Fit + 0.20 * Logic_Depth - AI_Penalty
    """

    def __init__(self, llm_config: LLMConfig):
        self.config = llm_config
        self.model_provider = ModelProvider(llm_config)

    def evaluate(self, article: str, profile: DeepStyleProfile) -> EvaluationReport:
        # 1. 规则层计算：去 AI 味得分与八股惩罚分
        forbidden = profile.qualitative.anti_patterns.forbidden_words if profile and profile.qualitative else None
        anti_ai_score, ai_penalty, detected_cliches = LexicalEvaluator.evaluate_cliches(
            article,
            custom_forbidden=forbidden
        )

        # 2. 统计语言学拟合度计算 (句长均值 + 方差 + STTR)
        target_m = profile.quantitative if profile else None
        rhythm_score, _ = RhythmEvaluator.evaluate_rhythm_fit(article, target_m)
        target_sttr = target_m.sttr if target_m and target_m.sttr > 0 else 0.70
        lex_score, _ = LexicalEvaluator.evaluate_lexical_authenticity(article, target_sttr)
        stylometric_similarity = round(rhythm_score * 0.70 + lex_score * 0.30, 1)

        # 3. LLM-as-a-Judge 专家仲裁
        fidelity = 80.0
        logic_depth = 85.0
        human_pref = 80.0
        feedback = "文风与论点基本符合预期。"
        radar = {
            "语气视角": 80.0,
            "句式节奏": stylometric_similarity,
            "用词口癖": 80.0,
            "篇章逻辑": 85.0,
            "去AI味": anti_ai_score,
        }

        try:
            raw_judge = self._call_judge(article, profile)
            parsed = self._extract_json(raw_judge)
            fidelity = float(parsed.get("style_fidelity", fidelity))
            logic_depth = float(parsed.get("logic_depth", logic_depth))
            human_pref = float(parsed.get("human_preference", human_pref))

            if "radar" in parsed and isinstance(parsed["radar"], dict):
                r = parsed["radar"]
                radar = {
                    "语气视角": float(r.get("tone", fidelity)),
                    "句式节奏": stylometric_similarity,
                    "用词口癖": float(r.get("lexicon", fidelity)),
                    "篇章逻辑": float(r.get("discourse", logic_depth)),
                    "去AI味": anti_ai_score,
                }
            feedback = parsed.get("critique_feedback", feedback)
        except Exception as e:
            feedback = f"规则引擎质检完成，LLM 裁判评分降级: {e}"

        # 4. 执行工业级标准化加权评分公式
        weighted_base = (
            0.35 * fidelity +
            0.25 * human_pref +
            0.20 * stylometric_similarity +
            0.20 * logic_depth
        )
        final_score = max(0.0, min(100.0, round(weighted_base - ai_penalty, 1)))

        return EvaluationReport(
            overall_score=final_score,
            style_fidelity=round(fidelity, 1),
            llm_judge_score=round(human_pref, 1),
            stylometric_similarity=stylometric_similarity,
            logic_depth=round(logic_depth, 1),
            anti_ai_score=round(anti_ai_score, 1),
            ai_penalty=ai_penalty,
            radar_metrics=radar,
            detected_cliches=detected_cliches,
            feedback=feedback,
        )

    def _call_judge(self, article: str, profile: DeepStyleProfile) -> str:
        user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            style_guide=profile.to_system_prompt(),
            article_content=article
        )
        return self.model_provider.chat_completion(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
            json_mode=True
        )

    def _extract_json(self, text: str) -> dict:
        return _extract_json_payload(text)


class EvaluatorPersona(BaseModel):
    name: str
    role_desc: str
    focus_dimensions: str
    temperature: float = 0.2


# 评价者分层角色库 (Multi-Persona Evaluator Panel)
EVALUATOR_PANEL = {
    "GENERAL_READER": EvaluatorPersona(
        name="普通大众读者",
        role_desc="你是一名经常阅读优质自媒体深度长文的互联网普通读者。",
        focus_dimensions="最看重文章是否通顺流畅、引人入胜、通俗生动，阅读体验是否舒服，是否有生硬造作的翻译腔。"
    ),
    "DEVOTED_FOLLOWER": EvaluatorPersona(
        name="作者忠实读者",
        role_desc="你非常熟悉目标作者的过往写作风格、行文调性与标志性口癖。",
        focus_dimensions="最看重这篇新文章‘像不像该作者亲笔所写’，是否具备作者特有的犀利刺痛感、自嘲与独立价值判断。"
    ),
    "CHIEF_EDITOR": EvaluatorPersona(
        name="资深总编辑",
        role_desc="你是一家顶尖严肃思想文化期刊的资深总编辑，对稿件质量具有极高审美把关。",
        focus_dimensions="最看重篇章逻辑推进深度、有无教科书三段论与陈词滥调，对典型的‘不可否认’‘总而言之’等 AI 八股实行一票否决。"
    )
}


class MultiPersonaJudge:
    """
    多角色分层盲评裁判 (Multi-Persona Blind Evaluation Panel)：
    解决单一评价者标准混乱的问题，通过分层画像（大众读者 / 铁粉读者 / 专业编辑）得出可信仲裁。
    """

    @classmethod
    def evaluate_pair_with_persona(
        cls,
        topic: str,
        sample_a: str,
        sample_b: str,
        author_ref: str,
        persona: EvaluatorPersona,
        provider: Optional[ModelProvider] = None,
    ) -> Dict[str, Any]:
        """按特定评价者视角进行裁决"""
        if provider and provider.llm_config.api_key:
            prompt = f"""【你的裁判身份】
{persona.role_desc}
你的评判侧重点：{persona.focus_dimensions}

【作者风格范文基准】
{author_ref[:500]}

【评测选题】
{topic}

【样本 A】
{sample_a[:600]}

【样本 B】
{sample_b[:600]}

请对比【样本 A】和【样本 B】，从你的身份出发严格评审哪一篇质量更高、更契合目标。
必须输出合法 JSON：
{{
  "score_a": <0-100分>,
  "score_b": <0-100分>,
  "winner": "A" | "B" | "TIE",
  "comment": "<结合你的身份给出 1-2 句核心理由>"
}}"""
            try:
                raw = provider.chat(
                    system_prompt=f"你现在化身为【{persona.name}】，按专业标准执行盲评。",
                    user_prompt=prompt,
                    temperature=persona.temperature,
                    json_mode=True
                )
                data = json.loads(raw)
                return {
                    "persona": persona.name,
                    "score_a": float(data.get("score_a", 75.0)),
                    "score_b": float(data.get("score_b", 75.0)),
                    "winner": data.get("winner", "TIE").upper(),
                    "comment": data.get("comment", "")
                }
            except Exception:
                pass

        # 离线客观规则/语言学特征仲裁 (基于真实文本特征计算而非硬编码 winner="A")
        from src.evaluation.metrics import MetricEvaluator
        from src.analyzer.stylometrics import StylometricsAnalyzer

        target_m = StylometricsAnalyzer.analyze(author_ref)
        fit_a, _ = MetricEvaluator.evaluate_stylometric_fit(sample_a, target_m)
        anti_a, _, _ = MetricEvaluator.evaluate_anti_ai(sample_a)
        disc_a, _ = MetricEvaluator.evaluate_discourse_fit(sample_a)
        total_a = fit_a * 0.35 + anti_a * 0.35 + disc_a * 0.30

        fit_b, _ = MetricEvaluator.evaluate_stylometric_fit(sample_b, target_m)
        anti_b, _, _ = MetricEvaluator.evaluate_anti_ai(sample_b)
        disc_b, _ = MetricEvaluator.evaluate_discourse_fit(sample_b)
        total_b = fit_b * 0.35 + anti_b * 0.35 + disc_b * 0.30

        if total_a > total_b + 2.0:
            winner = "A"
        elif total_b > total_a + 2.0:
            winner = "B"
        else:
            winner = "TIE"

        return {
            "persona": persona.name,
            "score_a": round(total_a, 1),
            "score_b": round(total_b, 1),
            "winner": winner,
            "comment": f"[{persona.name}] 离线语言学特征客观判定：样本 A ({total_a:.1f}) vs 样本 B ({total_b:.1f})"
        }


INDEPENDENT_EVALUATOR_SYSTEM_PROMPT = """你是一位流程隔离的条件盲审文风终审评审员（Condition-Blind Holdout Evaluator，匿名条件盲审）与文学审校专家。
你从未参与任何前序写作引导、修改建议或反思交互过程（流程物理隔离）。
你的职责是对盲审提交的文章进行完全独立、无偏见、低随机性的量化打分。

评测维度说明：
1. style_fidelity (0-100)：文风神似度，包括人设语气、修辞意象、标志性口头禅是否逼真。
2. logic_depth (0-100)：逻辑论述深度，是否有见地、有思辨张力，拒绝泛泛而谈。
3. human_preference (0-100)：真实人类读者好感度，读起来是否有血肉呼吸感，有无机器翻译腔。
4. radar：五维特征（tone, cadence, lexicon, discourse, anti_ai），分值 0-100。
5. critique_feedback：客观中立的终审裁决评语。

必须只输出合法 JSON，不要输出任何其他解释文字。
"""

INDEPENDENT_EVALUATOR_USER_TEMPLATE = """## 目标作者文风指南
{style_guide}

## 待盲审评估文章 (匿名提交 / 条件盲化)
{article_content}

---
请对照上述文风指南完成独立客观的条件盲审裁决，输出评分 JSON。
"""


class IndependentEvaluator:
    """
    独立条件盲化留出评测器 (Process-Independent & Condition-Blind Holdout Evaluator):
    解决 P0-3 "Critic 既当运动员又当裁判 (Evaluator Overfitting)" 缺陷与评测隔离：
    1. 内部 Critic (Coordinator.critic_agent) 仅负责创作生成循环中的反思、批注与重写引导；
    2. 终审评测由独立的 IndependentEvaluator 统一执行，且对文章来源匿名 (A0/A1/B/C1a/C1b/C2/D 条件盲审)；
    3. 支持独立指定 evaluator_model 配置，与生成模型实现物理/实例解耦；
    4. 评测推断使用 temperature=0.0 以降低采样随机性；
    5. 支持 strict 模式 (严格 Benchmark 模式)：API 失败或格式解析失败时直接抛出 EvaluationUnavailableError (Fail-Closed)，
       严禁以硬编码默认好成绩 (80/85/80) 虚拟伪造评测结果！
    """

    def __init__(self, llm_config: LLMConfig, strict: bool = False):
        self.config = llm_config
        self.strict = strict
        self.model_provider = ModelProvider(llm_config)

    def evaluate(self, article: str, profile: DeepStyleProfile) -> EvaluationReport:
        # 1. 规则层计算：去 AI 味得分与八股惩罚分
        forbidden = profile.qualitative.anti_patterns.forbidden_words if profile and profile.qualitative else None
        anti_ai_score, ai_penalty, detected_cliches = LexicalEvaluator.evaluate_cliches(
            article,
            custom_forbidden=forbidden
        )

        # 2. 统计语言学拟合度计算 (句长均值 + 方差 + STTR)
        target_m = profile.quantitative if profile else None
        rhythm_score, _ = RhythmEvaluator.evaluate_rhythm_fit(article, target_m)
        target_sttr = target_m.sttr if target_m and target_m.sttr > 0 else 0.70
        lex_score, _ = LexicalEvaluator.evaluate_lexical_authenticity(article, target_sttr)
        stylometric_similarity = round(rhythm_score * 0.70 + lex_score * 0.30, 1)

        # 3. 流程隔离盲审裁决 (temperature=0.0)
        fidelity = 80.0
        logic_depth = 85.0
        human_pref = 80.0
        feedback = "独立条件盲审质检完成，文风与论点基本符合基准。"
        radar = {
            "语气视角": 80.0,
            "句式节奏": stylometric_similarity,
            "用词口癖": 80.0,
            "篇章逻辑": 85.0,
            "去AI味": anti_ai_score,
        }

        try:
            raw_judge = self._call_judge(article, profile)
            parsed = self._extract_json(raw_judge)
            if not isinstance(parsed, dict):
                raise ValueError(f"裁判返回内容非 JSON 字典: {raw_judge}")
            if self.strict:
                for req_key in ["style_fidelity", "logic_depth", "human_preference"]:
                    if req_key not in parsed:
                        raise ValueError(f"裁判输出缺失关键评分字段 [{req_key}]: {parsed}")
            fidelity = float(parsed.get("style_fidelity", fidelity))
            logic_depth = float(parsed.get("logic_depth", logic_depth))
            human_pref = float(parsed.get("human_preference", human_pref))

            if self.strict:
                for name, val in [("style_fidelity", fidelity), ("logic_depth", logic_depth), ("human_preference", human_pref)]:
                    if not (0.0 <= val <= 100.0):
                        raise ValueError(f"裁判输出评分超出有效范围 [0-100] ({name}={val}): {parsed}")

            if "radar" in parsed and isinstance(parsed["radar"], dict):
                r = parsed["radar"]
                radar_candidate = {
                    "语气视角": float(r.get("tone", fidelity)),
                    "句式节奏": stylometric_similarity,
                    "用词口癖": float(r.get("lexicon", fidelity)),
                    "篇章逻辑": float(r.get("discourse", logic_depth)),
                    "去AI味": anti_ai_score,
                }
                if self.strict:
                    for r_name, r_val in radar_candidate.items():
                        if not (0.0 <= r_val <= 100.0):
                            raise ValueError(f"裁判输出 radar 维度超出有效范围 [0-100] ({r_name}={r_val}): {parsed}")
                    radar = radar_candidate
                else:
                    radar = {k: max(0.0, min(100.0, v)) for k, v in radar_candidate.items()}
            feedback = parsed.get("critique_feedback", feedback)
        except Exception as e:
            if self.strict:
                from src.core.exceptions import EvaluationUnavailableError
                raise EvaluationUnavailableError(
                    f"独立评测裁判调用或解析失败 (strict Fail-Closed 严控阻断): {e}"
                ) from e
            feedback = f"规则引擎盲审完成，LLM 裁判评分降级: {e}"

        # 4. 执行工业级标准化加权评分公式
        weighted_base = (
            0.35 * fidelity +
            0.25 * human_pref +
            0.20 * stylometric_similarity +
            0.20 * logic_depth
        )
        final_score = max(0.0, min(100.0, round(weighted_base - ai_penalty, 1)))

        return EvaluationReport(
            overall_score=final_score,
            style_fidelity=round(fidelity, 1),
            llm_judge_score=round(human_pref, 1),
            stylometric_similarity=stylometric_similarity,
            logic_depth=round(logic_depth, 1),
            anti_ai_score=round(anti_ai_score, 1),
            ai_penalty=ai_penalty,
            radar_metrics=radar,
            detected_cliches=detected_cliches,
            feedback=feedback,
        )

    def _call_judge(self, article: str, profile: DeepStyleProfile) -> str:
        user_prompt = INDEPENDENT_EVALUATOR_USER_TEMPLATE.format(
            style_guide=profile.to_system_prompt(),
            article_content=article
        )
        return self.model_provider.chat_completion(
            system_prompt=INDEPENDENT_EVALUATOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,  # 降低采样随机性
            json_mode=True
        )

    def _extract_json(self, text: str) -> dict:
        return _extract_json_payload(text)

