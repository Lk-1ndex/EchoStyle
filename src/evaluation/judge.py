import json
import re
from typing import Dict, Any, Optional
from src.core.config import LLMConfig
from src.core.models import DeepStyleProfile, EvaluationReport
from src.core.model_provider import ModelProvider
from .metrics import MetricEvaluator


JUDGE_SYSTEM_PROMPT = """你是一名资深文学总编辑兼大模型内容评测仲裁员 (LLM Judge)。
你的职责是：对照目标作者的【文风指纹档案】，对刚刚生成的仿写文章进行严苛、客观的盲评审校打分，并指出瑕疵。

请从以下几个维度进行 0-100 分的量化打分，并严格输出 JSON 格式：
1. style_fidelity (文风神似度 0-100)：用词习惯、作者人设口吻、口癖与典型句式是否真正神似原作者？
2. logic_depth (论述与内容深度 0-100)：论点是否展开充分、论据是否有力、篇章递进是否自然？
3. human_preference (人类读者好感与呼吸感 0-100)：是否彻底摒弃了死板八股模板？读起来是否像真人手笔？
4. radar (五维雷达评分 0-100)：
   - tone (语气视角还原)
   - cadence (句式呼吸节奏)
   - lexicon (口头禅与修辞)
   - discourse (篇章逻辑展开)
   - anti_ai (去套话纯净度)
5. critique_feedback (具体修改建议)：指出文章中哪些句子写得太假/太空，或者哪段需要加强作者口吻（100字以内犀利批注）。

只输出合法 JSON，不要输出任何其他解释文字。
"""

JUDGE_USER_PROMPT_TEMPLATE = """## 目标作者文风指南
{style_guide}

## 待审校生成的文章
{article_content}

---
请对照上述文风指南完成独立裁决，输出评分 JSON。
"""


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
        anti_ai_score, ai_penalty, detected_cliches = MetricEvaluator.evaluate_anti_ai(
            article,
            custom_forbidden=profile.qualitative.anti_patterns.forbidden_words
        )

        # 2. 统计语言学拟合度计算 (句长均值 + 方差 + TTR)
        stylometric_similarity, breakdown = MetricEvaluator.evaluate_stylometric_fit(
            article,
            target_metrics=profile.quantitative
        )

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
        # Final Score = 0.35 * Fidelity + 0.25 * HumanPref + 0.20 * Stylometrics + 0.20 * Logic - AI_Penalty
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
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n", "", text)
            text = re.sub(r"\n```$", "", text)
        return json.loads(text)
