from typing import List, Optional
from .base import BaseAgent
from .state import AgentState, AgentStatus
from src.core.config import LLMConfig
from src.core.models import DeepStyleProfile
from src.core.model_provider import ModelProvider
from src.memory.memory_manager import MemoryManager

REWRITE_PROMPT_TEMPLATE = """这是上一轮初稿：
{draft}

上一轮总编辑审校批注与扣分项反馈：
{feedback}

---
请针对上述具体批注进行【反思重构】：
1. 立即清除审校指出的所有违规八股词汇与死板转折。
2. 强化目标作者的句长节奏（平均句长约 {avg_len} 字）与口癖特征。
3. 严格保留上一轮的核心论证要点，但将文字质感彻底重塑。
直接输出反思修改后的全新成文。
"""


class WriterAgent(BaseAgent):
    """
    记忆驱动的创作智能体：
    负责结合长期风格记忆 (Style Memory 动态召回) 与文风档案进行精准创作，
    管理草稿版本演进链，并支持根据 Critic 反馈进行反思迭代。
    """

    def __init__(self, llm_config: LLMConfig, memory_manager: MemoryManager):
        super().__init__(name="WriterAgent", description="负责结合风格记忆与反思反馈进行高质量创作与重构")
        self.llm_config = llm_config
        self.memory_manager = memory_manager
        self.model_provider = ModelProvider(llm_config)

    def run(
        self,
        state: AgentState,
        profile: DeepStyleProfile,
        feedback: Optional[str] = None,
        previous_draft: Optional[str] = None,
    ) -> str:
        # 1. 动态语义召回最匹配的 Few-shot 片段
        if not state.memory_snapshot:
            query = f"{state.topic} {state.key_points}"
            state.transition_to(
                AgentStatus.DRAFTING if not feedback else AgentStatus.REFLECTING,
                f"向 Style Memory 检索与选题 [{state.topic}] 契合的高光范例..."
            )
            few_shots = self.memory_manager.retrieve_dynamic_few_shots(query, top_k=3)
            state.memory_snapshot = [{"content": s} for s in few_shots]

        dynamic_shots = [s["content"] for s in state.memory_snapshot]
        system_prompt = profile.to_system_prompt(dynamic_few_shots=dynamic_shots)

        # 2. 区分【首轮创作】还是【反思重写】
        if feedback and previous_draft:
            state.transition_to(AgentStatus.REFLECTING, f"执行第 {state.retry_count} 轮反思重构...")
            avg_len = profile.quantitative.avg_sentence_length if profile.quantitative else 20.0
            user_prompt = REWRITE_PROMPT_TEMPLATE.format(
                draft=previous_draft,
                feedback=feedback,
                avg_len=avg_len,
            )
        else:
            state.transition_to(AgentStatus.DRAFTING, f"开始初稿撰写: 主题 [{state.topic}], 目标字数: {state.word_count} 字...")
            user_prompt = f"""围绕以下新主题创作一篇完整的文章：
- **文章主题**：{state.topic}
- **核心论述要点**：
{state.key_points or '由作者根据主题自由构思，突出深度与独特见解'}
- **目标字数**：{state.word_count} 左右
- **目标读者**：{state.target_audience or '大众读者'}

请全情投入作者人设，严格遵守作者的句长与呼吸节奏，杜绝任何 AI 套话。
"""

        generated_text = self.model_provider.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
        )

        # 记录版本草稿入链
        state.record_draft(agent_name=self.name, draft=generated_text)
        return generated_text
