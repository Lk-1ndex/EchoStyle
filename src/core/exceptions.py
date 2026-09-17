class EchoStyleException(Exception):
    """EchoStyle 核心异常基类"""
    pass


class InvalidStateTransitionError(EchoStyleException):
    """状态机非法状态跳变异常"""
    def __init__(self, current_status: str, target_status: str, allowed: list):
        super().__init__(
            f"非法状态迁移拒绝: 不能从 [{current_status}] 直接跃迁至 [{target_status}]。"
            f"当前状态合法目标为: {allowed}"
        )
        self.current_status = current_status
        self.target_status = target_status
        self.allowed = allowed


class CheckpointNotFoundError(EchoStyleException):
    """找不到指定的检查点快照异常"""
    pass


class ModelProviderError(EchoStyleException):
    """模型层重试与降级后仍失败的异常"""
    pass


class EmbeddingUnavailableError(EchoStyleException):
    """向量服务不可用异常（Fail-Closed 严控检索语义纯度，拒绝静默退化）"""
    pass

