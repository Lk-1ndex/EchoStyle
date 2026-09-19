import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from filelock import FileLock

from src.core.models import DeepStyleProfile


class ProfileStore:
    """Persistent registry for complete DeepStyleProfile records."""

    SCHEMA_VERSION = 1

    def __init__(self, storage_path: str = "./profiles/deep_style_profiles_v2.json"):
        self.storage_path = Path(storage_path)
        self._file_lock = FileLock(f"{self.storage_path}.lock", timeout=60)

    def save(self, profile: DeepStyleProfile, make_active: bool = True) -> None:
        profile_id = self._require_profile_id(profile)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        with self._file_lock:
            state = self._read_state()
            state["profiles"][profile_id] = profile.model_dump(mode="json")
            if make_active:
                state["active_profile_id"] = profile_id
            self._write_state(state)

    def get(self, profile_id: str) -> Optional[DeepStyleProfile]:
        if not profile_id or not profile_id.strip():
            return None
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock:
            state = self._read_state()
        payload = state["profiles"].get(profile_id)
        return DeepStyleProfile.model_validate(payload) if payload is not None else None

    def get_active(self) -> Optional[DeepStyleProfile]:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock:
            state = self._read_state()
        profile_id = state.get("active_profile_id")
        payload = state["profiles"].get(profile_id) if profile_id else None
        return DeepStyleProfile.model_validate(payload) if payload is not None else None

    def list_profiles(self) -> List[DeepStyleProfile]:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock:
            state = self._read_state()
        return [DeepStyleProfile.model_validate(payload) for payload in state["profiles"].values()]

    def set_active(self, profile_id: str) -> DeepStyleProfile:
        if not profile_id or not profile_id.strip():
            raise ValueError("profile_id 不能为空")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        with self._file_lock:
            state = self._read_state()
            payload = state["profiles"].get(profile_id)
            if payload is None:
                raise KeyError(f"文风档案不存在: {profile_id}")
            state["active_profile_id"] = profile_id
            self._write_state(state)
        return DeepStyleProfile.model_validate(payload)

    def has_profile(self, profile_id: Optional[str]) -> bool:
        if not profile_id:
            return False
        return self.get(profile_id) is not None

    @classmethod
    def _empty_state(cls) -> Dict[str, Any]:
        return {
            "schema_version": cls.SCHEMA_VERSION,
            "active_profile_id": None,
            "profiles": {},
        }

    @staticmethod
    def _require_profile_id(profile: DeepStyleProfile) -> str:
        profile_id = (profile.profile_id or "").strip()
        if not profile_id:
            raise ValueError("DeepStyleProfile 缺少 profile_id，不能持久化")
        return profile_id

    def _read_state(self) -> Dict[str, Any]:
        if not self.storage_path.exists():
            return self._empty_state()

        with open(self.storage_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        if not isinstance(state, dict):
            raise ValueError(f"文风档案库格式损坏: {self.storage_path}")
        if state.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(f"不支持的文风档案库版本: {state.get('schema_version')}")
        profiles = state.get("profiles")
        if not isinstance(profiles, dict):
            raise ValueError(f"文风档案库缺少 profiles 对象: {self.storage_path}")

        for profile_id, payload in profiles.items():
            profile = DeepStyleProfile.model_validate(payload)
            if profile.profile_id != profile_id:
                raise ValueError(f"文风档案 ID 与索引键不一致: {profile_id}")

        active_profile_id = state.get("active_profile_id")
        if active_profile_id is not None and active_profile_id not in profiles:
            raise ValueError(f"当前激活的文风档案不存在: {active_profile_id}")
        return state

    def _write_state(self, state: Dict[str, Any]) -> None:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.storage_path.parent,
                suffix=".tmp",
                delete=False,
            ) as f:
                temp_path = Path(f.name)
                json.dump(state, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.storage_path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
