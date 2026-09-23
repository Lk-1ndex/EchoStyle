"""Document and URL ingestion with per-source outcomes."""

import hashlib
import re
import tempfile
from pathlib import Path
from typing import Any, Protocol, Sequence

from src.agents.base import AgentContext
from src.agents.coordinator import CoordinatorAgent
from src.extractors.capabilities import validate_source
from src.extractors.results import (
    ExtractionBatchResult,
    ExtractionFailure,
    ExtractionStatus,
    SourceExtractionResult,
)


class BinaryWriter(Protocol):
    def write(self, data: bytes, /) -> int: ...


def stream_upload_to_file(uploaded_file: Any, destination: BinaryWriter) -> str:
    digest = hashlib.sha256()
    seek = getattr(uploaded_file, "seek", None)
    read = getattr(uploaded_file, "read", None)
    if callable(read):
        if callable(seek):
            seek(0)
        while True:
            chunk = read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            destination.write(chunk)
        if callable(seek):
            seek(0)
        return digest.hexdigest()

    getbuffer = getattr(uploaded_file, "getbuffer", None)
    payload = getbuffer() if callable(getbuffer) else uploaded_file.getvalue()
    digest.update(payload)
    destination.write(payload)
    return digest.hexdigest()


def extract_uploaded_files(
    uploaded_files: Sequence[Any],
    coordinator: CoordinatorAgent,
    known_hashes: set[str],
) -> tuple[ExtractionBatchResult, list[str], list[str]]:
    outcomes: list[SourceExtractionResult] = []
    skipped: list[str] = []
    state = AgentContext()

    for index, uploaded_file in enumerate(uploaded_files):
        temp_path: Path | None = None
        digest = ""
        try:
            suffix = Path(uploaded_file.name).suffix.lower()
            validate_source(f"upload{suffix}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                digest = stream_upload_to_file(uploaded_file, temp_file)
                temp_path = Path(temp_file.name)

            if digest in known_hashes:
                skipped.append(uploaded_file.name)
                continue

            result = coordinator.extract_sources_detailed([str(temp_path)], state=state)
            outcome = result.outcomes[0]
            if outcome.article is not None:
                outcome.article["title"] = Path(uploaded_file.name).stem
                outcome.article["document_id"] = digest
            outcomes.append(
                outcome.model_copy(update={"source_index": index, "source": uploaded_file.name})
            )
        except Exception as error:
            outcomes.append(
                SourceExtractionResult(
                    source_index=index,
                    source=uploaded_file.name,
                    status=ExtractionStatus.FAILED,
                    error=ExtractionFailure(
                        error_type=type(error).__name__,
                        message=str(error),
                    ),
                    exception=error,
                )
            )
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    return ExtractionBatchResult(outcomes=outcomes), state.execution_logs, skipped


def extract_wechat_urls(
    message: str,
    coordinator: CoordinatorAgent,
    known_hashes: set[str],
) -> tuple[ExtractionBatchResult, list[str]]:
    urls = re.findall(r"https?://mp\.weixin\.qq\.com/[^\s)]+", message)
    new_urls = [
        url
        for url in urls
        if hashlib.sha256(url.encode("utf-8")).hexdigest() not in known_hashes
    ]
    if not new_urls:
        return ExtractionBatchResult(), []

    state = AgentContext()
    result = coordinator.extract_sources_detailed(new_urls, state=state)
    for outcome in result.successes:
        if outcome.article is not None:
            outcome.article["document_id"] = hashlib.sha256(
                outcome.source.encode("utf-8")
            ).hexdigest()
    return result, state.execution_logs


def commit_successful_outcomes(
    documents: list[dict[str, Any]],
    document_hashes: set[str],
    *batches: ExtractionBatchResult,
) -> list[dict[str, Any]]:
    """Commit successful documents and their IDs together after extraction."""
    new_documents = [
        outcome.article
        for batch in batches
        for outcome in batch.successes
        if outcome.article is not None
    ]
    new_hashes = {
        str(article["document_id"])
        for article in new_documents
        if article.get("document_id")
    }
    documents.extend(new_documents)
    document_hashes.update(new_hashes)
    return new_documents


def failure_messages(*batches: ExtractionBatchResult) -> list[str]:
    return [
        f"{outcome.source}: {outcome.error.message}"
        for batch in batches
        for outcome in batch.failures
        if outcome.error is not None
    ]
