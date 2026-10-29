"""Download gate. Unapproved or unpinned sources are refused before any transfer."""

from __future__ import annotations

from minimol.errors import LicenseError, MinimolError


def select_files_within_budget(files: list[dict], max_bytes: int) -> list[dict]:
    chosen = []
    total = 0
    for item in files:
        size = int(item["size"])
        if total + size > max_bytes:
            break
        chosen.append(item)
        total += size
    return chosen


def assert_download_allowed(decision: str, revision: str, license_name: str) -> None:
    # LOCAL_USE is a local training cache. It does not approve redistribution.
    if decision not in {"APPROVED", "LOCAL_USE"}:
        raise LicenseError(f"download blocked: decision={decision}")
    if not license_name or license_name.strip().lower() in {"unknown", "unverified"}:
        raise LicenseError("download blocked: license is not named")
    text = "" if revision is None else str(revision)
    if not text or "UNVERIFIED" in text or "REPLACE" in text:
        raise MinimolError("download blocked: revision is not pinned")


def huggingface_download(repo: str, revision: str, dest, max_bytes: int, decision: str, license_name: str):
    """Optional Hugging Face download. Tests should not call this."""

    assert_download_allowed(decision, revision, license_name)
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise MinimolError("huggingface_hub is not installed") from exc
    api = HfApi()
    files = []
    for item in api.list_repo_files(repo, revision=revision):
        info = api.get_paths_info(repo, [item], revision=revision)[0]
        size = int(getattr(info, "size", 0) or 0)
        files.append({"path": item, "size": size})
    chosen = select_files_within_budget(files, max_bytes)
    if not chosen:
        raise MinimolError("no files fit the download budget")
    saved = []
    for item in chosen:
        saved.append(
            hf_hub_download(
                repo_id=repo,
                filename=item["path"],
                revision=revision,
                local_dir=str(dest),
            )
        )
    return saved
