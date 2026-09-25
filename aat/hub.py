"""Hugging Face Hub checkpoint sync: work that outlives a 12 h Kaggle session.

Every training script pushes its resume checkpoint after each epoch and pulls it at start-up,
so a new session continues from the last finished epoch. Standalone (no aat imports), so the
same file is copied into the official RAMP checkout as `hub_sync.py`.

Token: env HF_TOKEN, or the Kaggle secret named HF_TOKEN (Add-ons -> Secrets).
Repo:  env HF_REPO, e.g. "your-hf-username/aat-checkpoints" (created private if missing).

CLI:
    python -m aat.hub pull <repo> <path_in_repo> <local_path>
    python -m aat.hub push <repo> <local_path> <path_in_repo>
"""
import os
import shutil
import sys


def get_token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        os.environ["HF_TOKEN"] = tok
        return tok
    except Exception:
        return None


def get_repo(repo=None):
    return repo or os.environ.get("HF_REPO")


_created = set()


def _api(repo):
    from huggingface_hub import HfApi
    api = HfApi(token=get_token())
    if repo not in _created:
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        _created.add(repo)
    return api


def push(local_path, path_in_repo, repo=None, message=None, squash=None):
    """Upload one file. Never raises: a failed upload must not kill a training run.
    squash=True drops old revisions afterwards so frequently overwritten resume checkpoints
    (~180 MB each) don't pile up in the repo history. Default: squash for resume files."""
    if squash is None:
        squash = os.path.basename(path_in_repo) in ("resume.pth", "last.pt")
    repo = get_repo(repo)
    if not repo or not os.path.exists(local_path):
        return False
    try:
        _api(repo).upload_file(path_or_fileobj=local_path, path_in_repo=path_in_repo, repo_id=repo,
                               commit_message=message or f"update {path_in_repo}")
        print(f"[hub] pushed {path_in_repo} -> {repo}", flush=True)
        if squash:
            try:
                _api(repo).super_squash_history(repo_id=repo)
            except Exception as e:
                print(f"[hub] squash skipped: {e}", flush=True)
        return True
    except Exception as e:  # network hiccup, rate limit, ...
        print(f"[hub] WARNING push failed for {path_in_repo}: {e}", flush=True)
        return False


def pull(path_in_repo, local_path, repo=None):
    """Download one file if it exists on the Hub. Returns True on success."""
    repo = get_repo(repo)
    if not repo:
        return False
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=repo, filename=path_in_repo, token=get_token(), repo_type="model")
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        shutil.copy(p, local_path)
        print(f"[hub] pulled {path_in_repo} <- {repo}", flush=True)
        return True
    except Exception as e:
        print(f"[hub] nothing to pull for {path_in_repo} ({type(e).__name__})", flush=True)
        return False


if __name__ == "__main__":
    op, repo, a, b = sys.argv[1:5]
    ok = pull(a, b, repo) if op == "pull" else push(a, b, repo)
    sys.exit(0 if ok else 1)
