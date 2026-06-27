"""One-shot deploy of app/ to a public HF Gradio Space. Uses the cached HF token."""
from huggingface_hub import HfApi

REPO_ID = "britod/name-nationality-classifier"


def deploy():
    api = HfApi()
    url = api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="gradio",
                          private=False, exist_ok=True)
    print("space repo:", url)

    api.upload_folder(
        folder_path="app",
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="Deploy name-nationality classifier (deep + baselines, single + batch)",
        ignore_patterns=["__pycache__/*", "*.pyc", "_smoke*", "*_test.log", ".gitignore"],
        # remove any stale remote artifacts no longer present locally (e.g. the old
        # top-level deep_model.pt now relocated to models/deep/)
        delete_patterns=["models/deep_model.pt"],
    )
    print(f"\nDONE -> https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    deploy()
