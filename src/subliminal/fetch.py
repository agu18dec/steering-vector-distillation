"""Fetch v_teacher vector (and optional steered-gen datasets) from Hugging Face.

    sl-fetch                                     # v_teacher only
    sl-fetch download_data=True                  # + v_teacher SL datasets
    sl-fetch download_cross_model_vectors=True   # + per-(model,trait) cross-model vectors
"""

import shutil
from pathlib import Path

import pydra
from huggingface_hub import hf_hub_download


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.repo_vectors = "camilablank/v_teacher_v_student_vectors"
        self.repo_vectors_type = "model"
        self.repo_data = "camilablank/sufficiency-necessity-data"
        self.repo_data_type = "dataset"

        self.vector_filename = "section-3_numbers_vectors/v_teacher_all_prompt_tokens.pt"
        self.vector_local_path = "data/vectors/v_teacher_qwen25_cat.pt"

        self.download_data = False

        self.suff_remote = "section-3_numbers_data/v_teacher_sufficiency_qwen2.5-7b_L23_a6_30k_numbers.jsonl"
        self.suff_local_dir = "data/filtered/v_teacher_suff_hf"
        self.suff_local_basename = "filtered_30000.jsonl"

        self.nec_remote = "section-3_numbers_data/v_teacher_necessity_qwen2.5-7b_L20_10k_numbers.jsonl"
        self.nec_local_dir = "data/filtered/v_teacher_nec_hf"
        self.nec_local_basename = "filtered_10000.jsonl"

        self.download_cross_model_vectors = False
        self.cross_model_remote_prefix = "cross_model_vectors"
        self.cross_model_local_dir = "data/vectors/cross_model"
        self.cross_model_aliases = ["qwen", "olmo"]
        self.cross_model_traits = [
            "cat",
            "dog",
            "owl",
            "otter",
            "anger",
            "happiness",
            "sadness",
            "fear",
            "surprise",
            "disgust",
        ]


def _pull(repo_id: str, filename: str, repo_type: str, local_path: Path) -> None:
    print(f"[fetch] {repo_type}:{repo_id}::{filename}")
    cached = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cached, local_path)
    print(f"[fetch]   -> {local_path.resolve()}")


@pydra.main(Config)
def main(config: Config):
    _pull(config.repo_vectors, config.vector_filename, config.repo_vectors_type, Path(config.vector_local_path))

    if config.download_cross_model_vectors:
        for alias in config.cross_model_aliases:
            for trait in config.cross_model_traits:
                remote = f"{config.cross_model_remote_prefix}/{alias}/v_{trait}.pt"
                local = Path(config.cross_model_local_dir) / alias / f"v_{trait}.pt"
                _pull(config.repo_vectors, remote, config.repo_vectors_type, local)
        print()
        print(f"[fetch] cross-model vectors -> {Path(config.cross_model_local_dir).resolve()}")
        print("Run the matrix:")
        print("  sl-cross-model")

    if config.download_data:
        suff_local = Path(config.suff_local_dir) / config.suff_local_basename
        nec_local = Path(config.nec_local_dir) / config.nec_local_basename
        _pull(config.repo_data, config.suff_remote, config.repo_data_type, suff_local)
        _pull(config.repo_data, config.nec_remote, config.repo_data_type, nec_local)
        print()
        print("Train directly against the fetched data (skip sl-gen-steered + sl-filter):")
        print("  sl-train \\")
        print(f"    dataset_run_name={Path(config.suff_local_dir).name} \\")
        print(f"    filtered_dir={Path(config.suff_local_dir).parent} \\")
        print(f"    filtered_basename={config.suff_local_basename} \\")
        print("    run_name=cat_qwen25_v_teacher_suff_train_hf_s1")
        print("  sl-train \\")
        print(f"    dataset_run_name={Path(config.nec_local_dir).name} \\")
        print(f"    filtered_dir={Path(config.nec_local_dir).parent} \\")
        print(f"    filtered_basename={config.nec_local_basename} \\")
        print("    run_name=cat_qwen25_v_teacher_nec_train_hf_s1")
    print()
    print("[fetch] done")


if __name__ == "__main__":
    main()
