#!/usr/bin/env python3
"""
Build target-comment retrieval artifacts for PluRule eval RAG.

The artifact stores cosine similarities from query split target comments
to candidate split target comments. Each thread pair contributes two target
comments: the violating leaf and the compliant leaf.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(1, str(PROJECT_ROOT))

import config as eval_config
import helpers


def _load_root_config() -> Any:
    spec = importlib.util.spec_from_file_location(
        "plurule_root_config",
        PROJECT_ROOT / "config.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load root config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT_CONFIG = _load_root_config()
DEFAULT_TASK_DESCRIPTION = "Retrieve similar Reddit target comments for moderation examples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a dense target-comment similarity matrix for eval RAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query-split", default="test", choices=list(eval_config.DATASET_FILES.keys()))
    parser.add_argument("--candidate-split", default="train", choices=list(eval_config.DATASET_FILES.keys()))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cuda", default="0", help="CUDA_VISIBLE_DEVICES for vLLM embedding")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--embed-batch-size", type=int, default=512)
    parser.add_argument("--similarity-batch-size", type=int, default=256)
    parser.add_argument("--query-limit", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--candidate-limit", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--task-description", default=DEFAULT_TASK_DESCRIPTION)
    return parser.parse_args()


def _quiet_logger():
    class Logger:
        def info(self, message):
            print(message)

        def warning(self, message):
            print(f"WARNING: {message}")

        def error(self, message):
            print(f"ERROR: {message}")

    return Logger()


def _correct_rule(pair: Dict[str, Any], thread_type: str) -> str:
    correct = pair[f"{thread_type}_correct_answer"]
    for option in pair.get(f"{thread_type}_answer_options", []):
        if option.get("label") == correct:
            return option.get("rule", "")
    return ""


def flatten_targets(split: str, limit: int = None) -> List[Dict[str, Any]]:
    pairs = helpers.load_dataset(split, _quiet_logger(), debug=False)
    targets = []
    for pair_index, pair in enumerate(pairs):
        for thread_type in ("violating", "compliant"):
            thread = pair[f"{thread_type}_thread"]
            leaf = thread[-1] if thread else {}
            metadata = pair.get("metadata", {})
            comment_id = metadata.get(f"{thread_type}_comment_id") or leaf.get("id", "")
            target_key = f"{split}:{pair['mod_comment_id']}:{thread_type}"
            body = leaf.get("body", "") or ""
            rule_cluster_id = metadata.get("rule_cluster_id", -1)
            rule_cluster_label = metadata.get("rule_cluster_label", "Other")

            targets.append({
                "target_key": target_key,
                "split": split,
                "pair_index": pair_index,
                "comment_id": comment_id,
                "thread_type": thread_type,
                "mod_comment_id": pair["mod_comment_id"],
                "submission_id": pair["submission_id"],
                "subreddit": pair["subreddit"],
                "subreddit_cluster_id": pair.get("subreddit_cluster_id", -1),
                "subreddit_cluster_label": pair.get("subreddit_cluster_label", "Other"),
                "rule_cluster_id": rule_cluster_id,
                "rule_cluster_label": rule_cluster_label,
                "rule": metadata.get("rule", ""),
                "correct_answer": pair[f"{thread_type}_correct_answer"],
                "correct_rule": _correct_rule(pair, thread_type),
                "body": body,
            })
            if limit is not None and len(targets) >= limit:
                return targets
    return targets


def build_embedding_texts(records: List[Dict[str, Any]], task_description: str) -> List[str]:
    texts = []
    for record in records:
        body = (record.get("body") or "").strip() or "[empty]"
        texts.append(f"Instruct: {task_description}\nQuery: {body}")
    return texts


def embed_texts(texts: List[str], model_name: str, args: argparse.Namespace):
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.inputs import TokensPrompt
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )["input_ids"]
    max_len = max((len(tokens) for tokens in tokenized), default=16)
    max_model_len = max(16, max_len)
    print(f"Loading embedding model {model_name} with max_model_len={max_model_len}")
    model = LLM(
        model=model_name,
        task="embed",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=max_model_len,
        seed=0,
    )

    embeddings = []
    for start in range(0, len(tokenized), args.embed_batch_size):
        end = min(start + args.embed_batch_size, len(tokenized))
        prompts = [TokensPrompt(prompt_token_ids=tokens) for tokens in tokenized[start:end]]
        outputs = model.embed(prompts)
        batch = torch.tensor([output.outputs.embedding for output in outputs], dtype=torch.float32)
        embeddings.append(batch)
        print(f"Embedded {end}/{len(tokenized)} texts")

    if not embeddings:
        return torch.empty((0, 0), dtype=torch.float32), max_model_len
    matrix = torch.cat(embeddings, dim=0)
    matrix = torch.nn.functional.normalize(matrix, p=2, dim=1)
    return matrix, max_model_len


def compute_similarity_matrix(query_embeddings, candidate_embeddings, dtype: str, batch_size: int):
    import torch

    output_dtype = torch.float16 if dtype == "float16" else torch.float32
    sims = torch.empty(
        (query_embeddings.shape[0], candidate_embeddings.shape[0]),
        dtype=output_dtype,
        device="cpu",
    )
    candidates_t = candidate_embeddings.float().T.contiguous()
    for start in range(0, query_embeddings.shape[0], batch_size):
        end = min(start + batch_size, query_embeddings.shape[0])
        scores = query_embeddings[start:end].float() @ candidates_t
        sims[start:end] = scores.to(output_dtype).cpu()
        print(f"Computed similarities for queries {end}/{query_embeddings.shape[0]}")
    return sims


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    output = args.output
    if output is None:
        output = (
            eval_config.OUTPUT_DIR
            / "rag"
            / f"{args.query_split}_to_{args.candidate_split}_target_comment_similarity.pt"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    query_records = flatten_targets(args.query_split, args.query_limit)
    candidate_records = flatten_targets(args.candidate_split, args.candidate_limit)
    if not query_records:
        raise ValueError(f"No query records found for split {args.query_split}")
    if not candidate_records:
        raise ValueError(f"No candidate records found for split {args.candidate_split}")

    model_name = ROOT_CONFIG.EMBEDDING_MODEL
    query_texts = build_embedding_texts(query_records, args.task_description)
    candidate_texts = build_embedding_texts(candidate_records, args.task_description)

    all_texts = query_texts + candidate_texts
    print(
        f"Embedding {len(query_records)} query targets and "
        f"{len(candidate_records)} candidate targets"
    )
    all_embeddings, max_model_len = embed_texts(all_texts, model_name, args)
    query_embeddings = all_embeddings[:len(query_texts)]
    candidate_embeddings = all_embeddings[len(query_texts):]

    similarity_matrix = compute_similarity_matrix(
        query_embeddings,
        candidate_embeddings,
        args.dtype,
        args.similarity_batch_size,
    )

    artifact = {
        "query_split": args.query_split,
        "candidate_split": args.candidate_split,
        "embedding_model": model_name,
        "scoring_method": "cosine_similarity",
        "text_format": f"Instruct: {args.task_description}\\nQuery: {{body}}",
        "dtype": args.dtype,
        "max_model_len": max_model_len,
        "queries": query_records,
        "candidates": candidate_records,
        "similarity_matrix": similarity_matrix,
    }

    import torch

    torch.save(artifact, output)
    artifact_sha256 = helpers.file_sha256(output)
    artifact_size_bytes = output.stat().st_size
    summary_path = output.with_suffix(".summary.json")
    summary = {
        "query_split": args.query_split,
        "candidate_split": args.candidate_split,
        "num_queries": len(query_records),
        "num_candidates": len(candidate_records),
        "matrix_shape": list(similarity_matrix.shape),
        "embedding_model": model_name,
        "dtype": args.dtype,
        "max_model_len": max_model_len,
        "output": str(output),
        "artifact_sha256": artifact_sha256,
        "artifact_size_bytes": artifact_size_bytes,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved retrieval artifact to {output}")
    print(f"Saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
