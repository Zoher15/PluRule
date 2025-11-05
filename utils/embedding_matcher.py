"""Embedding-based comment-rule matcher using vLLM."""

import os
import time
import torch
from transformers import AutoTokenizer
from vllm import LLM
from vllm.inputs import TokensPrompt

from config import PATHS, EMBEDDING_MODEL


class SimpleCommentRuleMatcher:
    """Comment-rule matcher using embeddings."""

    def __init__(self, model_name: str = None, max_model_len: int = 2048):
        self.model_name = model_name or EMBEDDING_MODEL

        print(f"Loading embedding model: {self.model_name} with max_model_len={max_model_len}")
        self.model = LLM(
            model=self.model_name,
            task="embed",
            gpu_memory_utilization=0.97,
            enforce_eager=True,
            max_model_len=max_model_len,
            seed=0
        )
        print("✅ Embedding model loaded successfully")

    @staticmethod
    def save_similarity_matrix(cosine_similarities, comments, rules, subreddit_name):
        """Save similarity matrix to disk in PyTorch format."""
        comment_mapping = {comment['id']: row_idx for row_idx, comment in enumerate(comments)}
        rule_indices = [rule.get("rule_index", i) for i, rule in enumerate(rules)]

        similarity_data = {
            'cosine_similarity_matrix': cosine_similarities.float(),
            'comment_mapping': comment_mapping,
            'rule_indices': rule_indices,
            'subreddit': subreddit_name,
            'num_comments': len(comments),
            'num_rules': len(rules),
            'scoring_method': 'cosine_similarity'
        }

        output_dir = PATHS.get('matched_comments')
        if not output_dir or not os.path.exists(output_dir):
            raise ValueError(f"Output directory not configured or missing: {output_dir}")

        matrix_file = os.path.join(output_dir, f"{subreddit_name}_similarity_matrix.pt")
        torch.save(similarity_data, matrix_file)
        print(f"💾 Saved similarity matrix to {matrix_file}")

    def calculate_similarities_pretokenized(self, comments, rules, tokenized_comments, tokenized_rules):
        """Calculate cosine similarities from pre-tokenized inputs and save to .pt file."""
        if not comments or not rules or len(rules) <= 1:
            print("⚠️  Insufficient data to process")
            return False

        if len(tokenized_comments) != len(comments) or len(tokenized_rules) != len(rules):
            print("❌ Mismatch between original and tokenized data lengths")
            return False

        print(f"Creating TokensPrompt objects for {len(tokenized_comments)} comments and {len(tokenized_rules)} rules...")
        comment_prompts = [TokensPrompt(prompt_token_ids=tokens) for tokens in tokenized_comments]
        rule_prompts = [TokensPrompt(prompt_token_ids=tokens) for tokens in tokenized_rules]

        print(f"Embedding {len(comment_prompts)} comments...")
        comment_outputs = self.model.embed(comment_prompts)
        comment_embeddings = torch.tensor([o.outputs.embedding for o in comment_outputs])

        print(f"Embedding {len(rule_prompts)} rule documents...")
        rule_outputs = self.model.embed(rule_prompts)
        rule_embeddings = torch.tensor([o.outputs.embedding for o in rule_outputs])

        print("Computing similarities...")
        similarities = comment_embeddings @ rule_embeddings.T

        subreddit_name = comments[0].get('subreddit', 'unknown') if comments else 'unknown'
        self.save_similarity_matrix(similarities, comments, rules, subreddit_name)

        print(f"✅ Calculated similarities for {len(comments)} comments and {len(rules)} rules")
        return True

    @classmethod
    def pretokenize_inputs(cls, comments, rules, model_name, task_description):
        """
        Pretokenize comments and rules to find optimal max_model_len.

        Returns:
            - tokenized_comments: List of token IDs for each comment
            - tokenized_rules: List of token IDs for each rule
            - max_length: Maximum token length across all inputs
        """
        if not comments or not rules or len(rules) <= 1:
            print("⚠️  Insufficient data to tokenize")
            return [], [], 0

        print(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Tokenize comments with instruction (batch processing for speed)
        print(f"Tokenizing {len(comments)} comments...")
        comment_texts = [f'Instruct: {task_description}\nQuery: {comment.get("body_clean", "")}'
                        for comment in comments]

        tokenized_comments = tokenizer(
            comment_texts,
            add_special_tokens=True,
            padding=False,
            truncation=False
        )['input_ids']
        comment_lengths = [len(tokens) for tokens in tokenized_comments]

        # Tokenize rules (documents, no instruction) - batch processing
        print(f"Tokenizing {len(rules)} rules...")
        rule_texts = [rule['rule_comprehensive'] for rule in rules]

        tokenized_rules = tokenizer(
            rule_texts,
            add_special_tokens=True,
            padding=False,
            truncation=False
        )['input_ids']
        rule_lengths = [len(tokens) for tokens in tokenized_rules]

        # Calculate statistics
        max_comment_len = max(comment_lengths) if comment_lengths else 0
        max_rule_len = max(rule_lengths) if rule_lengths else 0
        max_length = max(max_comment_len, max_rule_len)

        avg_comment_len = sum(comment_lengths) / len(comment_lengths) if comment_lengths else 0
        avg_rule_len = sum(rule_lengths) / len(rule_lengths) if rule_lengths else 0

        print(f"Tokenization complete:")
        print(f"  Max comment length: {max_comment_len}")
        print(f"  Max rule length: {max_rule_len}")
        print(f"  Overall max length: {max_length}")
        print(f"  Avg comment length: {avg_comment_len:.1f}")
        print(f"  Avg rule length: {avg_rule_len:.1f}")

        return tokenized_comments, tokenized_rules, max_length
