"""
Statistics utilities for Reddit moderation pipeline.

Provides reusable functions for Jensen-Shannon Divergence,
ranking, and rule distribution analysis.
"""

import numpy as np
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Any
from collections import defaultdict


def calculate_jsd_from_uniform(distribution: Dict[str, int]) -> float:
    """
    Calculate Jensen-Shannon Divergence from uniform distribution.

    Args:
        distribution: Dictionary mapping categories to counts

    Returns:
        JSD value (0 = uniform, 1 = maximum divergence)
    """
    if not distribution:
        return 1.0  # Maximum divergence

    counts = list(distribution.values())
    total = sum(counts)

    if total == 0:
        return 1.0

    # Observed probabilities
    observed = np.array([count / total for count in counts])

    # Uniform probabilities (ideal distribution)
    uniform = np.array([1.0 / len(counts)] * len(counts))

    # Calculate Jensen-Shannon Distance using scipy
    # Note: scipy returns distance (sqrt of divergence), we want divergence
    js_distance = jensenshannon(observed, uniform)
    jsd = js_distance ** 2  # Convert distance to divergence

    return float(jsd)


def rank_by_score(items: List[Dict], score_key: str, ascending: bool = True,
                  filter_func: callable = None) -> List[Dict]:
    """
    Generic ranking function by score field.

    Args:
        items: List of dictionaries to rank
        score_key: Key to use for ranking
        ascending: True for ascending order (lower = better rank)
        filter_func: Optional function to filter items before ranking

    Returns:
        List with added 'rank' field
    """
    # Filter items if function provided
    if filter_func:
        valid_items = [item for item in items if filter_func(item)]
        invalid_items = [item for item in items if not filter_func(item)]
    else:
        valid_items = items[:]
        invalid_items = []

    # Sort by score
    valid_items.sort(key=lambda x: x.get(score_key, float('inf') if ascending else float('-inf')),
                     reverse=not ascending)

    # Add ranks
    for i, item in enumerate(valid_items):
        item['rank'] = i + 1

    # Add high ranks to invalid items
    for item in invalid_items:
        item['rank'] = 999999

    return valid_items + invalid_items


def analyze_rule_distribution(stats_list: List[Dict], rule_matches_key: str = 'rule_matches') -> Dict:
    """
    Analyze rule distribution across multiple datasets.

    Args:
        stats_list: List of statistics dictionaries
        rule_matches_key: Key containing rule match data

    Returns:
        Dictionary with rule distribution analysis
    """
    rule_totals = defaultdict(int)

    for stats in stats_list:
        rule_matches = stats.get(rule_matches_key, {})
        for rule_id, count in rule_matches.items():
            rule_totals[rule_id] += count

    # Sort rules by total matches
    sorted_rules = sorted(rule_totals.items(), key=lambda x: x[1], reverse=True)

    return {
        'total_rules': len(rule_totals),
        'rule_distribution': dict(sorted_rules),
        'top_rules': dict(sorted_rules[:20]) if sorted_rules else {},
        'total_matches': sum(rule_totals.values())
    }