#!/usr/bin/env python3
"""
Stage 11: Human Evaluation from Final Dataset

Creates Google Forms for human evaluation of rule matching quality using the final
Stage 10 dataset. Samples 4 moderator comments per subreddit and creates forms where
each subreddit is a separate page with rules shown once at the top.

Usage: python 11_human_evaluation.py
"""

import os
import sys
import json
import random
import hashlib
import time
from typing import List, Dict, Any
from collections import defaultdict

# Google Forms API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES
from utils.files import write_json_file
from utils.logging import get_stage_logger, log_stage_start, log_stage_end

# ============================================================================
# Helper Functions
# ============================================================================

def stable_hash(value: str) -> int:
    """Create deterministic integer hash from string (reproducible across runs)."""
    return int.from_bytes(hashlib.sha256(value.encode('utf-8')).digest(), 'big')

# Configuration
COMMENTS_PER_SUBREDDIT = 4
SUBREDDITS_PER_FORM = 20
RANDOM_SEED = 0

# Google Forms API Configuration
SCOPES = [
    "https://www.googleapis.com/auth/forms.body",
    "https://www.googleapis.com/auth/drive.file",
]
CLIENT_SECRETS = "/data3/zkachwal/reddit-mod-collection-pipeline/credentials/client_secret_795576073496-qo2r4ntgn1drrqo31p98it9bmtd2hvm4.apps.googleusercontent.com.json"
TOKEN_FILE = "/data3/zkachwal/reddit-mod-collection-pipeline/credentials/token.json"


def authenticate():
    """Authenticate with Google APIs using OAuth2."""
    creds = None
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    except Exception:
        pass

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save for next run
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return creds


def load_final_dataset(logger) -> Dict[str, Any]:
    """Load the final hydrated dataset from Stage 9."""
    dataset_file = os.path.join(PATHS['data'], 'reddit_moderation_dataset_hydrated_v1.0.json.zst')

    if not os.path.exists(dataset_file):
        logger.error(f"❌ Final dataset not found: {dataset_file}")
        return None

    logger.info(f"📂 Loading final dataset from: {dataset_file} (using {PROCESSES} threads)")

    # Read compressed file using multi-threaded decompressor
    import zstandard
    with open(dataset_file, 'rb') as f:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(f, read_size=2**24) as reader:
            decompressed = reader.read()
            dataset = json.loads(decompressed.decode('utf-8'))

    logger.info(f"✅ Loaded dataset with {len(dataset.get('subreddits', []))} subreddits")
    return dataset


def sample_comments_from_subreddit(subreddit_data: Dict, num_samples: int, subreddit_name: str) -> List[Dict]:
    """Sample moderator comments from a subreddit's thread pairs.

    Uses subreddit name as seed for deterministic, independent sampling.
    """
    thread_pairs = subreddit_data.get('thread_pairs', [])

    if len(thread_pairs) == 0:
        return []

    # Sample thread pairs using subreddit-specific RNG for independence
    sample_size = min(num_samples, len(thread_pairs))
    subreddit_rng = random.Random(stable_hash(subreddit_name))
    sampled_pairs = subreddit_rng.sample(thread_pairs, sample_size)

    # Extract moderator comments
    sampled_comments = []
    for pair in sampled_pairs:
        comment = {
            'comment_id': pair.get('mod_comment_id'),
            'comment_body': pair.get('mod_comment', {}).get('body', ''),
            'matched_rule': pair.get('matched_rule', {}),
            'submission_id': pair.get('submission_id')
        }
        sampled_comments.append(comment)

    return sampled_comments


def prepare_subreddit_pages(dataset: Dict) -> List[Dict]:
    """Prepare data for each subreddit page."""
    subreddit_pages = []

    for subreddit_data in dataset['subreddits']:
        subreddit_name = subreddit_data['subreddit']
        rules = subreddit_data['rules']

        # Get subreddit description (from Stage 2 data stored in rules metadata)
        # The description should be in the subreddit metadata if preserved
        subreddit_description = "A community on Reddit"  # Default fallback

        # Sample comments (using subreddit-specific seed for independence)
        sampled_comments = sample_comments_from_subreddit(subreddit_data, COMMENTS_PER_SUBREDDIT, subreddit_name)

        if not sampled_comments:
            continue

        page_data = {
            'subreddit': subreddit_name,
            'description': subreddit_description,
            'rules': rules,
            'sampled_comments': sampled_comments,
            'jsd_score': subreddit_data.get('jsd_from_uniform', 0),
            'rank': subreddit_data.get('rank', 999)
        }

        subreddit_pages.append(page_data)

    # Sort alphabetically by subreddit name
    subreddit_pages.sort(key=lambda x: x['subreddit'].lower())

    return subreddit_pages


def create_rules_display_text(rules: List[Dict]) -> str:
    """Create formatted text for displaying rules at top of page."""
    rules_text_parts = []

    for rule in rules:
        rule_index = rule.get('rule_index', 0)
        short_name = rule.get('short_name_clean', rule.get('short_name', ''))
        description = rule.get('description_clean', rule.get('description', ''))
        violation = rule.get('violation_reason_clean', rule.get('violation_reason', ''))

        rule_text = f"📌 Rule {rule_index}\n"
        rule_text += f"SHORT NAME: {short_name}\n"
        rule_text += f"DESCRIPTION: {description}"

        if violation:
            rule_text += f"\nVIOLATION: {violation}"

        rules_text_parts.append(rule_text)

    return "\n\n".join(rules_text_parts)


def create_evaluation_form(service, subreddit_pages: List[Dict], form_part: int = 1, total_forms: int = 1) -> str:
    """Create Google Form with subreddit pages."""

    # Step 1: Create the basic form
    title = "Reddit Moderation Dataset - Human Evaluation"
    if total_forms > 1:
        title += f" (Part {form_part} of {total_forms})"

    form = {
        "info": {
            "title": title
        }
    }
    result = service.forms().create(body=form).execute()
    form_id = result["formId"]
    print(f"✅ Created form with ID: {form_id}")

    # Step 2: Build requests for form content
    requests = [
        # Update form description
        {
            "updateFormInfo": {
                "info": {
                    "title": title,
                    "description": (
                        f"Please evaluate moderator comments and select which rule(s) they refer to. "
                        f"Each page shows one subreddit with its rules listed at the top, followed by 4 moderator comments to evaluate. "
                        f"This form contains {len(subreddit_pages)} subreddits ({len(subreddit_pages) * COMMENTS_PER_SUBREDDIT} total questions)."
                        + (f" This is part {form_part} of {total_forms} forms." if total_forms > 1 else "")
                    )
                },
                "updateMask": "title,description"
            }
        }
    ]

    question_requests = []

    # Create questions for each subreddit page
    for page_index, page_data in enumerate(subreddit_pages):
        subreddit = page_data['subreddit']
        description = page_data['description']
        rules = page_data['rules']
        comments = page_data['sampled_comments']

        # Create page header with rules display
        rules_display = create_rules_display_text(rules)
        page_title = f"r/{subreddit}"
        page_description = f"{description}\n\n{'='*60}\nCOMMUNITY RULES\n{'='*60}\n\n{rules_display}\n\n{'='*60}\n\nPlease evaluate the following moderator comments:"

        # Add page break with rules display (except for first page)
        if page_index > 0:
            page_break_request = {
                "createItem": {
                    "item": {
                        "title": page_title,
                        "description": page_description,
                        "pageBreakItem": {}
                    }
                }
            }
            question_requests.append(page_break_request)
        else:
            # For first page, add an info section
            info_section = {
                "createItem": {
                    "item": {
                        "title": page_title,
                        "description": page_description,
                        "questionItem": {
                            "question": {
                                "required": False,
                                "choiceQuestion": {
                                    "type": "RADIO",
                                    "options": [{"value": "Ready to start"}]
                                }
                            }
                        }
                    }
                }
            }
            question_requests.append(info_section)

        # Create checkbox question for each comment
        for comment_index, comment in enumerate(comments):
            question_num = page_index * COMMENTS_PER_SUBREDDIT + comment_index + 1

            question_title = f"Comment {comment_index + 1} of {len(comments)}"
            question_description = f"Moderator Comment:\n\"{comment['comment_body']}\"\n\nWhich rule(s) is this comment referring to? Select all that apply."

            # Create choice options from rule short names
            choice_options = []
            for rule in rules:
                rule_index = rule.get('rule_index', 0)
                short_name = rule.get('short_name_clean', rule.get('short_name', ''))
                choice_options.append({
                    "value": f"{rule_index}. {short_name}"
                })

            # Add "None of the above" option
            choice_options.append({
                "value": "None of the above rules apply"
            })

            # Add "Other" option for notes
            choice_options.append({
                "isOther": True
            })

            question_request = {
                "createItem": {
                    "item": {
                        "title": question_title,
                        "description": question_description,
                        "questionItem": {
                            "question": {
                                "required": True,
                                "choiceQuestion": {
                                    "type": "CHECKBOX",
                                    "options": choice_options
                                }
                            }
                        }
                    }
                }
            }
            question_requests.append(question_request)

    # Assign locations to all requests
    for i, request in enumerate(question_requests):
        request["createItem"]["location"] = {"index": i}
        requests.append(request)

    # Execute batch update
    body = {"requests": requests}
    service.forms().batchUpdate(formId=form_id, body=body).execute()

    total_questions = len(subreddit_pages) * COMMENTS_PER_SUBREDDIT
    print(f"✅ Added {len(subreddit_pages)} subreddit pages with {total_questions} questions")

    return form_id


def save_evaluation_metadata(all_forms_data: List[Dict], subreddit_pages: List[Dict], output_dir: str):
    """Save evaluation metadata for analysis."""

    # Save individual form data
    all_form_ids = []
    all_public_urls = []

    for i, form_data in enumerate(all_forms_data):
        evaluation_file = os.path.join(output_dir, f'stage9.5_human_evaluation_part{i+1}.json')
        write_json_file(form_data, evaluation_file, pretty=True)
        all_form_ids.append(form_data['form_id'])
        all_public_urls.append(form_data['public_url'])
        print(f"  ✅ Saved metadata: {evaluation_file}")

    # Save combined summary
    combined_summary = {
        'metadata': {
            'total_forms': len(all_forms_data),
            'total_subreddits': len(subreddit_pages),
            'total_questions': len(subreddit_pages) * COMMENTS_PER_SUBREDDIT,
            'comments_per_subreddit': COMMENTS_PER_SUBREDDIT,
            'subreddits_per_form': SUBREDDITS_PER_FORM,
            'random_seed': RANDOM_SEED,
            'creation_date': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'forms': all_forms_data,
        'subreddit_pages': [
            {
                'subreddit': page['subreddit'],
                'rank': page['rank'],
                'jsd_score': page['jsd_score'],
                'num_rules': len(page['rules']),
                'num_comments_sampled': len(page['sampled_comments'])
            }
            for page in subreddit_pages
        ]
    }

    summary_file = os.path.join(output_dir, 'stage9.5_human_evaluation_summary.json')
    write_json_file(combined_summary, summary_file, pretty=True)
    print(f"  ✅ Saved summary: {summary_file}")

    return all_form_ids, all_public_urls


def main():
    """Main execution function."""
    logger = get_stage_logger(11, "human_evaluation")
    log_stage_start(logger, 11, "Human Evaluation from Final Dataset")

    start_time = time.time()

    try:
        print("🔬 Stage 11: Human Evaluation Setup")
        print(f"📊 Configuration:")
        print(f"  Comments per subreddit: {COMMENTS_PER_SUBREDDIT}")
        print(f"  Subreddits per form: {SUBREDDITS_PER_FORM}")
        print(f"  Random seed: {RANDOM_SEED}")

        # Load final dataset
        dataset = load_final_dataset(logger)
        if not dataset:
            logger.error("❌ Failed to load final dataset!")
            return 1

        total_subreddits = len(dataset['subreddits'])
        print(f"✅ Loaded dataset with {total_subreddits} subreddits")

        # Prepare subreddit pages
        print(f"\n🎲 Sampling {COMMENTS_PER_SUBREDDIT} comments per subreddit...")
        subreddit_pages = prepare_subreddit_pages(dataset)
        print(f"✅ Prepared {len(subreddit_pages)} subreddit pages")

        if len(subreddit_pages) == 0:
            logger.error("❌ No subreddit pages prepared!")
            return 1

        # Split into forms (alphabetically)
        form_chunks = []
        for i in range(0, len(subreddit_pages), SUBREDDITS_PER_FORM):
            chunk = subreddit_pages[i:i + SUBREDDITS_PER_FORM]
            form_chunks.append(chunk)

        print(f"\n📊 Splitting into {len(form_chunks)} forms:")
        for i, chunk in enumerate(form_chunks):
            first_sub = chunk[0]['subreddit']
            last_sub = chunk[-1]['subreddit']
            print(f"  Form {i+1}: {len(chunk)} subreddits (r/{first_sub} to r/{last_sub})")

        # Authenticate with Google
        print("\n🔐 Authenticating with Google APIs...")
        creds = authenticate()
        service = build('forms', 'v1', credentials=creds)
        print("✅ Authenticated successfully")

        # Create forms
        all_forms_data = []

        for form_index, chunk in enumerate(form_chunks):
            form_part = form_index + 1
            total_forms = len(form_chunks)

            print(f"\n📝 Creating Google Form {form_part}/{total_forms}...")
            print(f"  Subreddits: {len(chunk)}")
            print(f"  Questions: {len(chunk) * COMMENTS_PER_SUBREDDIT}")

            form_id = create_evaluation_form(service, chunk, form_part, total_forms)
            form_url = f"https://docs.google.com/forms/d/{form_id}/edit"
            public_url = f"https://docs.google.com/forms/d/{form_id}/viewform"

            form_data = {
                'form_id': form_id,
                'form_url': form_url,
                'public_url': public_url,
                'form_part': form_part,
                'total_forms': total_forms,
                'num_subreddits': len(chunk),
                'num_questions': len(chunk) * COMMENTS_PER_SUBREDDIT,
                'subreddits': [page['subreddit'] for page in chunk]
            }

            all_forms_data.append(form_data)

        # Save metadata
        print(f"\n💾 Saving evaluation metadata...")
        output_dir = os.path.join(PATHS['data'], 'evaluation')
        os.makedirs(output_dir, exist_ok=True)

        all_form_ids, all_public_urls = save_evaluation_metadata(all_forms_data, subreddit_pages, output_dir)

        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 Evaluation Summary")
        print(f"{'='*60}")
        print(f"Total subreddits: {len(subreddit_pages)}")
        print(f"Total questions: {len(subreddit_pages) * COMMENTS_PER_SUBREDDIT}")
        print(f"Number of forms: {len(all_forms_data)}")
        print(f"\n📝 Google Forms Created:")
        for i, public_url in enumerate(all_public_urls):
            print(f"  Form {i+1}: {public_url}")

        print(f"\n🚀 Next steps:")
        print(f"  1. Share the public URLs with human evaluators")
        print(f"  2. Collect responses from Google Forms")
        print(f"  3. Analyze agreement between embedding predictions and human labels")

        print(f"\n✅ Stage 11 completed successfully!")
        print(f"📁 Metadata saved to: {output_dir}")

        elapsed = time.time() - start_time
        log_stage_end(logger, 11, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        logger.error(f"❌ Stage 11 failed: {e}")
        import traceback
        traceback.print_exc()
        log_stage_end(logger, 11, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
