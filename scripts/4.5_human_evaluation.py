#!/usr/bin/env python3
"""
Stage 4.5: Human Evaluation Setup

This script creates a human evaluation survey to compare embedding-based rule matching
with human judgment. It samples matched comments from Stage 4 and creates Google Forms
for human evaluators to rate the rule matching accuracy.

Usage: python 4.5_human_evaluation.py
"""

import os
import sys
import json
import random
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

from config import PATHS
from utils.files import read_json_file, write_json_file, read_zst_lines, json_loads
from utils.logging import get_stage_logger, log_stage_start, log_stage_end

# Configuration
SAMPLES_PER_SUBREDDIT = 1
SUBREDDITS_TO_EVALUATE = None  # Set to None to evaluate all ranked subreddits
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

def load_matched_comments(subreddit_name: str) -> List[Dict[str, Any]]:
    """Load matched comments for a subreddit."""
    comments_file = os.path.join(PATHS['matched_comments_sample'], f"{subreddit_name}_match.jsonl.zst")

    if not os.path.exists(comments_file):
        return []

    comments = []
    for line in read_zst_lines(comments_file):
        try:
            comment = json_loads(line)
            comments.append(comment)
        except Exception:
            continue

    return comments

def load_subreddit_rules(subreddit_name: str) -> List[Dict[str, Any]]:
    """Load rules for a specific subreddit from Stage 2 output."""
    rules_file = os.path.join(PATHS['data'], 'stage2_top_2000_sfw_subreddits.json')

    if not os.path.exists(rules_file):
        return []

    data = read_json_file(rules_file)

    for entry in data['subreddits']:
        subreddit_data = entry['subreddit']
        if subreddit_data.get('display_name', '').lower() == subreddit_name.lower():
            rules = []
            for rule in entry.get('rules', []):
                cleaned_rule = {
                    'rule_index': rule.get('rule_index', 0),
                    'short_name': rule.get('short_name_clean', ''),
                    'description': rule.get('description_clean', ''),
                    'combined_text': f"Short Name: {rule.get('short_name_clean', '')}\nDescription: {rule.get('description_clean', '')}"
                }
                if rule.get('violation_reason_clean'):
                    cleaned_rule['combined_text'] += f"\nViolation Reason: {rule.get('violation_reason_clean', '')}"
                cleaned_rule['combined_text'] = cleaned_rule['combined_text'].strip()
                rules.append(cleaned_rule)
            return rules

    return []

def sample_comments_balanced(all_comments: Dict[str, List[Dict]], sample_size: int) -> List[Dict[str, Any]]:
    """Sample comments balanced across subreddits."""
    random.seed(RANDOM_SEED)

    # Calculate samples per subreddit
    total_comments = sum(len(comments) for comments in all_comments.values())
    sampled_comments = []

    for subreddit, comments in all_comments.items():
        # Proportional sampling based on available comments
        subreddit_proportion = len(comments) / total_comments
        subreddit_sample_size = max(1, int(sample_size * subreddit_proportion))

        # Don't sample more than available
        subreddit_sample_size = min(subreddit_sample_size, len(comments))

        # Random sample
        subreddit_sample = random.sample(comments, subreddit_sample_size)

        # Add subreddit info to each comment
        for comment in subreddit_sample:
            comment['evaluation_subreddit'] = subreddit

        sampled_comments.extend(subreddit_sample)
        print(f"  📊 Sampled {len(subreddit_sample)} comments from r/{subreddit}")

    # If we're under the target, randomly fill from largest subreddit
    if len(sampled_comments) < sample_size:
        largest_subreddit = max(all_comments.keys(), key=lambda k: len(all_comments[k]))
        remaining_needed = sample_size - len(sampled_comments)

        # Get comments not already sampled
        sampled_ids = {c['id'] for c in sampled_comments if c['evaluation_subreddit'] == largest_subreddit}
        available_comments = [c for c in all_comments[largest_subreddit] if c['id'] not in sampled_ids]

        if available_comments:
            additional_sample = random.sample(available_comments, min(remaining_needed, len(available_comments)))
            for comment in additional_sample:
                comment['evaluation_subreddit'] = largest_subreddit
            sampled_comments.extend(additional_sample)
            print(f"  📊 Added {len(additional_sample)} more comments from r/{largest_subreddit}")

    # Shuffle the final sample
    random.shuffle(sampled_comments)
    return sampled_comments[:sample_size]

def prepare_evaluation_data(sampled_comments: List[Dict], subreddit_rules: Dict[str, List[Dict]]) -> List[Dict]:
    """Prepare evaluation questions with answer choices."""
    evaluation_questions = []

    for i, comment in enumerate(sampled_comments, 1):
        subreddit = comment['evaluation_subreddit']
        rules = subreddit_rules.get(subreddit, [])

        if not rules:
            continue

        # Create question
        question = {
            'question_id': i,
            'comment_id': comment['id'],
            'subreddit': subreddit,
            'question_text': f"Question {i}: Which rule is this moderator's comment referring to? Please do not rely solely on the rule numbers for matching, as they may have changed since the comment was made. Your task is to select the EXACT rule that the moderator was referring to. If multiple rules apply, please select all of them. The 'other' field is a notepad for errors, bugs, feedback, or interesting observations that you feel the need to mark and revisit later.",
            'comment_body': comment.get('body_clean', comment.get('body', '')),
            'embedding_prediction': comment.get('matched_rule', {}),
            'answer_choices': []
        }

        # Add rule options
        for rule in rules:
            choice = {
                'rule_index': rule['rule_index'],
                'choice_text': f"Rule {rule['rule_index']}: {rule['combined_text']}"
            }
            question['answer_choices'].append(choice)

        # Add "None of the above" option
        question['answer_choices'].append({
            'rule_index': -1,
            'choice_text': "None of the above rules apply"
        })

        evaluation_questions.append(question)

    return evaluation_questions

def create_google_form_data(evaluation_questions: List[Dict], form_part: int = 1, total_forms: int = 1) -> Dict:
    """Prepare data structure for Google Form creation."""
    title = 'Reddit Moderation Rule Matching Evaluation'
    if total_forms > 1:
        title += f' (Part {form_part} of {total_forms})'

    description = (
        'Please help evaluate how well our AI system matches moderator comments to subreddit rules. '
        'For each question, read the moderator comment and select which rule it most likely refers to. '
        f'This form contains {len(evaluation_questions)} questions.'
    )
    if total_forms > 1:
        description += f' This is part {form_part} of {total_forms} forms.'

    form_data = {
        'title': title,
        'description': description,
        'questions': evaluation_questions,
        'metadata': {
            'total_questions': len(evaluation_questions),
            'subreddits_evaluated': list(set(q['subreddit'] for q in evaluation_questions)),
            'created_date': None,  # Will be filled when form is created
            'sample_seed': RANDOM_SEED
        }
    }

    return form_data

def create_evaluation_form(service, evaluation_questions: List[Dict], form_part: int = 1, total_forms: int = 1) -> str:
    """Create the actual Google Form with evaluation questions."""
    # Step 1: Create the basic form with our title
    title = "Reddit Moderation Rule Matching Evaluation"
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

    # Step 2: Add description and questions via batchUpdate
    requests = [
        # Update form title and description
        {
            "updateFormInfo": {
                "info": {
                    "title": title,
                    "description": (
                        "Please help evaluate how well our AI system matches moderator comments to subreddit rules. "
                        "For each question, read the moderator comment and select which rule it most likely refers to. "
                        f"This form contains {len(evaluation_questions)} questions."
                        + (f" This is part {form_part} of {total_forms} forms." if total_forms > 1 else "")
                    )
                },
                "updateMask": "title,description"
            }
        }
    ]

    # Build all requests first, then assign locations
    question_requests = []

    for question_index, question_data in enumerate(evaluation_questions):
        # Clean question title (no newlines allowed in titles)
        question_title = question_data['question_text']

        # Detailed context in description (newlines allowed)
        question_description = f"Subreddit: r/{question_data['subreddit']}\n\nModerator Comment:\n\"{question_data['comment_body']}\""

        # Create choice options (no newlines allowed in value field)
        choice_options = []
        for choice in question_data['answer_choices']:
            if choice['rule_index'] == -1:
                # "None of the above" option - no formatting needed
                choice_options.append({
                    "value": choice['choice_text']
                })
            else:
                # Clean up the choice text - replace newlines with spaces
                clean_choice_text = choice['choice_text'].replace('\n', ' ').replace('\r', ' ')
                # Also compress multiple spaces into single spaces
                clean_choice_text = ' '.join(clean_choice_text.split())

                # Format with visual separation (newlines don't work in Google Forms)
                formatted_text = clean_choice_text

                # Use clear visual separators with emojis and caps
                formatted_text = formatted_text.replace('Short Name:', '📌 SHORT NAME:')
                formatted_text = formatted_text.replace('Description:', '  📝 DESCRIPTION:')
                formatted_text = formatted_text.replace('Violation Reason:', '  ⚠️ VIOLATION:')

                choice_options.append({
                    "value": formatted_text
                })

        # Add an "Other" option with text field for additional comments
        choice_options.append({
            "isOther": True
        })

        # Create the question request
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

        # Add a page break after every group of 3 questions (except the last one)
        if (question_index + 1) % 3 == 0 and question_index < len(evaluation_questions) - 1:
            next_question_num = question_index + 2
            page_break_request = {
                "createItem": {
                    "item": {
                        "title": f"Questions {next_question_num}-{min(next_question_num + 2, len(evaluation_questions))}",
                        "pageBreakItem": {}
                    }
                }
            }
            question_requests.append(page_break_request)

    # Now assign locations to all requests (starting from index 0)
    for i, request in enumerate(question_requests):
        request["createItem"]["location"] = {"index": i}  # Start from 0, no offset needed
        requests.append(request)

    # Execute all requests
    body = {"requests": requests}
    service.forms().batchUpdate(formId=form_id, body=body).execute()
    print(f"✅ Added {len(evaluation_questions)} questions to form")
    return form_id

def save_evaluation_data(evaluation_data: Dict, output_file: str):
    """Save evaluation data for form creation and analysis."""
    write_json_file(evaluation_data, output_file, pretty=True)
    print(f"✅ Saved evaluation data to {output_file}")

def main():
    """Main function for Stage 4.5."""
    logger = get_stage_logger(4.5, "human_evaluation")
    log_stage_start(logger, 4.5, "Human Evaluation Setup")

    start_time = time.time()

    try:
        print("🔬 Stage 4.5: Setting up Human Evaluation")
        print(f"📊 Samples per subreddit: {SAMPLES_PER_SUBREDDIT}")

        # Determine which subreddits to evaluate
        if SUBREDDITS_TO_EVALUATE is None:
            # Load all ranked subreddits from the updated summary
            summary_file = os.path.join(PATHS['data'], 'stage4_matching_summary.json')
            if os.path.exists(summary_file):
                summary_data = read_json_file(summary_file)
                # Only include subreddits that have been properly ranked (not 999999)
                subreddits_to_evaluate = [s['subreddit'] for s in summary_data['subreddit_stats'] if s.get('rank', 999999) != 999999]
                print(f"🎯 Evaluating all {len(subreddits_to_evaluate)} ranked subreddits")
            else:
                logger.error("❌ No ranked subreddits summary found!")
                return 1
        else:
            subreddits_to_evaluate = SUBREDDITS_TO_EVALUATE
            print(f"🎯 Evaluating specified subreddits: {', '.join(subreddits_to_evaluate)}")

        # Load matched comments from each subreddit
        all_comments = {}
        subreddit_rules = {}

        for subreddit in subreddits_to_evaluate:
            print(f"\n📂 Loading data for r/{subreddit}...")

            # Load matched comments
            comments = load_matched_comments(subreddit)
            if comments:
                all_comments[subreddit] = comments
                print(f"  ✅ Loaded {len(comments)} matched comments")
            else:
                print(f"  ⚠️  No matched comments found for r/{subreddit}")
                continue

            # Load rules
            rules = load_subreddit_rules(subreddit)
            if rules:
                subreddit_rules[subreddit] = rules
                print(f"  ✅ Loaded {len(rules)} rules")
            else:
                print(f"  ⚠️  No rules found for r/{subreddit}")

        if not all_comments:
            logger.error("❌ No matched comments found for any subreddit!")
            return 1

        # Sample comments using SAMPLES_PER_SUBREDDIT
        print(f"\n🎲 Sampling {SAMPLES_PER_SUBREDDIT} comments per subreddit...")
        sampled_comments = []
        random.seed(RANDOM_SEED)

        for subreddit, comments in all_comments.items():
            # Sample SAMPLES_PER_SUBREDDIT comments from each subreddit
            sample_size = min(SAMPLES_PER_SUBREDDIT, len(comments))
            subreddit_sample = random.sample(comments, sample_size)

            # Add subreddit info to each comment
            for comment in subreddit_sample:
                comment['evaluation_subreddit'] = subreddit

            sampled_comments.extend(subreddit_sample)
            print(f"  📊 Sampled {len(subreddit_sample)} comments from r/{subreddit}")

        print(f"✅ Selected {len(sampled_comments)} comments for evaluation")

        # Prepare evaluation questions
        print("\n📝 Preparing evaluation questions...")
        evaluation_questions = prepare_evaluation_data(sampled_comments, subreddit_rules)
        print(f"✅ Created {len(evaluation_questions)} evaluation questions")

        # Split questions into chunks for multiple forms (20 pages = 60 questions per form)
        questions_per_page = 3
        pages_per_form = 20
        questions_per_form = questions_per_page * pages_per_form  # 60 questions per form

        question_chunks = []
        for i in range(0, len(evaluation_questions), questions_per_form):
            chunk = evaluation_questions[i:i + questions_per_form]
            question_chunks.append(chunk)

        print(f"📊 Splitting into {len(question_chunks)} forms with up to {questions_per_form} questions each")

        # Authenticate once for all forms
        print("\n🔐 Authenticating with Google APIs...")
        creds = authenticate()
        service = build('forms', 'v1', credentials=creds)

        # Create multiple forms
        all_forms_data = []
        for form_index, question_chunk in enumerate(question_chunks):
            form_part = form_index + 1
            total_forms = len(question_chunks)

            print(f"\n📝 Creating Google Form {form_part}/{total_forms} ({len(question_chunk)} questions)...")

            form_id = create_evaluation_form(service, question_chunk, form_part, total_forms)
            form_url = f"https://docs.google.com/forms/d/{form_id}/edit"
            public_url = f"https://docs.google.com/forms/d/{form_id}/viewform"

            # Create form data structure for this form
            form_data = create_google_form_data(question_chunk, form_part, total_forms)
            form_data['metadata']['created_date'] = time.time()
            form_data['metadata']['form_id'] = form_id
            form_data['metadata']['form_url'] = form_url
            form_data['metadata']['public_url'] = public_url
            form_data['metadata']['form_part'] = form_part
            form_data['metadata']['total_forms'] = total_forms

            all_forms_data.append(form_data)

        # Save evaluation data for all forms
        output_dir = os.path.join(PATHS['data'], 'evaluation')
        os.makedirs(output_dir, exist_ok=True)

        # Save individual form data and collect summary
        all_form_ids = []
        all_public_urls = []

        for i, form_data in enumerate(all_forms_data):
            evaluation_file = os.path.join(output_dir, f'stage4.5_human_evaluation_data_part{i+1}.json')
            save_evaluation_data(form_data, evaluation_file)
            all_form_ids.append(form_data['metadata']['form_id'])
            all_public_urls.append(form_data['metadata']['public_url'])

        # Save combined summary
        combined_summary = {
            'total_forms': len(all_forms_data),
            'total_questions': len(evaluation_questions),
            'questions_per_form': [len(form_data['questions']) for form_data in all_forms_data],
            'all_forms': all_forms_data
        }
        summary_file = os.path.join(output_dir, 'stage4.5_human_evaluation_summary.json')
        save_evaluation_data(combined_summary, summary_file)

        # Print summary
        print(f"\n📊 Evaluation Summary:")
        print(f"  Total questions: {len(evaluation_questions)}")
        print(f"  Number of forms: {len(all_forms_data)}")
        for i, form_data in enumerate(all_forms_data):
            print(f"  Form {i+1}: {len(form_data['questions'])} questions")

        subreddit_counts = defaultdict(int)
        for q in evaluation_questions:
            subreddit_counts[q['subreddit']] += 1

        print(f"\n📋 Subreddit distribution:")
        for subreddit, count in sorted(subreddit_counts.items()):
            print(f"  r/{subreddit}: {count} questions")

        print(f"\n✅ Stage 4.5 completed successfully!")
        print(f"📁 Evaluation data saved to: {output_dir}")
        print(f"\n📝 Google Forms Created ({len(all_forms_data)} forms):")
        for i, (form_id, public_url) in enumerate(zip(all_form_ids, all_public_urls)):
            print(f"  Form {i+1}: {public_url}")

        print(f"\n🚀 Next steps:")
        print(f"  1. Share the public URLs with human evaluators")
        print(f"  2. Collect responses from all forms")
        print(f"  3. Analyze agreement between embedding and human ratings")

        log_stage_end(logger, 4.5, success=True, elapsed_time=time.time() - start_time)
        return 0

    except Exception as e:
        logger.error(f"❌ Stage 4.5 failed: {e}")
        log_stage_end(logger, 4.5, success=False, elapsed_time=time.time() - start_time)
        return 1

if __name__ == "__main__":
    import time
    exit(main())