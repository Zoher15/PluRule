#!/usr/bin/env python3
"""
Simple pipeline runner for Reddit mod collection pipeline.

Usage:
  python run_pipeline.py           # Run full pipeline
  python run_pipeline.py 1         # Run stage 1 only
  python run_pipeline.py 3 5       # Run stages 3-5
  python run_pipeline.py status    # Show pipeline status
"""

import subprocess
import sys
import os
import time

from config import DATA_FLOW, print_pipeline_status, validate_stage_inputs


def get_stage_script(stage_num: int) -> str:
    """Get script path for a stage number."""
    stage_key = f"stage{stage_num}_" + list(DATA_FLOW.keys())[stage_num].split('_', 1)[1]
    stage_info = DATA_FLOW.get(stage_key)

    if not stage_info:
        return None

    script_name = stage_info['script']
    return os.path.join('scripts', script_name)


def run_stage(stage_num: int) -> bool:
    """Run a single stage and return success status."""
    script_path = get_stage_script(stage_num)

    if not script_path or not os.path.exists(script_path):
        print(f"❌ Stage {stage_num} script not found: {script_path}")
        return False

    stage_key = f"stage{stage_num}_" + list(DATA_FLOW.keys())[stage_num].split('_', 1)[1]
    stage_info = DATA_FLOW[stage_key]

    print(f"\n{'='*60}")
    print(f"🚀 Stage {stage_num}: {stage_info['name']}")
    print(f"📜 Script: {script_path}")
    print(f"{'='*60}")

    # Validate inputs
    valid, msg = validate_stage_inputs(stage_num)
    if not valid:
        print(f"❌ Input validation failed: {msg}")
        print("💡 Make sure previous stages have completed successfully")
        return False

    # Run the script
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False  # Show output in real-time
        )

        elapsed = time.time() - start_time
        print(f"\n✅ Stage {stage_num} completed successfully in {elapsed:.1f}s")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Stage {stage_num} failed after {elapsed:.1f}s (exit code: {e.returncode})")
        return False

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n⏹️ Stage {stage_num} interrupted after {elapsed:.1f}s")
        return False


def run_pipeline(start_stage: int = 0, end_stage: int = 13, stop_on_failure: bool = True) -> int:
    """Run multiple pipeline stages."""
    print("🔄 Reddit Mod Collection Pipeline")
    print(f"📊 Running stages {start_stage}-{end_stage}")

    pipeline_start = time.time()
    completed_stages = []
    failed_stages = []

    for stage_num in range(start_stage, end_stage + 1):
        success = run_stage(stage_num)

        if success:
            completed_stages.append(stage_num)
        else:
            failed_stages.append(stage_num)

            if stop_on_failure:
                print(f"\n🛑 Pipeline stopped due to stage {stage_num} failure")
                break

    pipeline_elapsed = time.time() - pipeline_start

    # Print final summary
    print(f"\n{'='*60}")
    print("📋 Pipeline Summary")
    print(f"{'='*60}")
    print(f"⏱️  Total time: {pipeline_elapsed:.1f}s")
    print(f"✅ Completed: {len(completed_stages)} stages")
    print(f"❌ Failed: {len(failed_stages)} stages")

    if completed_stages:
        print(f"📈 Completed stages: {', '.join(map(str, completed_stages))}")

    if failed_stages:
        print(f"📉 Failed stages: {', '.join(map(str, failed_stages))}")

    # Return appropriate exit code
    return 0 if not failed_stages else 1


def show_usage():
    """Show usage information."""
    print("Reddit Mod Collection Pipeline Runner")
    print("=" * 40)
    print()
    print("Usage:")
    print("  python run_pipeline.py                # Run full pipeline (stages 0-13)")
    print("  python run_pipeline.py 0              # Run stage 0 only")
    print("  python run_pipeline.py 3 5            # Run stages 3-5")
    print("  python run_pipeline.py status         # Show pipeline status")
    print("  python run_pipeline.py help           # Show this help")
    print()
    print("Available stages:")
    for i in range(0, 14):
        stage_key = f"stage{i}_" + list(DATA_FLOW.keys())[i].split('_', 1)[1]
        stage_info = DATA_FLOW.get(stage_key)
        if stage_info:
            print(f"  {i:2d}. {stage_info['name']}")
    print()


def main():
    """Main execution function."""
    args = sys.argv[1:]

    # Handle special commands
    if not args or (len(args) == 1 and args[0] in ['help', '--help', '-h']):
        show_usage()
        return 0

    if len(args) == 1 and args[0] == 'status':
        print_pipeline_status()
        return 0

    # Parse stage arguments
    try:
        if len(args) == 1:
            # Single stage or full pipeline
            if args[0].isdigit():
                stage_num = int(args[0])
                if 0 <= stage_num <= 13:
                    return 0 if run_stage(stage_num) else 1
                else:
                    print(f"❌ Invalid stage number: {stage_num} (must be 0-13)")
                    return 1
            else:
                print(f"❌ Invalid argument: {args[0]}")
                show_usage()
                return 1

        elif len(args) == 2:
            # Stage range
            start_stage = int(args[0])
            end_stage = int(args[1])

            if not (0 <= start_stage <= 13 and 0 <= end_stage <= 13):
                print(f"❌ Invalid stage range: {start_stage}-{end_stage} (must be 0-13)")
                return 1

            if start_stage > end_stage:
                print(f"❌ Invalid stage range: start ({start_stage}) > end ({end_stage})")
                return 1

            return run_pipeline(start_stage, end_stage)

        else:
            print(f"❌ Too many arguments: {args}")
            show_usage()
            return 1

    except ValueError:
        print(f"❌ Invalid stage numbers: {args}")
        show_usage()
        return 1

    # Default: run full pipeline
    return run_pipeline()


if __name__ == "__main__":
    exit(main())