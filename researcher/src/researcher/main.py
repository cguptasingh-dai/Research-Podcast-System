from datetime import datetime
import os
import researcher.crew as crew_module
from researcher.crew import researcher
from researcher.report_refinement import run_report_refinement


def run():
    """
    Run the crew - uses refine_report internally.
    """
    # Direct to the stable refine_report workflow
    refine_report()


def refine_report():
    """
    Direct 3-iteration refinement: Report Writer ↔ Critic Agent
    """
    topic = input("Enter research topic: ").strip()
    while not topic:
        print("[ERROR] Topic cannot be empty")
        topic = input("Enter research topic: ").strip()

    print(f"\n[START] Starting 3-iteration report refinement for: {topic}")
    print("[INFO] Researcher agent will conduct real web research\n")

    research_findings = None  # Let researcher agent do real search

    try:
        print(f"\n{'='*70}")
        print(f"Starting report refinement process...")
        print(f"{'='*70}\n")

        summary = run_report_refinement(
            research_findings=research_findings,
            topic=topic,
            num_iterations=2
        )

        print(f"\n{'='*70}")
        print(f"[OK] REFINEMENT COMPLETE")
        print(f"{'='*70}\n")
        print(f"Topic: {summary['topic']}")
        print(f"Best Score: {summary['best_score']:.1f}/100")
        print(f"Best Iteration: {summary['best_iteration']}\n")

        # Generate safe topic name for display
        safe_topic = summary['topic'].replace(' ', '_').replace('/', '_').lower()
        if not safe_topic:
            safe_topic = 'report'

        print(f"PDF generated: Report/{safe_topic}.pdf\n")

    except ValueError as e:
        print(f"[ERROR] Configuration issue: {e}")
        print("[INFO] Please check your API keys and environment setup")
        raise
    except Exception as e:
        print(f"[ERROR] Report refinement failed: {e}")
        import traceback
        traceback.print_exc()
        raise
