import os
import re
from typing import Any
from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, router, start

from researcher.crew import researcher
from researcher.pdf_generator_professional import convert_report_to_pdf_professional


class ReportState(BaseModel):
    """State for the report refinement flow."""
    topic: str = ""
    iteration: int = 0
    max_iterations: int = 3
    quality_threshold: float = 75.0
    current_report: str = ""
    critique: str = ""
    quality_score: float = 0.0
    critique_history: list[dict] = []
    final_report: str = ""
    should_stop: bool = False


class CritiqueFlow(Flow[ReportState]):
    """Flow that iteratively improves a report through critique and refinement."""

    @start()
    def setup(self):
        """Initialize the flow state."""
        self.state.topic = "AI and Machine Learning in Healthcare 2024"
        self.state.iteration = 0
        self.state.max_iterations = 3
        self.state.quality_threshold = 75.0
        self.state.critique_history = []

    @listen(setup)
    def generate_initial_report(self):
        """Generate the initial report using the researcher crew."""
        print(f"\n{'='*60}")
        print(f"ITERATION {self.state.iteration + 1}/{self.state.max_iterations}")
        print(f"{'='*60}\n")

        crew_instance = researcher()
        result = crew_instance.crew().kickoff(inputs={"topic": self.state.topic})
        self.state.current_report = result.raw
        self.state.iteration += 1

    @listen(generate_initial_report)
    def critique_report(self):
        """Critique the current report."""
        print("\n📋 Running critique evaluation...\n")

        crew_instance = researcher()
        critique_result = crew_instance.crew().kickoff(inputs={
            "topic": self.state.topic,
            "report": self.state.current_report
        })
        self.state.critique = critique_result.raw

        # Extract quality score from critique
        quality_score = self._extract_quality_score(self.state.critique)
        self.state.quality_score = quality_score

        print(f"\n✓ Report Quality Score: {quality_score:.1f}/100")

        # Store critique in history
        self.state.critique_history.append({
            "iteration": self.state.iteration,
            "score": quality_score,
            "critique": self.state.critique
        })

    @router(critique_report)
    def decide_next_action(self):
        """Route based on quality score and iteration count."""
        score_met = self.state.quality_score >= self.state.quality_threshold
        iterations_left = self.state.iteration < self.state.max_iterations

        if score_met:
            print(f"\n[OK] Quality threshold ({self.state.quality_threshold}) reached!")
            return "publish"
        elif not iterations_left:
            print(f"\n[WARN] Max iterations ({self.state.max_iterations}) reached")
            return "publish"
        else:
            print(f"\n[RETRY] Score below threshold. Iterating (attempt {self.state.iteration + 1}/{self.state.max_iterations})")
            return "refine"

    @listen("refine")
    def refine_report(self):
        """Refine the report based on critique."""
        print("\n[REFINE] Refining report based on feedback...\n")

        # Extract improvement suggestions from critique
        improvement_prompt = self._extract_improvement_prompt(self.state.critique)

        crew_instance = researcher()

        refined_result = crew_instance.crew().kickoff(inputs={
            "topic": self.state.topic,
            "previous_report": self.state.current_report,
            "suggestions": self._extract_improvement_prompt(self.state.critique)
        })
        self.state.current_report = refined_result.raw

        # Re-critique the refined report
        print("\n📋 Running critique on refined report...\n")
        critique_result = crew_instance.crew().kickoff(inputs={
            "topic": self.state.topic,
            "report": refined_result.raw
        })
        self.state.critique = critique_result.raw
        quality_score = self._extract_quality_score(self.state.critique)
        self.state.quality_score = quality_score

        print(f"\n✓ Refined Report Quality Score: {quality_score:.1f}/100")

        self.state.critique_history.append({
            "iteration": self.state.iteration,
            "score": quality_score,
            "critique": self.state.critique
        })

        self.state.iteration += 1

    @listen("publish")
    def finalize_report(self):
        """Finalize and save the best report."""
        print(f"\n{'='*60}")
        print("✨ FINAL REPORT SUMMARY")
        print(f"{'='*60}\n")

        # Show iteration history
        print("📊 Quality Scores Across Iterations:")
        for entry in self.state.critique_history:
            print(f"  Iteration {entry['iteration']}: {entry['score']:.1f}/100")

        # Select best report
        best_critique = max(self.state.critique_history, key=lambda x: x['score'])
        print(f"\n🏆 Best Score: {best_critique['score']:.1f}/100 (Iteration {best_critique['iteration']})")

        self.state.final_report = self.state.current_report

        # Save detailed critique history
        self._save_critique_summary()

        # Generate PDF versions
        self._generate_pdfs()

    def _extract_quality_score(self, critique: str) -> float:
        """Extract average quality score from critique text using XML first."""
        # Try XML tag first (most reliable)
        xml_match = re.search(r'<score>([\d]+(?:\.[\d]+)?)</score>', critique, re.IGNORECASE)
        if xml_match:
            try:
                return min(100.0, max(0.0, float(xml_match.group(1))))
            except ValueError:
                pass

        # Look for "AVERAGE SCORE: X.X/100"
        patterns = [
            r"Average score[:\s]+(\d+(?:\.\d+)?)/100",
            r"Overall.*?(\d+(?:\.\d+)?)/100",
            r"Total.*?(\d+(?:\.\d+)?)/100",
        ]

        for pattern in patterns:
            match = re.search(pattern, critique, re.IGNORECASE)
            if match:
                try:
                    return min(100.0, max(0.0, float(match.group(1))))
                except ValueError:
                    continue

        # Fallback: extract dimension scores and average
        dimension_pattern = r"(?:CORRECTNESS|CLARITY|DEPTH|ACTIONABILITY|ENGAGEMENT|SOURCE ATTRIBUTION)[:\s]+(\d+)/100"
        scores = re.findall(dimension_pattern, critique, re.IGNORECASE)
        if scores:
            return sum(int(s) for s in scores) / len(scores)

        return 70.0

    def _extract_improvement_prompt(self, critique: str) -> str:
        """Extract improvement guidance from critique using XML first."""
        # Try XML tag first
        xml_match = re.search(r'<suggestions>(.*?)</suggestions>', critique, re.IGNORECASE | re.DOTALL)
        if xml_match:
            return xml_match.group(1).strip()

        # Fallback to section headers
        patterns = [
            r"(?:IMPROVEMENT SUGGESTIONS|Priority areas|Next steps)[:\s]*(.*?)(?:AVERAGE SCORE|<score>|$)",
            r"(?:Rewrite|Improvement) prompt[:\s]*(.*?)(?:Overall|Confidence|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, critique, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:1000]

        return critique[:1000]

    def _save_critique_summary(self):
        """Save critique history to file."""
        summary_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Report', 'critique_history.md')
        )
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)

        with open(summary_path, 'w') as f:
            f.write("# Report Critique History\n\n")
            f.write(f"Topic: {self.state.topic}\n\n")

            for entry in self.state.critique_history:
                f.write(f"## Iteration {entry['iteration']}\n")
                f.write(f"**Quality Score: {entry['score']:.1f}/100**\n\n")
                f.write(f"### Critique\n{entry['critique']}\n\n")
                f.write("---\n\n")

        print(f"\n💾 Critique history saved to: {summary_path}")

    def _generate_pdfs(self):
        """Generate PDF versions of the reports."""
        report_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Report')
        )

        # Save markdown report first
        report_md_path = os.path.join(report_dir, 'report.md')
        os.makedirs(report_dir, exist_ok=True)

        try:
            # Write the final report to markdown before conversion
            with open(report_md_path, 'w', encoding='utf-8') as f:
                f.write(self.state.final_report)

            # Generate PDF from markdown report with critique history
            critique_history = [{'iteration': e['iteration'], 'score': e['score'], 'status': ''} 
                              for e in self.state.critique_history]
            pdf_path = convert_report_to_pdf_enhanced(report_md_path, topic=self.state.topic, 
                                                     critique_history=critique_history)
            print(f"✓ Report PDF: {pdf_path}")
        except Exception as e:
            print(f"[WARN] Failed to generate report PDF: {e}")


def run_critique_flow(topic: str = None, max_iterations: int = 3, threshold: float = 75.0):
    """Run the critique flow with specified parameters."""
    flow = CritiqueFlow()

    if topic:
        flow.state.topic = topic

    flow.state.max_iterations = max_iterations
    flow.state.quality_threshold = threshold

    result = flow.kickoff()
    return result
