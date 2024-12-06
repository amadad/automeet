from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from enum import Enum
import instructor
from openai import AsyncOpenAI
import json
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from datetime import datetime

# Configuration
API_BASE_URL = 'http://localhost:11434/v1'
API_KEY = 'ollama'
MODEL_NAME = "qwen2.5:32b"
TEMPERATURE = 0.1
MAX_TOKENS = 2000

console = Console()

# Initialize OpenAI client with instructor
client = instructor.from_openai(
    AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY),
    mode=instructor.Mode.JSON
)

class SubCategory(str, Enum):
    # Tasks
    ASSIGNED = "assigned"
    PROPOSED = "proposed"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    
    # Decisions
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    
    # Questions
    ASKED = "asked"
    ANSWERED = "answered"
    UNRESOLVED = "unresolved"
    IMPLIED = "implied"
    
    # Attendees
    NAMED = "named"
    ROLES = "roles"
    TEAMS = "teams"
    MENTIONED = "mentioned"
    
    # Deadlines
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    
    # Follow-ups
    MEETINGS = "meetings"
    REVIEWS = "reviews"
    DEPENDENCIES = "dependencies"
    DECISIONS = "decisions"
    
    # Risks
    TECHNICAL = "technical"
    TIMELINE = "timeline"
    STAKEHOLDER = "stakeholder"
    RESOURCE = "resource"

class InsightItem(BaseModel):
    """Represents a single extracted insight from the transcript."""
    point: str = Field(..., description="One-line summary of the insight")
    quote: str = Field(..., description="Exact supporting quote from the transcript")
    speaker: str = Field(..., description="Speaker identifier or name")
    subcategory: SubCategory = Field(..., description="Subcategory of the insight")

class MeetingInsight(BaseModel):
    """Holds categorized insights extracted from a meeting transcript."""
    tasks: List[InsightItem] = Field(default_factory=list, description="Action items and work to be done")
    decisions: List[InsightItem] = Field(default_factory=list, description="Decisions and agreements reached")
    questions: List[InsightItem] = Field(default_factory=list, description="Questions raised or concerns expressed")
    attendees: List[InsightItem] = Field(default_factory=list, description="Mentioned participants or roles")
    deadlines: List[InsightItem] = Field(default_factory=list, description="Stated time constraints or due dates")
    follow_ups: List[InsightItem] = Field(default_factory=list, description="Items requiring further action")
    risks: List[InsightItem] = Field(default_factory=list, description="Identified risks and concerns")

    def to_markdown(self) -> str:
        """Convert insights into a human-readable Markdown format."""
        md_lines = ["# Meeting Analysis Results\n"]
        sections = {
            "Tasks": self.tasks,
            "Decisions": self.decisions,
            "Questions": self.questions,
            "Attendees": self.attendees,
            "Deadlines": self.deadlines,
            "Follow-ups": self.follow_ups,
            "Risks": self.risks
        }
        
        for title, items in sections.items():
            md_lines.append(f"## {title}")
            if not items:
                md_lines.append("No items found.\n")
            else:
                for item in items:
                    md_lines.append(f"- **{item.point}**")
                    md_lines.append(f"  - Quote: \"{item.quote}\"")
                    md_lines.append(f"  - Speaker: {item.speaker}")
                    md_lines.append(f"  - Category: {item.subcategory}\n")
            md_lines.append("")
        
        return "\n".join(md_lines)

class OutputManager:
    """Manages file outputs and directories for storing results."""
    def __init__(self, transcript_name: str):
        self.transcript_name = transcript_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_output(self, content: str, stage: str) -> Path:
        """Save output content to a versioned directory."""
        output_dir = Path("output") / self.transcript_name / self.timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{stage}.md"
        output_path.write_text(content)
        return output_path

class TranscriptProcessor:
    """Handles the end-to-end process of analyzing a transcript."""
    def __init__(self, auto_mode: bool = False):
        self.auto_mode = auto_mode

    async def analyze_transcript(self, transcript: str) -> MeetingInsight:
        """
        Send the transcript to the model for analysis and extraction of insights.
        Returns a MeetingInsight object.
        """
        console.print("[cyan]Starting analysis...[/cyan]")
        try:
            insight = await client.chat.completions.create(
                model=MODEL_NAME,
                response_model=MeetingInsight,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at analyzing product development meeting transcripts.\n"
                            "Extract all relevant information according to the categories:\n\n"
                            "TASKS: Actions or work to be done.\n"
                            "DECISIONS: Agreements or choices made.\n"
                            "QUESTIONS: Any explicit questions or doubts raised.\n"
                            "ATTENDEES: Mentioned participants or teams.\n"
                            "DEADLINES: Mentioned due dates or timeframes.\n"
                            "FOLLOW-UPS: Future actions or meetings to revisit.\n"
                            "RISKS: Potential problems or constraints.\n\n"
                            "Rules:\n"
                            "1. Use exact quotes from the transcript.\n"
                            "2. Use actual names/roles if mentioned.\n"
                            "3. Provide a clear one-line summary for each point."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Analyze the following meeting transcript:\n\n{transcript}"
                    }
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            console.print("[green]Analysis complete.[/green]")
            return insight
        except Exception as e:
            console.print(f"[red]Error in analyze_transcript: {e}[/red]")
            raise

    async def improve_insights(self, previous: MeetingInsight, feedback: str, transcript: str) -> MeetingInsight:
        """
        Request improved insights from the model based on human feedback.
        Returns updated MeetingInsight.
        """
        console.print("[cyan]Improving insights based on feedback...[/cyan]")
        try:
            new_insight = await client.chat.completions.create(
                model=MODEL_NAME,
                response_model=MeetingInsight,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at improving meeting analysis.\n"
                            "Based on the previous analysis and the given feedback, do the following:\n"
                            "1. Keep good existing insights.\n"
                            "2. Add missing information if available.\n"
                            "3. Use exact quotes from the transcript.\n"
                            "4. Correct any misclassifications.\n"
                            "5. Follow the same categories and format as before."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Previous insights:\n{previous.model_dump_json(indent=2)}\n\n"
                            f"Feedback:\n{feedback}\n\n"
                            f"Original transcript:\n{transcript}\n\n"
                            "Please improve the analysis based on the feedback."
                        )
                    }
                ],
                temperature=TEMPERATURE
            )
            console.print("[green]Improvement complete.[/green]")
            return new_insight
        except Exception as e:
            console.print(f"[red]Error in improve_insights: {e}[/red]")
            raise

    async def process_transcript(self, transcript_path: Path, progress: Progress) -> MeetingInsight:
        """
        Orchestrate the full analysis process:
        1. Read transcript
        2. Analyze it
        3. Optionally handle human feedback
        4. Save results
        """
        transcript_name = transcript_path.stem
        output_manager = OutputManager(transcript_name)
        
        # Read transcript
        transcript = transcript_path.read_text().strip()
        if not transcript:
            console.print("[red]Error: Transcript is empty![/red]")
            return MeetingInsight()

        console.print(Panel.fit("[bold blue]Stage 1: Analysis[/bold blue]", border_style="blue"))
        analysis_task = progress.add_task("[cyan]Analyzing transcript...", total=1)
        
        # Perform initial analysis
        insight = await self.analyze_transcript(transcript)
        progress.update(analysis_task, completed=1, visible=False)
        
        # Save initial analysis result
        analysis_path = output_manager.save_output(insight.to_markdown(), "1_analysis")
        console.print(f"[green]✓ Analysis saved to:[/green] [bold blue]{analysis_path}[/bold blue]")
        
        # If human review is enabled and insights are empty, handle that
        if not self.auto_mode:
            insight = await self._handle_human_review(insight, transcript, transcript_name)
        
        # Save final output
        final_path = output_manager.save_output(insight.to_markdown(), "2_final_output")
        console.print(f"[green]✓ Final output saved to:[/green] [bold blue]{final_path}[/bold blue]")
        
        return insight

    async def _handle_human_review(
        self, insight: MeetingInsight, transcript: str, transcript_name: str
    ) -> MeetingInsight:
        """
        Handle human review step: if user rejects current insights, prompt for feedback and attempt improvements.
        """
        console.print(Panel.fit("[bold yellow]Human Review[/bold yellow]", border_style="yellow"))
        
        console.print("\n[bold]Current Insights:[/bold]")
        console.print_json(data=insight.model_dump())
        
        # If no insights found
        if all(len(getattr(insight, field)) == 0 for field in insight.model_fields):
            return await self._handle_empty_insights(insight, transcript, transcript_name)

        # Ask for approval
        approval = console.input("\n[bold yellow]Approve these insights? (y/n): [/bold yellow]")
        if approval.lower() == 'y':
            return insight
        
        # If not approved, request feedback and try improving
        console.print("[yellow]Please provide feedback for improvement:[/yellow]")
        feedback = console.input()
        return await self._handle_iteration(insight, feedback, transcript, transcript_name)

    async def _handle_empty_insights(
        self, insight: MeetingInsight, transcript: str, transcript_name: str
    ) -> MeetingInsight:
        """
        If no insights were extracted, allow user to:
        1. Provide feedback to try again
        2. Enter insights manually
        3. Accept empty
        """
        console.print("\n[red]No insights extracted. Options:[/red]")
        console.print("1. Provide feedback for another attempt")
        console.print("2. Enter insights manually")
        console.print("3. Accept empty insights")
        
        choice = console.input("\n[bold yellow]Enter choice (1-3): [/bold yellow]")
        
        if choice == "1":
            console.print("[yellow]Please provide feedback:[/yellow]")
            feedback = console.input()
            return await self._handle_iteration(insight, feedback, transcript, transcript_name)
        elif choice == "2":
            return self._handle_manual_entry()
        
        return insight

    async def _handle_iteration(
        self, insight: MeetingInsight, feedback: str, transcript: str, transcript_name: str
    ) -> MeetingInsight:
        """Re-run the analysis with given feedback to improve the insights."""
        console.print(Panel.fit("[bold red]Iteration Based on Feedback[/bold red]", border_style="red"))
        try:
            new_insight = await self.improve_insights(insight, feedback, transcript)
            output_manager = OutputManager(transcript_name)
            iteration_path = output_manager.save_output(new_insight.to_markdown(), "3_iteration")
            console.print(f"[green]✓ Iteration saved to:[/green] [bold blue]{iteration_path}[/bold blue]")
            return new_insight
        except Exception as e:
            console.print(f"[red]Error during iteration: {e}[/red]")
            return insight

    def _handle_manual_entry(self) -> MeetingInsight:
        """If no insights are extracted, the user can manually provide them."""
        console.print("[yellow]Enter insights manually (one per line, empty line to move to next category):[/yellow]")
        insight = MeetingInsight()
        
        for field in insight.model_fields:
            console.print(f"\n[bold]Enter {field} (Press Enter on empty line to skip):[/bold]")
            items = []
            while True:
                item = console.input()
                if not item:
                    break
                items.append(
                    InsightItem(
                        point=item,
                        quote="Manually entered",
                        speaker="Manual Entry",
                        subcategory=SubCategory.PROPOSED
                    )
                )
            setattr(insight, field, items)
        
        return insight

async def main():
    """Main entry point for processing a transcript."""
    console.print(Panel.fit("[bold green]Meeting Transcript Processor[/bold green]", border_style="green"))
    
    processor = TranscriptProcessor(auto_mode=False)
    transcript_dir = Path("transcripts")
    transcript_dir.mkdir(exist_ok=True)
    
    transcript_path = next(transcript_dir.glob("*.md"), None)
    if not transcript_path:
        console.print("[yellow]No transcripts found in /transcripts/ directory. Please add a .md file.[/yellow]")
        return
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        console.print(f"\n[bold blue]Processing: {transcript_path.name}[/bold blue]")
        await processor.process_transcript(transcript_path, progress)
        console.print("\n[bold green]✨ Transcript processed successfully![/bold green]")

if __name__ == "__main__":
    asyncio.run(main())