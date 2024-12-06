from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from openai import AsyncOpenAI
import json
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from datetime import datetime

console = Console()

class SubCategory(str, Enum):
    # Tasks
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    
    # Decisions
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    
    # Questions
    ASKED = "asked"
    ANSWERED = "answered"
    UNRESOLVED = "unresolved"
    
    # Attendees
    NAMED = "named"
    ROLES = "roles"
    TEAMS = "teams"
    
    # Deadlines
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    
    # Follow-ups
    MEETINGS = "meetings"
    REVIEWS = "reviews"
    DEPENDENCIES = "dependencies"
    
    # Risks
    TECHNICAL = "technical"
    TIMELINE = "timeline"
    STAKEHOLDER = "stakeholder"

class InsightItem(BaseModel):
    """Single insight item with supporting evidence."""
    point: str = Field(..., description="Clear one-line summary of the point")
    quote: str = Field(..., description="Exact quote from transcript supporting this point")
    speaker: str = Field(..., description="Speaker identifier")
    subcategory: SubCategory = Field(..., description="Category of this insight")

class MeetingInsight(BaseModel):
    """Structured meeting insights with supporting quotes."""
    tasks: List[InsightItem] = Field(default_factory=list, description="Action items and work to be done")
    decisions: List[InsightItem] = Field(default_factory=list, description="Decisions and agreements reached")
    questions: List[InsightItem] = Field(default_factory=list, description="Questions raised or concerns expressed")
    attendees: List[InsightItem] = Field(default_factory=list, description="Meeting participants mentioned")
    deadlines: List[InsightItem] = Field(default_factory=list, description="Time constraints and due dates")
    follow_ups: List[InsightItem] = Field(default_factory=list, description="Items requiring further action")
    risks: List[InsightItem] = Field(default_factory=list, description="Identified risks and concerns")

    def to_markdown(self) -> str:
        """Convert insights to markdown format."""
        md = ["# Meeting Analysis Results\n"]
        
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
            md.append(f"## {title}")
            if not items:
                md.append("No items found.\n")
            else:
                for item in items:
                    md.append(f"- **{item.point}**")
                    md.append(f"  - Quote: \"{item.quote}\"")
                    md.append(f"  - Speaker: {item.speaker}")
                    md.append(f"  - Category: {item.subcategory}\n")
            md.append("")
        
        return "\n".join(md)

class TranscriptProcessor:
    """Processes meeting transcripts into structured insights."""
    
    def __init__(self, auto_mode: bool = False):
        self.auto_mode = auto_mode
        self.client = AsyncOpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama'
        )

    def _preprocess_transcript(self, transcript: str) -> str:
        """Preprocess the transcript text for better analysis."""
        # Remove excessive whitespace
        transcript = ' '.join(transcript.split())
        
        # Try to identify and format speaker segments
        lines = transcript.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Look for common timestamp patterns
            if any(time_indicator in line.lower() for time_indicator in ['am', 'pm', ':']) and len(line) < 20:
                # This is likely a timestamp line
                continue
                
            # Look for speaker indicators
            if line.strip().startswith('Speaker') or 'Today' in line:
                formatted_lines.append(f"\n{line.strip()}")
            else:
                formatted_lines.append(line.strip())
        
        # Join lines back together
        processed_transcript = ' '.join(formatted_lines)
        
        # Remove any duplicate whitespace
        processed_transcript = ' '.join(processed_transcript.split())
        
        return processed_transcript

    async def analyze_transcript(self, transcript: str) -> MeetingInsight:
        """Analyze transcript and extract structured insights."""
        try:
            system_prompt = """You are an expert at analyzing unformatted meeting transcripts.
            The transcript may contain timestamps and speaker indicators like 'Speaker 1', 'Today', etc.
            Focus on the actual content and conversation flow to extract key information.
            
            Extract key information and insights from the transcript, focusing on:
            
            1. Tasks and action items (look for mentions of work to be done, assignments)
            2. Decisions made (look for agreements, conclusions)
            3. Important questions raised (including both asked and answered)
            4. Meeting attendees (look for names, roles mentioned)
            5. Deadlines mentioned (any time-related commitments)
            6. Required follow-ups (future meetings, pending items)
            7. Potential risks (technical issues, concerns, blockers)
            
            For each insight:
            - Provide a clear summary
            - Include the exact supporting quote (even if it spans multiple speakers)
            - Identify the speaker (use 'Speaker 1', etc. if that's how they're labeled)
            - Categorize appropriately using the specified subcategories
            
            Format the response as a JSON object with these categories:
            {
                "tasks": [{"point": "", "quote": "", "speaker": "", "subcategory": "proposed"}],
                "decisions": [{"point": "", "quote": "", "speaker": "", "subcategory": "approved"}],
                "questions": [{"point": "", "quote": "", "speaker": "", "subcategory": "asked"}],
                "attendees": [{"point": "", "quote": "", "speaker": "", "subcategory": "named"}],
                "deadlines": [{"point": "", "quote": "", "speaker": "", "subcategory": "immediate"}],
                "follow_ups": [{"point": "", "quote": "", "speaker": "", "subcategory": "meetings"}],
                "risks": [{"point": "", "quote": "", "speaker": "", "subcategory": "technical"}]
            }"""

            response = await self.client.chat.completions.create(
                model="qwen2.5:32b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please analyze this transcript and extract insights:\n\n{transcript}"}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            result = json.loads(response.choices[0].message.content)
            
            # Add default subcategories if missing
            for category in ['tasks', 'decisions', 'questions', 'attendees', 'deadlines', 'follow_ups', 'risks']:
                if category not in result:
                    result[category] = []
                for item in result[category]:
                    if 'subcategory' not in item:
                        item['subcategory'] = {
                            'tasks': 'proposed',
                            'decisions': 'approved',
                            'questions': 'asked',
                            'attendees': 'named',
                            'deadlines': 'immediate',
                            'follow_ups': 'meetings',
                            'risks': 'technical'
                        }[category]

            # Validate and create MeetingInsight object
            return MeetingInsight.model_validate(result)
                
        except Exception as e:
            console.print(f"[red]Error in analyze_transcript: {str(e)}[/red]")
            return MeetingInsight()

    async def improve_insights(self, previous: MeetingInsight, feedback: str, transcript: str) -> MeetingInsight:
        """Improve insights based on feedback."""
        try:
            response = await self.client.chat.completions.create(
                model="qwen2.5:32b",
                messages=[
                    {"role": "system", "content": """You are an expert at improving meeting analysis.
                    Enhance the existing insights based on feedback while following these rules:
                    1. Keep valid existing insights
                    2. Add missing information
                    3. Use exact quotes from the transcript
                    4. Use actual names and roles when mentioned
                    5. Ensure all points have supporting evidence"""},
                    {"role": "user", "content": f"""Previous insights:\n{previous.model_dump_json(indent=2)}
                    
                    Feedback:\n{feedback}
                    
                    Original transcript:\n{transcript}
                    
                    Provide improved insights addressing the feedback."""}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return MeetingInsight.model_validate(result)
            
        except Exception as e:
            console.print(f"[red]Error in improve_insights: {str(e)}[/red]")
            raise

    async def process_transcript(self, transcript_path: Path, progress: Progress) -> MeetingInsight:
        """Process a transcript file into structured insights."""
        transcript_name = transcript_path.stem
        output_manager = OutputManager(transcript_name)
        
        # Read transcript
        read_task = progress.add_task(f"[cyan]Reading {transcript_path.name}...", total=1)
        transcript = transcript_path.read_text(encoding='utf-8')
        progress.update(read_task, completed=1, visible=False)
        
        # Analysis step
        analysis_task = progress.add_task("[cyan]Analyzing transcript...", total=1)
        try:
            insight = await self.analyze_transcript(transcript)
            progress.update(analysis_task, completed=1, visible=False)
            return insight
        except Exception as e:
            console.print(f"[red]Error during analysis: {e}[/red]")
            raise

    async def _handle_human_review(
        self, insight: MeetingInsight, transcript: str,
        transcript_name: str, progress: Progress
    ) -> MeetingInsight:
        """Handle human review and feedback."""
        console.print(Panel.fit("[bold yellow]Human Review[/bold yellow]", border_style="yellow"))
        
        # Show current insights
        console.print("\n[bold]Current Insights:[/bold]")
        console.print_json(data=insight.model_dump())
        
        if all(len(getattr(insight, field)) == 0 for field in insight.model_fields):
            return await self._handle_empty_insights(
                insight, transcript, transcript_name, progress
            )
        
        # Get approval or feedback
        approval = console.input("\n[bold yellow]Approve these insights? (y/n): [/bold yellow]")
        if approval.lower() != 'y':
            console.print("[yellow]Please provide feedback for improvement:[/yellow]")
            feedback = console.input()
            return await self._handle_iteration(
                insight, feedback, transcript, transcript_name
            )
        
        return insight

    async def _handle_empty_insights(
        self, insight: MeetingInsight, transcript: str,
        transcript_name: str, progress: Progress
    ) -> MeetingInsight:
        """Handle case when no insights were extracted."""
        console.print("\n[red]Warning: No insights were extracted. Would you like to:[/red]")
        console.print("1. Provide feedback for another attempt")
        console.print("2. Enter insights manually")
        console.print("3. Skip and continue with empty insights")
        
        choice = console.input("\n[bold yellow]Enter choice (1-3): [/bold yellow]")
        
        if choice == "1":
            console.print("[yellow]Please provide feedback for improvement:[/yellow]")
            feedback = console.input()
            return await self._handle_iteration(
                insight, feedback, transcript, transcript_name
            )
        elif choice == "2":
            return self._handle_manual_entry()
        
        return insight

    async def _handle_iteration(
        self, insight: MeetingInsight, feedback: str,
        transcript: str, transcript_name: str
    ) -> MeetingInsight:
        """Handle iteration based on feedback."""
        console.print(Panel.fit("[bold red]Iteration Based on Feedback[/bold red]", border_style="red"))
        
        try:
            new_insight = await self.improve_insights(insight, feedback, transcript)
            
            output_manager = OutputManager(transcript_name)
            iteration_path = output_manager.save_output(
                new_insight.to_markdown(),
                "3_iteration"
            )
            console.print(f"[green]✓[/green] Iteration saved to: [bold blue]{iteration_path}[/bold blue]")
            
            return new_insight
        except Exception as e:
            console.print(f"[red]Error during iteration: {e}[/red]")
            return insight

    def _handle_manual_entry(self) -> MeetingInsight:
        """Handle manual insight entry."""
        console.print("[yellow]Enter insights manually (one per line, empty line to finish):[/yellow]")
        insight = MeetingInsight()
        
        for field in insight.model_fields:
            console.print(f"\n[bold]Enter {field}:[/bold]")
            items = []
            while True:
                item = console.input()
                if not item:
                    break
                items.append(InsightItem(
                    point=item,
                    quote="Manually entered",
                    speaker="Manual Entry",
                    subcategory=SubCategory.PROPOSED
                ))
            setattr(insight, field, items)
        
        return insight

class OutputManager:
    """Manages file output and storage."""
    
    def __init__(self, transcript_name: str):
        self.transcript_name = transcript_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_output(self, content: str, stage: str) -> Path:
        """Save output to a file."""
        output_dir = Path("output") / self.transcript_name / self.timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{stage}.md"
        output_path.write_text(content)
        return output_path

async def main():
    """Main entry point."""
    console.print(Panel.fit("[bold green]Meeting Transcript Processor[/bold green]", border_style="green"))
    
    processor = TranscriptProcessor(auto_mode=False)
    transcript_dir = Path("transcripts")
    transcript_dir.mkdir(exist_ok=True)
    
    # List available transcripts
    transcripts = list(transcript_dir.glob("*.md"))
    if not transcripts:
        console.print("[yellow]No transcripts found in /transcripts/ directory. Please add a .md file.[/yellow]")
        return
    
    # Show available transcripts
    console.print("\n[bold cyan]Available transcripts:[/bold cyan]")
    for i, transcript in enumerate(transcripts, 1):
        console.print(f"{i}. {transcript.name}")
    
    # Get user selection
    while True:
        try:
            selection = console.input("\n[bold yellow]Enter transcript number to analyze (or 'q' to quit): [/bold yellow]")
            if selection.lower() == 'q':
                return
            
            idx = int(selection) - 1
            if 0 <= idx < len(transcripts):
                transcript_path = transcripts[idx]
                break
            else:
                console.print("[red]Invalid selection. Please try again.[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")
    
    # Process selected transcript
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        console.print(f"\n[bold blue]Processing: {transcript_path.name}[/bold blue]")
        await processor.process_transcript(transcript_path, progress)
        console.print("\n[bold green]✨ Transcript processed successfully![/bold green]")

if __name__ == "__main__":
    asyncio.run(main())