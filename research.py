import os
import asyncio
import datetime
from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from tavily import AsyncTavilyClient
from pathlib import Path

# Tavily client setup
tavily_client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Define dependencies and models
@dataclass
class SearchDataclass:
    max_results: int
    todays_date: str

@dataclass
class ResearchDependencies:
    todays_date: str

class ResearchResult(BaseModel):
    research_title: str = Field(description='This is a top-level Markdown heading that covers the topic of the query and is prefixed with #.')
    research_main: str = Field(description='This is a main section that provides detailed answers for the query and research.')
    research_bullets: str = Field(description='This is a set of bullet points summarizing the answers for the query.')

# Create the research agent
search_agent = Agent('openai:gpt-4o',
                    deps_type=ResearchDependencies,
                    result_type=ResearchResult,
                    system_prompt='Your a helpful research assistant, you are an expert in research '
                    'If you are given a question you write strong keywords to do 3-5 searches in total '
                    '(each with a query_number) and then combine the results',)

# Tool to perform searches
@search_agent.tool
async def get_search(search_data: RunContext[SearchDataclass], query: str, query_number: int) -> dict[str, Any]:
    """Perform a search for the given query.

    Args:
        search_data: Context data for the search.
        query: Keywords to search.
        query_number: The number of the query in the sequence.
    """
    print(f"Search query {query_number}: {query}")
    max_results = search_data.deps.max_results
    results = await tavily_client.get_search_context(query=query, max_results=max_results)
    return results

# Update the system prompt dynamically with the current date
@search_agent.system_prompt
async def add_current_date(ctx: RunContext[ResearchDependencies]) -> str:
    todays_date = ctx.deps.todays_date
    system_prompt = (
        "You're a helpful research assistant. You are an expert in research. "
        "If you are given a question, you write strong keywords to perform 3-5 searches in total "
        "(each with a query_number) and then combine the results. "
        f"If you need today's date, it is {todays_date}."
    )
    return system_prompt

# Set up dependencies
current_date = datetime.date.today()
deps = SearchDataclass(max_results=3, todays_date=current_date.strftime("%Y-%m-%d"))

# Function to run the agent
async def main():
    # Create rich console for better output
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    
    console.print(Panel.fit("[bold green]Research Assistant[/bold green]", border_style="green"))
    
    while True:
        # Get research query from user
        query = console.input("\n[bold yellow]Enter your research query (or 'q' to quit): [/bold yellow]")
        
        if query.lower() == 'q':
            console.print("[green]Goodbye![/green]")
            break
            
        if not query.strip():
            console.print("[red]Please enter a valid query.[/red]")
            continue
            
        try:
            # Set up dependencies
            current_date = datetime.date.today()
            deps = SearchDataclass(max_results=3, todays_date=current_date.strftime("%Y-%m-%d"))
            
            # Run research
            console.print(f"\n[cyan]Researching: {query}[/cyan]")
            result = await search_agent.run(query, deps=deps)
            
            # Print results
            console.print("\n[bold green]Research Results:[/bold green]")
            console.print("[bold blue]# " + result.data.research_title + "[/bold blue]")
            console.print("\n" + result.data.research_main)
            console.print("\n[bold cyan]Key Points:[/bold cyan]")
            console.print(result.data.research_bullets)
            
            # Ask if user wants to save results
            save = console.input("\n[yellow]Save results to file? (y/n): [/yellow]").lower()
            if save == 'y':
                # Create output directory if it doesn't exist
                output_dir = Path("research_output")
                output_dir.mkdir(exist_ok=True)
                
                # Create filename from research title
                filename = result.data.research_title.lower().replace(" ", "_")[:50] + ".md"
                output_path = output_dir / filename
                
                # Save markdown output
                markdown = "\n\n".join([
                    "# " + result.data.research_title,
                    result.data.research_main,
                    "## Key Points",
                    result.data.research_bullets
                ])
                output_path.write_text(markdown)
                console.print(f"[green]Results saved to: {output_path}[/green]")
            
            # Ask if user wants to continue
            continue_research = console.input("\n[yellow]Research another topic? (y/n): [/yellow]").lower()
            if continue_research != 'y':
                console.print("[green]Goodbye![/green]")
                break
                
        except Exception as e:
            console.print(f"[red]Error during research: {str(e)}[/red]")
            console.print("[yellow]Would you like to try again? (y/n): [/yellow]")
            if console.input().lower() != 'y':
                break

# Run the script
if __name__ == "__main__":
    asyncio.run(main())