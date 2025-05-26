import asyncio
import time
from typing import List, Dict, Any
import aiohttp
import numpy as np
from rich.console import Console
from rich.table import Table

# Configuration
API_URL = "http://localhost:8000"
console = Console()

async def health_check() -> Dict[str, Any]:
    """Check if the server is healthy."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/health") as response:
            return await response.json()

async def get_cache_stats() -> Dict[str, Any]:
    """Get current cache statistics."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/cache/stats") as response:
            return await response.json()

async def query(text: str, use_cache: bool = True, similarity_threshold: float = 0.85) -> Dict[str, Any]:
    """Send a query to the server."""
    payload = {
        "query": text,
        "use_cache": use_cache,
        "similarity_threshold": similarity_threshold
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{API_URL}/query", json=payload) as response:
            return await response.json()

def print_query_result(result: Dict[str, Any]):
    """Print query result in a formatted table."""
    table = Table(title="Query Result")
    
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Query", result["query"])
    table.add_row("Response", result["response"])
    table.add_row("Cache Hit", str(result["cache_hit"]))
    table.add_row("Processing Time", f"{result['processing_time']:.4f}s")
    table.add_row("Embedding Dimension", str(result["embedding_dimension"]))
    
    if result["cached_data"]:
        table.add_row("Cache Similarity", f"{result['cached_data']['similarity_score']:.4f}")
        table.add_row("Cached Query", result["cached_data"]["cached_query"])
    
    console.print(table)
    console.print()

async def run_demo():
    """Run a demonstration of the semantic cache."""
    # Check server health
    try:
        health = await health_check()
        console.print("[green]Server is healthy![/green]")
    except Exception as e:
        console.print(f"[red]Error: Could not connect to server. Is it running?[/red]")
        return
    
    # Initial cache stats
    stats = await get_cache_stats()
    console.print(f"Initial cache size: {stats['cache_size']}")
    console.print()
    
    # Test queries
    test_queries = [
        "What is the capital of France?",
        "Tell me about Paris, the capital city of France",
        "What's the main city in France?",
        "How do I make a chocolate cake?",
        "What's the recipe for chocolate cake?",
        "Tell me the steps to bake a chocolate cake",
    ]
    
    for query_text in test_queries:
        console.print(f"[yellow]Sending query:[/yellow] {query_text}")
        result = await query(query_text)
        print_query_result(result)
        await asyncio.sleep(1)  # Small delay between queries
    
    # Final cache stats
    stats = await get_cache_stats()
    console.print(f"Final cache size: {stats['cache_size']}")

if __name__ == "__main__":
    asyncio.run(run_demo()) 