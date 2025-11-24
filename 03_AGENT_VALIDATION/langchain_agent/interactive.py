"""Interactive CLI for Agent Orchestrator with real-time pipeline visualization."""
import sys
import json
import time
from typing import Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich import box

from agent_service import (
    call_t1_qa_lookup,
    call_t2_generate,
    call_t3_validate,
    call_t1_write,
    verdict_chain,
    get_cfg
)

console = Console()

def _parse_verdict_status(text: str) -> str:
    if not text or not text.strip():
        return "FAIL"
    first = text.strip().split()[0].upper()
    return "PASS" if first == "PASS" else "FAIL"

class InteractivePipeline:
    """Interactive pipeline with real-time visualization."""
    
    def __init__(self):
        self.console = console
        self.model = get_cfg("T2_DEFAULT_MODEL", "gemini")
        self.rag_enabled = True  # NEW: Global RAG toggle
        self.loopback_fallback = get_cfg("LOOPBACK_ON_FAIL", True)  # NEW: Loopback on failure
    
    def print_header(self):
        """Print welcome header."""
        self.console.print(Panel.fit(
            "[bold cyan]🤖 Agent Orchestrator - Interactive Mode[/bold cyan]\n"
            "[dim]Type your query to generate network configurations[/dim]\n"
            "[dim]Commands: exit, model, rag, loopback, status, help[/dim]",
            border_style="cyan"
        ))
    
    def print_help(self):
        """Print help information."""
        help_table = Table(title="Available Commands", box=box.ROUNDED)
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="white")
        
        help_table.add_row("exit, quit, q", "Exit the interactive session")
        help_table.add_row("model", "Show current model")
        help_table.add_row("model gemini", "Switch to Gemini model")
        help_table.add_row("model llama", "Switch to Llama model")
        help_table.add_row("rag", "Show RAG status")
        help_table.add_row("rag on", "Enable RAG (retrieval-augmented generation)")
        help_table.add_row("rag off", "Disable RAG (direct model inference)")
        help_table.add_row("loopback", "Show loopback fallback status")
        help_table.add_row("loopback on", "Enable loopback on validation failure")
        help_table.add_row("loopback off", "Disable loopback fallback")
        help_table.add_row("status", "Show current configuration")
        help_table.add_row("help, ?", "Show this help message")
        help_table.add_row("<query>", "Process a network configuration query")
        
        self.console.print(help_table)
    
    def print_status(self):
        """Print current status."""
        status_table = Table(title="Current Configuration", box=box.ROUNDED)
        status_table.add_column("Setting", style="cyan")
        status_table.add_column("Value", style="white")
        
        status_table.add_row("Model", self.model)
        status_table.add_row("RAG Mode", "[green]Enabled[/green]" if self.rag_enabled else "[yellow]Disabled[/yellow]")
        status_table.add_row("Loopback Fallback", "[green]Enabled[/green]" if self.loopback_fallback else "[red]Disabled[/red]")
        status_table.add_row("T1 Base URL", get_cfg("T1_BASE_URL", "http://localhost:8000"))
        status_table.add_row("T2 Base URL", get_cfg("T2_BASE_URL", "http://localhost:8001"))
        status_table.add_row("T3 Base URL", get_cfg("T3_BASE_URL", "http://localhost:5000"))
        status_table.add_row("Timeout", f"{get_cfg('HTTP_TIMEOUT', 90)}s")
        
        self.console.print(status_table)
    
    def step_1_t1_lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """Step 1: T1 Q&A Lookup with retry."""
        while True:
            try:
                with self.console.status("[bold yellow]🔍 Step 1/5: Checking Q&A Knowledge Base (T1)...[/bold yellow]"):
                    time.sleep(0.5)
                    result = call_t1_qa_lookup(query)
                
                if result.get("found"):
                    self.console.print(Panel(
                        f"[bold green]✓ Cache Hit![/bold green]\n\n"
                        f"[white]{result.get('answer', 'No answer')}[/white]",
                        title="T1 Q&A Result",
                        border_style="green"
                    ))
                    return result
                else:
                    self.console.print(Panel(
                        "[yellow]⚠ No cached answer found[/yellow]\n"
                        "[dim]Proceeding to generate new configuration...[/dim]",
                        title="T1 Q&A Result",
                        border_style="yellow"
                    ))
                    return None
                    
            except Exception as e:
                self.console.print(Panel(
                    f"[bold red]✗ T1 Lookup Failed[/bold red]\n\n"
                    f"[red]{str(e)}[/red]",
                    title="❌ Step 1 Error",
                    border_style="red"
                ))
                
                retry = self.console.input("[yellow]Retry T1 lookup? (y/n):[/yellow] ").lower()
                if retry != 'y':
                    raise
    
    def step_2_t2_generate(self, query: str, force_no_rag: bool = False) -> Dict[str, Any]:
        """Step 2: T2 Config Generation with retry."""
        # Determine effective RAG status
        use_rag = self.rag_enabled and not force_no_rag
        
        while True:
            try:
                mode_msg = "RAG" if use_rag else "No-RAG"
                with self.console.status(f"[bold cyan]⚙️  Step 2/5: Generating Configuration (T2 - {self.model} - {mode_msg})...[/bold cyan]"):
                    time.sleep(0.5)
                    result = call_t2_generate(query, model=self.model, rag_enabled=use_rag)
                
                devices = list(result["evaluate_payload"]["changes"].keys())
                total_cmds = sum(len(cmds) for cmds in result["evaluate_payload"]["changes"].values())
                
                config_summary = f"[bold green]✓ Configuration Generated[/bold green]\n\n"
                config_summary += f"[cyan]Model:[/cyan] {self.model}\n"
                config_summary += f"[cyan]RAG:[/cyan] {'Enabled' if use_rag else 'Disabled'}\n"
                config_summary += f"[cyan]Devices:[/cyan] {', '.join(devices)}\n"
                config_summary += f"[cyan]Total Commands:[/cyan] {total_cmds}\n"
                
                self.console.print(Panel(config_summary, title="T2 Generation Result", border_style="green"))
                
                if self.console.input("\n[dim]Show full configuration? (y/n):[/dim] ").lower() == 'y':
                    syntax = Syntax(result["joined_config"], "cisco", theme="monokai", line_numbers=True)
                    self.console.print(Panel(syntax, title="Generated Configuration", border_style="blue"))
                
                return result
                
            except Exception as e:
                self.console.print(Panel(
                    f"[bold red]✗ T2 Generation Failed[/bold red]\n\n"
                    f"[red]{str(e)}[/red]",
                    title="❌ Step 2 Error",
                    border_style="red"
                ))
                
                retry = self.console.input("[yellow]Retry T2 generation? (y/n):[/yellow] ").lower()
                if retry != 'y':
                    raise
    
    def step_3_t3_validate(self, evaluate_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: T3 Validation with retry."""
        while True:
            try:
                with self.console.status("[bold magenta]🧪 Step 3/5: Validating Configuration (T3)...[/bold magenta]"):
                    time.sleep(0.5)
                    result = call_t3_validate(evaluate_payload)
                
                status = result.get("result", "UNKNOWN")
                summary = result.get("summary", {})
                
                val_table = Table(title="Validation Results", box=box.ROUNDED)
                val_table.add_column("Check", style="cyan")
                val_table.add_column("Status", style="white")
                val_table.add_column("Details", style="dim")
                
                cp = summary.get("CP", {})
                cp_status = "✓" if cp.get("status") == "PASS" else "✗"
                val_table.add_row("Control Plane", cp_status, f"{cp.get('rows', 0)} interfaces")
                
                tp = summary.get("TP", {})
                tp_status = "✓" if tp.get("status") == "PASS" else "✗"
                val_table.add_row("Topology", tp_status, f"{tp.get('edges', 0)} edges")
                
                reach = summary.get("REACH", [])
                for r in reach:
                    reach_status = "✓" if r.get("status") == "PASS" else "✗"
                    val_table.add_row(
                        f"Reach: {r.get('src')}→{r.get('dst')}", 
                        reach_status,
                        r.get("error", "OK")
                    )
                
                border_color = "green" if status == "OK" else "red"
                self.console.print(Panel(val_table, title=f"T3 Validation - {status}", border_style=border_color))
                
                return result
                
            except Exception as e:
                self.console.print(Panel(
                    f"[bold red]✗ T3 Validation Failed[/bold red]\n\n"
                    f"[red]{str(e)}[/red]",
                    title="❌ Step 3 Error",
                    border_style="red"
                ))
                
                retry = self.console.input("[yellow]Retry T3 validation? (y/n):[/yellow] ").lower()
                if retry != 'y':
                    raise
    
    def step_4_verdict(self, query: str, config: str, validation: Dict[str, Any]) -> Dict[str, str]:
        while True:
            try:
                with self.console.status("[bold blue]💭 Step 4/5: Synthesizing Verdict (LLM)...[/bold blue]"):
                    time.sleep(0.5)
                    verdict_text = verdict_chain.invoke({
                        "query": query,
                        "config": config,
                        "validation_json": json.dumps(validation, ensure_ascii=False)
                    })
                verdict_status = _parse_verdict_status(verdict_text)
                self.console.print(Panel(
                    f"[white]{verdict_text}[/white]",
                    title="🎯 AI Verdict",
                    border_style="blue"
                ))
                return {"text": verdict_text, "status": verdict_status}
            except Exception as e:
                self.console.print(Panel(
                    f"[bold red]✗ Verdict Synthesis Failed[/bold red]\n\n"
                    f"[red]{str(e)}[/red]",
                    title="❌ Step 4 Error",
                    border_style="red"
                ))
                if self.console.input("[yellow]Retry verdict synthesis? (y/n):[/yellow] ").lower() != 'y':
                    raise

    def step_5_write_back(self, query: str, config: str, validation: Dict[str, Any], verdict_status: str) -> Dict[str, Any]:
        if verdict_status != "PASS":
            self.console.print(Panel(
                "[yellow]⚠ Skipping write-back (AI verdict FAIL)[/yellow]",
                title="T1 Write-back",
                border_style="yellow"
            ))
            return {"status": "SKIPPED", "reason": "ai_verdict_fail"}
        while True:
            try:
                with self.console.status("[bold green]💾 Step 5/5: Saving to Knowledge Base (T1)...[/bold green]"):
                    time.sleep(0.5)
                    result = call_t1_write({"query": query, "config": config, "validation": validation})
                if result.get("status") == "OK":
                    self.console.print(Panel(
                        "[bold green]✓ Configuration saved to knowledge base[/bold green]",
                        title="T1 Write-back",
                        border_style="green"
                    ))
                else:
                    self.console.print(Panel(
                        f"[red]✗ Write-back failed: {result.get('error','Unknown')}[/red]",
                        title="T1 Write-back",
                        border_style="red"
                    ))
                return result
            except Exception as e:
                self.console.print(Panel(
                    f"[bold red]✗ T1 Write-back Failed[/bold red]\n\n"
                    f"[red]{str(e)}[/red]",
                    title="❌ Step 5 Error",
                    border_style="red"
                ))
                if self.console.input("[yellow]Retry write-back? (y/n):[/yellow] ").lower() != 'y':
                    return {"status": "ERROR", "error": str(e)}

    def process_query(self, query: str):
        """Process a query through the full pipeline."""
        self.console.print(f"\n[bold]Query:[/bold] [cyan]{query}[/cyan]\n")
        
        start_time = time.time()
        
        try:
            # Step 1: T1 Lookup
            qa_result = self.step_1_t1_lookup(query)
            
            if qa_result:
                # Cache hit - done!
                elapsed = time.time() - start_time
                self.console.print(f"\n[dim]⏱️  Total time: {elapsed:.2f}s[/dim]\n")
                return
            
            # Step 2: T2 Generate (with current RAG setting)
            gen_result = self.step_2_t2_generate(query)
            
            # Step 3: T3 Validate
            val_result = self.step_3_t3_validate(gen_result["evaluate_payload"])
            
            # Step 4: Verdict
            verdict_obj = self.step_4_verdict(query, gen_result["joined_config"], val_result)
            verdict_status = verdict_obj["status"]

            # Loopback Fallback Logic (only if enabled AND verdict failed)
            if self.loopback_fallback and verdict_status == "FAIL":
                self.console.print(Panel(
                    "[bold yellow]⚠ AI Verdict: FAIL[/bold yellow]\n\n"
                    "Loopback fallback will regenerate configuration WITHOUT retrieval context (RAG disabled).\n"
                    "This may produce different results.",
                    title="Loopback Fallback Available",
                    border_style="yellow"
                ))
                
                # Ask user for confirmation
                retry_loopback = self.console.input(
                    "\n[yellow]Attempt loopback generation? (y/n):[/yellow] "
                ).lower().strip()
                
                if retry_loopback == 'y':
                    self.console.print(Panel(
                        "[bold yellow]↺ Triggering Loopback (RAG Disabled)[/bold yellow]",
                        border_style="yellow"
                    ))
                    # Retry pipeline steps 2-4 with RAG force-disabled
                    gen_result = self.step_2_t2_generate(query, force_no_rag=True)
                    val_result = self.step_3_t3_validate(gen_result["evaluate_payload"])
                    verdict_obj = self.step_4_verdict(query, gen_result["joined_config"], val_result)
                    verdict_status = verdict_obj["status"]
                else:
                    self.console.print("[dim]Skipping loopback retry (user declined)[/dim]")

            # Step 5: Write-back
            wb_result = self.step_5_write_back(query, gen_result["joined_config"], val_result, verdict_status)
            
            elapsed = time.time() - start_time
            self.console.print(f"\n[dim]⏱️  Total time: {elapsed:.2f}s[/dim]\n")

            # Final summary
            summary = Table(title="Pipeline Summary", box=box.ROUNDED)
            summary.add_column("Stage", style="cyan")
            summary.add_column("Status", style="white")
            
            summary.add_row("T1 Q&A Lookup", "❌ Miss")
            summary.add_row("T2 Generation", f"✓ {self.model}")
            summary.add_row("T3 Validation", "✓" if val_result.get("result") == "OK" else "✗")
            auto_reach_ct = len(gen_result["evaluate_payload"].get("intent", {}).get("reach", []))
            iface_ct = len(gen_result["evaluate_payload"].get("intent", {}).get("interface", []))
            policy_ct = len(gen_result["evaluate_payload"].get("intent", {}).get("policy", []))
            red_ct = len(gen_result["evaluate_payload"].get("intent", {}).get("redundancy", []))
            summary.add_row("Reach Intents", str(auto_reach_ct))
            summary.add_row("Interface Intents", str(iface_ct))
            summary.add_row("Policy Intents", str(policy_ct))
            summary.add_row("Redundancy Intents", str(red_ct))
            summary.add_row("Verdict", f"{'✓' if verdict_status=='PASS' else '✗'}")
            summary.add_row("T1 Write-back", "✓" if wb_result.get("status") == "OK" else wb_result.get("status", "?"))
            
            self.console.print(summary)
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]⚠ Query cancelled by user[/yellow]")
        except Exception as e:
            self.console.print(Panel(
                f"[bold red]Error:[/bold red] {str(e)}\n\n"
                f"[dim]{type(e).__name__}[/dim]",
                title="❌ Pipeline Error",
                border_style="red"
            ))
    
    def run(self):
        """Run interactive loop."""
        self.print_header()
        
        while True:
            try:
                # Get user input
                query = self.console.input("\n[bold green]>[/bold green] ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() in ['exit', 'quit', 'q']:
                    self.console.print("[cyan]Goodbye! 👋[/cyan]")
                    break
                
                elif query.lower() in ['help', '?']:
                    self.print_help()
                    continue
                
                elif query.lower() == 'status':
                    self.print_status()
                    continue
                
                elif query.lower().startswith('model'):
                    parts = query.split(' ', 1)
                    if len(parts) == 1:
                        # Just "model" - show current
                        self.console.print(f"[cyan]Current model:[/cyan] [bold]{self.model}[/bold]")
                        self.console.print("[dim]Available models: gemini, llama[/dim]")
                    else:
                        # "model <name>" - switch model
                        new_model = parts[1].strip()
                        if new_model in ['gemini', 'llama']:
                            self.model = new_model
                            self.console.print(f"[green]✓ Switched to [bold]{new_model}[/bold] model[/green]")
                        else:
                            self.console.print(f"[red]✗ Invalid model '[bold]{new_model}[/bold]'. Use 'gemini' or 'llama'[/red]")
                    continue
                
                elif query.lower().startswith('rag'):
                    parts = query.split(' ', 1)
                    if len(parts) == 1:
                        # Just "rag" - show current
                        status = "[green]ON[/green]" if self.rag_enabled else "[yellow]OFF[/yellow]"
                        self.console.print(f"[cyan]RAG Mode:[/cyan] {status}")
                        self.console.print("[dim]Use 'rag on' to enable retrieval, 'rag off' to disable[/dim]")
                    else:
                        # "rag on/off" - toggle
                        setting = parts[1].strip().lower()
                        if setting == 'on':
                            self.rag_enabled = True
                            self.console.print("[green]✓ RAG mode [bold]ON[/bold] - Retrieval-augmented generation enabled[/green]")
                        elif setting == 'off':
                            self.rag_enabled = False
                            self.console.print("[yellow]✓ RAG mode [bold]OFF[/bold] - Direct model inference (no retrieval)[/yellow]")
                        else:
                            self.console.print(f"[red]✗ Invalid setting '[bold]{setting}[/bold]'. Use 'on' or 'off'[/red]")
                    continue
                
                elif query.lower().startswith('loopback'):
                    parts = query.split(' ', 1)
                    if len(parts) == 1:
                        # Just "loopback" - show current
                        status = "[green]ON[/green]" if self.loopback_fallback else "[red]OFF[/red]"
                        self.console.print(f"[cyan]Loopback Fallback:[/cyan] {status}")
                        self.console.print("[dim]Use 'loopback on' to enable fallback on validation failure[/dim]")
                    else:
                        # "loopback on/off" - toggle
                        setting = parts[1].strip().lower()
                        if setting == 'on':
                            self.loopback_fallback = True
                            self.console.print("[green]✓ Loopback fallback [bold]ON[/bold] - Will retry without RAG on validation failure[/green]")
                        elif setting == 'off':
                            self.loopback_fallback = False
                            self.console.print("[red]✓ Loopback fallback [bold]OFF[/bold] - No automatic retry on failure[/red]")
                        else:
                            self.console.print(f"[red]✗ Invalid setting '[bold]{setting}[/bold]'. Use 'on' or 'off'[/red]")
                    continue
                
                # Process as query
                self.process_query(query)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use 'exit' or 'quit' to leave[/yellow]")
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    # Check if rich is installed
    try:
        import rich
    except ImportError:
        print("ERROR: 'rich' library not installed")
        print("Install with: pip install rich")
        sys.exit(1)
    
    pipeline = InteractivePipeline()
    pipeline.run()
