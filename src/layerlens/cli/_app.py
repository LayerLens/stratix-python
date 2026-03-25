from __future__ import annotations

import click

from .._version import __version__
from .commands.ci import ci
from .commands.auth import login, logout, whoami
from .commands.bulk import bulk
from .commands.judge import judge
from .commands.space import space
from .commands.trace import trace
from .commands.scorer import scorer
from .commands.evaluate import evaluate
from .commands.integration import integration


@click.group()
@click.option(
    "--api-key",
    envvar="LAYERLENS_STRATIX_API_KEY",
    default=None,
    help="API key for authentication.",
)
@click.option("--host", default=None, help="API host (e.g. api.layerlens.ai).")
@click.option("--port", default=None, type=int, help="API port.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format.",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose/debug output.")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Suppress the startup banner.")
@click.version_option(version=__version__, prog_name="layerlens")
@click.pass_context
def cli(
    ctx: click.Context,
    api_key: str | None,
    host: str | None,
    port: int | None,
    output_format: str,
    verbose: bool,
    quiet: bool,
) -> None:
    """LayerLens Stratix CLI — manage traces, judges, evaluations, integrations, and more."""
    import sys

    if not quiet and sys.stderr.isatty():
        from ._banner import banner

        click.echo(banner(__version__), err=True)

    ctx.ensure_object(dict)
    ctx.obj["api_key"] = api_key
    ctx.obj["output_format"] = output_format
    ctx.obj["verbose"] = verbose

    # Build base_url from --host / --port
    base_url = None
    if host is not None:
        scheme = "https" if port in (None, 443) else "http"
        if port and port not in (80, 443):
            base_url = f"{scheme}://{host}:{port}/api/v1"
        else:
            base_url = f"{scheme}://{host}/api/v1"
    ctx.obj["base_url"] = base_url


# Core commands
cli.add_command(trace)
cli.add_command(judge)
cli.add_command(evaluate)
cli.add_command(integration)

# Additional commands
cli.add_command(scorer)
cli.add_command(space)
cli.add_command(bulk)
cli.add_command(ci)

# Auth commands
cli.add_command(login)
cli.add_command(logout)
cli.add_command(whoami)


@cli.command("completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish", "powershell"]))
def completion(shell: str) -> None:
    """Print shell completion setup instructions.

    \b
    Examples:
      stratix completion bash
      stratix completion zsh
      stratix completion fish
      stratix completion powershell
    """
    import os

    # Detect which command name was used to invoke the CLI
    prog = os.path.basename(os.environ.get("_", "layerlens"))
    if prog not in ("layerlens", "stratix"):
        prog = "layerlens"
    env_var = f"_{prog.upper()}_COMPLETE"

    instructions = {
        "bash": f'eval "$({env_var}=bash_source {prog})"',
        "zsh": f'eval "$({env_var}=zsh_source {prog})"',
        "fish": f"{env_var}=fish_source {prog} | source",
        "powershell": (
            f"Register-ArgumentCompleter -Native -CommandName {prog} -ScriptBlock {{\n"
            "    param($wordToComplete, $commandAst, $cursorPosition)\n"
            f'    $env:{env_var} = "powershell_source"\n'
            f'    {prog} | ForEach-Object {{ [System.Management.Automation.CompletionResult]::new($_, $_, "ParameterValue", $_) }}\n'
            f"    Remove-Item Env:{env_var}\n"
            "}"
        ),
    }
    if shell == "powershell":
        print(f"Add this to your PowerShell profile:\n\n{instructions[shell]}")
    else:
        print(f"Add this to your shell profile:\n\n  {instructions[shell]}")
