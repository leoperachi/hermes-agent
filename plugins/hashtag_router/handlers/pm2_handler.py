"""Handler do #pm2 — executa subcomandos PM2 com whitelist de segurança."""
import subprocess

ALLOWED_SUBCMDS = {"list", "status", "restart", "stop", "start", "logs", "monit", "describe"}
MAX_OUTPUT = 3500  # WhatsApp ~4096, com folga
TIMEOUT_SEC = 15


def handle_pm2(command: str, user: str) -> str:
    parts = command.strip().split(maxsplit=1)
    
    if not parts:
        return _run(["list"])
    
    subcmd = parts[0].lower()
    
    if subcmd not in ALLOWED_SUBCMDS:
        return (
            f"❌ Subcomando '{subcmd}' não permitido.\n"
            f"Permitidos: {', '.join(sorted(ALLOWED_SUBCMDS))}"
        )
    
    # 'logs' precisa de --nostream pra não travar
    if subcmd == "logs":
        target = parts[1] if len(parts) > 1 else ""
        args = ["logs"]
        if target:
            args.append(target)
        args += ["--lines", "20", "--nostream"]
        return _run(args)
    
    return _run(command.split())


def _run(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["pm2"] + args,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
            check=False,
        )
        output = (result.stdout or result.stderr).strip()
        
        if len(output) > MAX_OUTPUT:
            output = output[:MAX_OUTPUT] + "\n\n... (truncado)"
        
        return f"```\n{output}\n```" if output else "✅ Comando executado (sem output)"
    
    except subprocess.TimeoutExpired:
        return f"⏱️ Timeout: comando demorou mais de {TIMEOUT_SEC}s"
    except FileNotFoundError:
        return "❌ PM2 não encontrado no PATH. Instalado? `which pm2`"
    except Exception as e:
        return f"❌ Erro: {type(e).__name__}: {e}"
