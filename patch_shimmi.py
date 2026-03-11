#!/usr/bin/env python3
"""
patch_shimmi.py — Surgical in-place fix for KeyError: '"action"'

Run directly on the server (no arguments needed):
    python3 patch_shimmi.py

It finds the actual running agent_engine.py, patches _build_orchestrator_messages
to stop calling ORCHESTRATOR_PROMPT.format(), and restarts the service.
"""
import os, re, shutil, subprocess, sys
from pathlib import Path

# ── Find the actual file being used ──────────────────────────────────────────
CANDIDATES = [
    "/opt/shimmi/app/agent_engine.py",
    "/opt/shimmi/src/app/agent_engine.py",
    os.path.expanduser("~/shimmi/app/agent_engine.py"),
    "./app/agent_engine.py",
]

target = None
for c in CANDIDATES:
    if Path(c).exists():
        content = Path(c).read_text()
        if "_build_orchestrator_messages" in content:
            target = Path(c)
            print(f"Found target: {target}  ({len(content.splitlines())} lines)")
            break

if not target:
    # Search more broadly
    result = subprocess.run(
        ["find", "/", "-name", "agent_engine.py", "-not", "-path", "*/site-packages/*"],
        capture_output=True, text=True, timeout=10
    )
    for line in result.stdout.splitlines():
        p = Path(line.strip())
        if p.exists():
            content = p.read_text()
            if "_build_orchestrator_messages" in content:
                target = p
                print(f"Found via find: {target}")
                break

if not target:
    print("ERROR: Could not find agent_engine.py with _build_orchestrator_messages")
    sys.exit(1)

# ── Read and patch ─────────────────────────────────────────────────────────
content = target.read_text()

# Nuke bytecode cache for this file
pyc_dir = target.parent / "__pycache__"
if pyc_dir.exists():
    for f in pyc_dir.glob("agent_engine*.pyc"):
        f.unlink()
        print(f"Deleted bytecode: {f}")

# The fix: replace `system_content = ORCHESTRATOR_PROMPT.format(...)` 
# with `system_content = ORCHESTRATOR_PROMPT`
# The format() call chokes on JSON braces {} in the prompt.
# The variables it was injecting are already in the user JSON payload — not needed in system.

# Strategy 1: replace multi-line .format() call
patched = re.sub(
    r'(system_content\s*=\s*)ORCHESTRATOR_PROMPT\.format\s*\(.*?\)',
    r'\1ORCHESTRATOR_PROMPT',
    content,
    flags=re.DOTALL,
)

if patched == content:
    # Strategy 2: simpler inline replacement
    patched = content.replace(
        "system_content = ORCHESTRATOR_PROMPT.format(",
        "system_content = ORCHESTRATOR_PROMPT  # patched: removed .format(\n    # "
    )
    # Now we need to close the now-commented-out format call safely.
    # Actually: just replace the whole function body's format call differently.
    # Let's try a line-by-line approach.
    lines = content.splitlines(keepends=True)
    new_lines = []
    skip_until_paren = False
    paren_depth = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if not skip_until_paren and "ORCHESTRATOR_PROMPT.format(" in line:
            # Replace this line and skip until closing paren
            new_lines.append("        system_content = ORCHESTRATOR_PROMPT  # FIX: no .format()\n")
            skip_until_paren = True
            # Count open parens on this line
            paren_depth = line.count("(") - line.count(")")
            if paren_depth <= 0:
                skip_until_paren = False
        elif skip_until_paren:
            paren_depth += line.count("(") - line.count(")")
            if paren_depth <= 0:
                skip_until_paren = False
        else:
            new_lines.append(line)
        i += 1
    patched = "".join(new_lines)

# Verify the patch actually removed the .format( call
if "ORCHESTRATOR_PROMPT.format(" in patched:
    print("ERROR: Patch failed — .format( still present")
    print("Please share: grep -n 'ORCHESTRATOR_PROMPT' /opt/shimmi/app/agent_engine.py")
    sys.exit(1)

# Verify syntax
import ast
try:
    ast.parse(patched)
    print("Syntax check: OK")
except SyntaxError as e:
    print(f"ERROR: Patch produced invalid syntax at line {e.lineno}: {e.msg}")
    sys.exit(1)

# Backup and write
backup = target.with_suffix(".py.bak2")
shutil.copy(target, backup)
print(f"Backup: {backup}")
target.write_text(patched)
print(f"Patched: {target}")

# Nuke cache again after writing
if pyc_dir.exists():
    for f in pyc_dir.glob("agent_engine*.pyc"):
        f.unlink()
        print(f"Deleted stale bytecode: {f}")

# Restart
print("\nRestarting shimmi service...")
r = subprocess.run(["systemctl", "restart", "shimmi"], capture_output=True, text=True)
if r.returncode == 0:
    print("Service restarted OK")
    print("\nWaiting 3s then checking status...")
    import time; time.sleep(3)
    subprocess.run(["journalctl", "-u", "shimmi", "-n", "15", "--no-pager"])
else:
    print(f"systemctl restart failed: {r.stderr}")
    print("Restart manually: systemctl restart shimmi")
