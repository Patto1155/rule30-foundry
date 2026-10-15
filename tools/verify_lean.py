#!/usr/bin/env python3
"""Build pinned Lean project and reject nonstandard proof dependencies."""
from pathlib import Path
import os
import shutil
import re
import tempfile
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STANDARD = {'propext', 'Classical.choice', 'Quot.sound'}


def lake_path():
    return shutil.which('lake') or (str(Path.home() / '.elan/bin/lake')
        if (Path.home() / '.elan/bin/lake').is_file() else None)


def audit_axioms(output, claimed):
    audited = set(re.findall(r"'Rule30\.([^']+)' (?:depends on axioms:|does not depend on any axioms)", output))
    errors = []
    if claimed != audited:
        errors.append(f'Incomplete axiom audit: missing={claimed-audited}, extra={audited-claimed}')
    for block in re.findall(r'depends on axioms: \[([^]]*)\]', output):
        dependencies = {name.strip() for name in block.split(',') if name.strip()}
        if not dependencies <= STANDARD:
            errors.append(f'Rejected nonstandard axioms: {dependencies - STANDARD}')
    return errors


def strip_lean_comments(source):
    """Blank comments and string literals, preserving positions and newlines."""
    out = list(source)
    i, depth, string = 0, 0, False
    while i < len(source):
        if depth:
            if source.startswith('/-', i):
                out[i:i+2] = '  '; depth += 1; i += 2; continue
            if source.startswith('-/', i):
                out[i:i+2] = '  '; depth -= 1; i += 2; continue
        elif string:
            if source[i] == '\\':
                out[i] = ' '
                i += 1
                if i < len(source) and source[i] != '\n':
                    out[i] = ' '
                i += 1; continue
            if source[i] == '"':
                string = False
        elif source.startswith('--', i):
            stop = source.find('\n', i)
            if stop < 0: stop = len(source)
            out[i:stop] = ' ' * (stop-i)
            i = stop; continue
        elif source.startswith('/-', i):
            out[i:i+2] = '  '; depth = 1; i += 2; continue
        elif source[i] == '"':
            string = True
        else:
            i += 1; continue
        if source[i] != '\n': out[i] = ' '
        i += 1
    if depth or string:
        raise ValueError('Unterminated Lean comment or string')
    return ''.join(out)


def source_theorems(source):
    """Collect declarations in every namespace; unsupported syntax fails closed."""
    clean = strip_lean_comments(source)
    commands = re.compile(
        r'(?m)^\s*(namespace|section|end)\b([^\n]*)|\b(theorem|lemma)\s+([^\s:(]+)')
    scopes, names = [], set()
    identifier = re.compile(r"[A-Za-z_\u0080-\uffff][\w'.]*(?:\.[\w']+)*\Z")
    for match in commands.finditer(clean):
        command, rest, declaration, name = match.groups()
        if command:
            argument = rest.strip().split()
            if command == 'end':
                if scopes: scopes.pop()
            else:
                namespace = argument[0] if command == 'namespace' and argument else ''
                if namespace and not identifier.fullmatch(namespace):
                    raise ValueError(f'Unsupported namespace: {namespace}')
                scopes.append(namespace)
        else:
            if '«' in name or '»' in name or not identifier.fullmatch(name):
                raise ValueError(f'Unsupported theorem name: {name}')
            prefix = '.'.join(scope for scope in scopes if scope)
            qualified = f'{prefix}.{name}' if prefix else name
            if qualified.startswith('_root_.'): qualified = qualified[7:]
            if not qualified.startswith('Rule30.'):
                raise ValueError(f'Theorem outside audited Rule30 namespace: {qualified}')
            names.add(qualified.removeprefix('Rule30.'))
    return names


def lean_sources(formal):
    return sorted(path for path in formal.rglob('*.lean')
                  if '.lake' not in path.relative_to(formal).parts)


def main():
    lake = lake_path()
    if lake is None:
        print('Lean toolchain absent; install elan and run lake update in formal/')
        return 1
    env = dict(os.environ)
    env['PATH'] = str(Path(lake).parent) + os.pathsep + env.get('PATH', '')
    probe = subprocess.run([lake, 'env', 'lean', '--version'], cwd=ROOT / 'formal',
                           env=env, text=True, capture_output=True)
    if 'failed to locate application' in probe.stderr or 'could not detect the configuration' in probe.stderr:
        # Compatibility shim changes only missing /proc/<pid>/exe lookup.
        shim = Path(tempfile.gettempdir()) / 'rule30-lean-proc-path.so'
        subprocess.run(['cc', '-shared', '-fPIC', str(ROOT / 'tools/lean_proc_path.c'),
                        '-o', str(shim), '-ldl'], check=True)
        env['LD_PRELOAD'] = str(shim) + (os.pathsep + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
        print('Lean: using /proc/self/exe compatibility shim (tools/lean_proc_path.c)')
    formal = ROOT / 'formal'
    sources = lean_sources(formal)
    claimed = set()
    for source in sources:
        claimed.update(source_theorems(source.read_text()))
    if not claimed:
        print('Rejected: no theorem declarations found', file=sys.stderr)
        return 1
    commands = [[lake, 'build']] + [
        [lake, 'env', 'lean', str(source.relative_to(formal))] for source in sources]
    # Generate the audit independently of handwritten #print statements. Import
    # every source module, so a new file cannot silently escape the audit.
    with tempfile.TemporaryDirectory(prefix='rule30-lean-audit-') as temporary:
        audit = Path(temporary) / 'Audit.lean'
        modules = ['.'.join(source.relative_to(formal).with_suffix('').parts)
                   for source in sources]
        audit.write_text(''.join(f'import {module}\n' for module in modules) +
                         ''.join(f'#print axioms Rule30.{name}\n' for name in sorted(claimed)))
        commands.append([lake, 'env', 'lean', str(audit)])
        for command in commands:
            p = subprocess.run(command, cwd=formal, env=env,
                               text=True, capture_output=True)
            print(p.stdout, end='')
            print(p.stderr, end='', file=sys.stderr)
            if p.returncode:
                return p.returncode
            if 'sorryAx' in p.stdout or 'sorry' in p.stdout + p.stderr:
                print('Rejected: admitted proof dependency or warning', file=sys.stderr)
                return 1
    errors = audit_axioms(p.stdout, claimed)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f'Lean: {len(claimed)} theorem declarations built and axiom-audited')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
