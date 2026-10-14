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
    audited = set(re.findall(r"'Rule30\.(\w+)' (?:depends on axioms:|does not depend on any axioms)", output))
    errors = []
    if claimed != audited:
        errors.append(f'Incomplete axiom audit: missing={claimed-audited}, extra={audited-claimed}')
    for block in re.findall(r'depends on axioms: \[([^]]*)\]', output):
        dependencies = {name.strip() for name in block.split(',') if name.strip()}
        if not dependencies <= STANDARD:
            errors.append(f'Rejected nonstandard axioms: {dependencies - STANDARD}')
    return errors


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
    for command in ([lake, 'build'], [lake, 'env', 'lean', 'Rule30/Basic.lean']):
        p = subprocess.run(command, cwd=ROOT / 'formal', env=env,
                           text=True, capture_output=True)
        print(p.stdout, end='')
        print(p.stderr, end='', file=sys.stderr)
        if p.returncode:
            return p.returncode
        # Lean reports "declaration uses 'sorry'" on stdout, not stderr.
        if 'sorryAx' in p.stdout or 'sorry' in p.stdout + p.stderr:
            print('Rejected: admitted proof dependency or warning', file=sys.stderr)
            return 1
    # Modifiers must not hide a theorem from the audit; an unprinted one fails closed.
    claimed = set(re.findall(r'^(?:(?:private|protected|noncomputable)\s+)*(?:theorem|lemma)\s+(\w+)',
        (ROOT / 'formal/Rule30/Basic.lean').read_text(), re.MULTILINE))
    errors = audit_axioms(p.stdout, claimed)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f'Lean: {len(claimed)} theorem declarations built and axiom-audited')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
