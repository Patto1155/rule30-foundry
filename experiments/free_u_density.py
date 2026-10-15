#!/usr/bin/env python3
"""Exact-exclusion SAT for finite free-boundary completion; no asymptotic claim."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import time
import tarfile
import tempfile
from pysat.card import CardEnc, EncType


class FreeUEncoding:
    def __init__(self):
        self.clauses = []
        self.nv = 0
        self.columns = {}
        self.u_vars = []
        self.e_vars = []

    def new_var(self):
        self.nv += 1
        return self.nv

    def add_clause(self, clause):
        self.clauses.append(list(clause))

    def bound(self, k):
        """Fresh CNF clauses, sum(e)<=k; auxiliaries start beyond this encoder."""
        if k < 0:
            return self.clauses + [[1], [-1]]
        if k >= len(self.e_vars):
            return list(self.clauses)
        card = CardEnc.atmost(self.e_vars, k, top_id=self.nv, encoding=EncType.totalizer)
        return self.clauses + card.clauses

    def dimacs(self, clauses=None):
        clauses = self.clauses if clauses is None else clauses
        nv = max([self.nv] + [abs(v) for c in clauses for v in c])
        return f'p cnf {nv} {len(clauses)}\n' + ''.join(' '.join(map(str,c)) + ' 0\n' for c in clauses)


def encode(N, rule=30, target='alternating'):
    """Rule30 alternating has ceil(N/2) free even right-column bits.

    Odd right bits are erased by the OR latch, set to zero representatives.
    Other targets and Rule90 use N independent full temporal right bits: the
    odd-bit erasure is specific to Rule30 and alternating centre.
    columns[(j,t)] is x[-j,t], with j=-1 denoting right column x[1,t].
    """
    if N < 1 or rule not in (30,90) or target not in ('alternating','zero'):
        raise ValueError('N>=1, rule 30/90 and alternating/zero required')
    enc=FreeUEncoding(); enc.N=N; enc.rule=rule; enc.target=target
    one=enc.new_var(); enc.add_clause([one])
    for t in range(N+1):
        enc.columns[0,t] = one if target=='alternating' and t%2 else -one
    reduced=rule==30 and target=='alternating'
    for t in range(N):
        if not reduced or t%2==0:
            v=enc.new_var(); enc.u_vars.append(v)
        else:
            v=-one
        enc.columns[-1,t]=v
    for j in range(1,N+1):
        for t in range(N-j+1):
            z=enc.new_var(); enc.columns[j,t]=z
            inputs=[enc.columns[j-1,t+1],enc.columns[j-1,t],enc.columns[j-2,t]]
            for a,b,c in itertools.product((0,1),repeat=3):
                value=a ^ ((b|c) if rule==30 else c)
                enc.add_clause([(-v if bit else v) for v,bit in zip(inputs,(a,b,c))]+[z if value else -z])
        enc.e_vars.append(enc.columns[j,0])
    return enc


def reference(path=None):
    if path:
        spec=importlib.util.spec_from_file_location('column_reference',path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module
    from experiments import period2_column_reduction
    return period2_column_reduction


def replay(enc, model, ref):
    pos={v for v in model if v>0}
    u=[int(v in pos) for v in enc.u_vars]
    e=sum(int(v in pos)<<j for j,v in enumerate(enc.e_vars))
    if enc.rule==30 and enc.target=='alternating':
        assert e==ref.completion_from_u(u,enc.N)
    else:
        c=sum(int(enc.target=='alternating' and t%2)<<t for t in range(enc.N+1))
        d=sum(bit<<t for t,bit in enumerate(u)); expected=0; mask=(1<<enc.N)-1
        for j in range(enc.N):
            a=((c>>1)^((c|d) if enc.rule==30 else d))&mask
            expected|=(a&1)<<j; d,c=c,a; mask>>=1
        assert e==expected
    return {'u':''.join(map(str,u)),'completion_hex':hex(e),'weight':e.bit_count()}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def query(enc,k,out,cadical,drat,timeout,ref):
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    cnf=out.with_suffix('.cnf'); proof=out.with_suffix('.drat'); log=out.with_suffix('.log')
    cnf.write_text(enc.dimacs(enc.bound(k)))
    cmd=[str(cadical),'--no-binary',str(cnf),str(proof)]
    start=time.monotonic()
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
        log.write_text(p.stdout+p.stderr)
        status={10:'SAT',20:'UNSAT'}.get(p.returncode,'UNKNOWN')
    except subprocess.TimeoutExpired as ex:
        log.write_text((ex.stdout or b'').decode() if isinstance(ex.stdout,bytes) else (ex.stdout or ''))
        status='UNKNOWN'; p=None
    result={'bound':k,'status':status,'seconds':time.monotonic()-start,'cnf':str(cnf),'cnf_sha256':sha(cnf)}
    if status=='SAT':
        model=[int(v) for line in p.stdout.splitlines() if line.startswith('v ') for v in line.split()[1:] if v!='0']
        result.update(replay(enc,model,ref)); assert result['weight']<=k
        proof.unlink(missing_ok=True)
    elif status=='UNSAT':
        checked=subprocess.run([str(drat),str(cnf),str(proof)],capture_output=True,text=True,timeout=max(30,timeout*2))
        assert checked.returncode==0 and 'VERIFIED' in checked.stdout,checked.stdout[-2000:]
        out.with_suffix('.check.log').write_text(checked.stdout+checked.stderr)
        result.update(proof=str(proof),proof_sha256=sha(proof),drat_verified=True)
    else:
        proof.unlink(missing_ok=True)
    return result


def tighten(result, ref):
    """Reuse all independently replayed witnesses, and certified prefix bounds."""
    pool=[r['best_witness']['u'] for r in result['rows']]
    pool += [q['u'] for r in result['rows'] for q in r['queries'] if q['status']=='SAT']
    pool += ['01010100110101101101010010010001100100010']
    for row in result['rows']:
        n=row['N']; count=row['free_input_bits']
        for text in pool:
            u=(text+'0'*count)[:count]
            e=ref.completion_from_u(map(int,u),n)
            if e.bit_count()<row['upper_bound']:
                row['upper_bound']=e.bit_count()
                row['best_witness']={'u':u,'completion_hex':hex(e),'weight':e.bit_count()}
        earlier=[r for r in result['rows'] if r['N']<n]
        if earlier:
            prior=max(earlier,key=lambda r:r['lower_bound'])
            if prior['lower_bound']>row['lower_bound']:
                row['lower_bound']=prior['lower_bound']; row['prefix_lower_bound_from_N']=prior['N']
        row['exact']=row['lower_bound']==row['upper_bound']
        row['density_interval']=[row['lower_bound']/n,row['upper_bound']/n]


def run(args):
    ref=reference(args.reference); ref.verify(random_rows=5,horizon=25)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    result={'purpose':'exact-exclusion','asymptotic_density_status':'undecided','reference_sha256':sha(ref.__file__),
            'tools':{str(p):sha(p) for p in (args.cadical,args.drat_trim)},'rows':[]}
    started=time.monotonic()
    for N in args.sizes:
        enc=encode(N); count=len(enc.u_vars)
        # Counting preflight is actual independent input bits, not a model-complexity claim.
        row={'N':N,'free_input_bits':count,'class_size':f'2^{count}',
             'counting_scope':'exact-exclusion only; no complexity or randomness inference','queries':[]}
        baseline=ref.completion_from_u([0]*count,N)
        candidates=[([0]*count,baseline)]
        for prior in result['rows']:
            bits=list(map(int,prior['best_witness']['u']))
            bits=(bits+[0]*count)[:count]
            candidates.append((bits,ref.completion_from_u(bits,N)))
        for phase in (0,1):
            u=[(i+phase)%2 for i in range(count)]; candidates.append((u,ref.completion_from_u(u,N)))
        u,e=min(candidates,key=lambda x:x[1].bit_count())
        upper=e.bit_count(); lower=0
        row['best_witness']={'u':''.join(map(str,u)),'completion_hex':hex(e),'weight':upper}
        # Initial universal positivity is cheap; binary searches retain UNKNOWN holes.
        bounds=[0]
        while lower<upper and time.monotonic()-started<args.budget:
            k=bounds.pop(0) if bounds else (lower+upper-1)//2
            q=query(enc,k,out/f'N{N:04d}_k{k:04d}',args.cadical,args.drat_trim,args.timeout,ref)
            row['queries'].append(q)
            if q['status']=='SAT':
                upper=q['weight']; row['best_witness']={k:q[k] for k in ('u','completion_hex','weight')}
            elif q['status']=='UNSAT': lower=max(lower,k+1)
            else: break
        if result['rows']:
            prior=max(result['rows'],key=lambda r:r['lower_bound'])
            if prior['lower_bound']>lower:
                lower=prior['lower_bound']; row['prefix_lower_bound_from_N']=prior['N']
        row.update(lower_bound=lower,upper_bound=upper,exact=lower==upper,density_interval=[lower/N,upper/N])
        result['rows'].append(row)
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:row[k] for k in ('N','lower_bound','upper_bound','exact')}),flush=True)
    tighten(result,ref)
    result['elapsed_seconds']=time.monotonic()-started
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')


def verify(path, ref, drat):
    """Regenerate every CNF, check hashes/DRAT, and replay every SAT witness."""
    path=Path(path); data=json.loads(path.read_text()); count=0
    manifest=path.parent/'SHA256SUMS'
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            expected,name=line.split('  ',1)
            assert Path(name).name==name and sha(path.parent/name)==expected
    assert data['asymptotic_density_status']=='undecided'
    assert sha(ref.__file__)==data['reference_sha256']
    with tempfile.TemporaryDirectory() as td:
        archive=path.parent/'certificates.tar.xz'
        if archive.exists():
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    if Path(member.name).name!=member.name or not member.isfile():
                        raise ValueError('unsafe archive member')
                tar.extractall(td,filter='data')
        previous={}
        for row in data['rows']:
            enc=encode(row['N']); assert len(enc.u_vars)==row['free_input_bits']
            best=row['best_witness']; u=list(map(int,best['u']))
            e=ref.completion_from_u(u,row['N'])
            assert hex(e)==best['completion_hex'] and e.bit_count()==best['weight']==row['upper_bound']
            lower=0
            for q in row['queries']:
                cnf=Path(td)/Path(q['cnf']).name if archive.exists() else Path(q['cnf'])
                assert sha(cnf)==q['cnf_sha256']
                assert cnf.read_text()==enc.dimacs(enc.bound(q['bound']))
                if q['status']=='SAT':
                    e=ref.completion_from_u(map(int,q['u']),row['N'])
                    assert hex(e)==q['completion_hex'] and e.bit_count()==q['weight']<=q['bound']
                elif q['status']=='UNSAT':
                    proof=Path(td)/Path(q['proof']).name if archive.exists() else Path(q['proof'])
                    assert sha(proof)==q['proof_sha256']
                    checked=subprocess.run([str(drat),str(cnf),str(proof)],capture_output=True,text=True,timeout=60)
                    assert checked.returncode==0 and 'VERIFIED' in checked.stdout
                    lower=max(lower,q['bound']+1); count+=1
                else: assert q['status']=='UNKNOWN'
            if 'prefix_lower_bound_from_N' in row:
                source=row['prefix_lower_bound_from_N']
                assert source<row['N'] and source in previous
                lower=max(lower,previous[source])
            assert row['lower_bound']==lower
            previous[row['N']]=lower
            assert row['exact']==(lower==row['upper_bound'])
    return {'result':'PASS','rows':len(data['rows']),'DRAT_checks':count}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sizes',type=lambda s:list(map(int,s.split(','))),default=list(range(4,85,4))+[100,117,126,190,300,400,500])
    p.add_argument('--output',default='runs/p2-free-density-2026-10-15')
    p.add_argument('--verify',type=Path)
    p.add_argument('--cadical',type=Path); p.add_argument('--drat-trim',type=Path,required=True)
    p.add_argument('--reference'); p.add_argument('--timeout',type=float,default=10); p.add_argument('--budget',type=float,default=300)
    args=p.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify,reference(args.reference),args.drat_trim)))
    else:
        if not args.cadical: p.error('--cadical required for run')
        run(args)
if __name__=='__main__': main()
