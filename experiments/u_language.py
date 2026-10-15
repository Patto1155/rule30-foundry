#!/usr/bin/env python3
"""Exact finite right-boundary cones and bounded physical-language sampling.

Purpose: exact-exclusion, never a complexity or all-length language claim.
Integer bit zero is initial x[1]; u[s]=x[1,2s], centre[t]=t%2.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import numpy as np


def boundary(r, length):
    if r < 0 or length < 1:
        raise ValueError('nonnegative row and positive length required')
    out = []
    for _ in range(length):
        out.append(r & 1)
        for c in (0, 1):
            r = ((r << 1) | c) ^ (r | (r >> 1))
    return tuple(out)


def factor_sets(words, max_length=12):
    seen = [set() for _ in range(max_length + 1)]
    for word in words:
        word = ''.join(map(str, word))
        for k in range(1, max_length + 1):
            seen[k].update(word[i:i+k] for i in range(len(word)-k+1))
    return seen


def minimal_missing(seen):
    """Minimal absent factors relative to the supplied finite sample only."""
    result = {}
    for k in range(1, len(seen)):
        missing = []
        for bits in itertools.product('01', repeat=k):
            word = ''.join(bits)
            if word not in seen[k] and (k == 1 or
                    word[:-1] in seen[k-1] and word[1:] in seen[k-1]):
                missing.append(word)
        result[str(k)] = missing
    return result


def exhaustive_packed(bits):
    states = np.arange(1 << bits, dtype=np.uint64)
    packed = np.zeros(len(states), dtype=np.uint32)
    for s in range(32):
        packed |= (states & np.uint64(1)).astype(np.uint32) << np.uint32(s)
        for c in (0, 1):
            states = ((states << np.uint64(1)) | np.uint64(c)) ^ (states | (states >> np.uint64(1)))
    return packed


def enumerate_physical(bits=20, samples=20000, seed=30, max_length=12):
    """All r<2**bits, first 32 u; plus seeded 256-bit rows, first 256 u.

    uint64 truncation is exact for observed times 0..62: initial cells
    beyond 64 cannot reach cell 1 before time 64. This is NOT exact cones
    for arbitrary length-12 words at arbitrary later times.
    """
    if not 1 <= bits <= 24 or max_length > 12:
        raise ValueError('supported exhaustive bits 1..24, factor length <=12')
    packed = exhaustive_packed(bits)
    seen = [set() for _ in range(max_length+1)]
    for k in range(1,max_length+1):
        mask = (1 << k)-1
        values = set()
        for offset in range(33-k):
            values.update(map(int,np.unique((packed >> offset) & mask)))
        seen[k] = {''.join(str((v >> j)&1) for j in range(k)) for v in values}
    witnesses={}
    for k in range(1,max_length+1):
        found={}
        for offset in range(33-k):
            values,indices=np.unique((packed >> offset) & ((1<<k)-1),return_index=True)
            for value,index in zip(values,indices):
                word=''.join(str((int(value)>>j)&1) for j in range(k))
                found.setdefault(word,{'r':int(index),'macro_offset':offset})
            if len(found)==len(seen[k]):
                break
        witnesses[str(k)]=found
    exhaustive_counts = [len(s) for s in seen]
    rng = random.Random(seed)
    random_seen = factor_sets((boundary(rng.getrandbits(256),256) for _ in range(samples)), max_length)
    for k in range(1,max_length+1):
        seen[k].update(random_seen[k])
    return {'purpose':'exact-exclusion, not complexity evidence',
            'independent_initial_bits':bits, 'exhaustive_rows':1<<bits,
            'exhaustive_u_observations_per_row':32,
            'random_rows':samples,'random_initial_bits':256,'random_u_length':256,'seed':seed,
            'exhaustive_factor_counts':exhaustive_counts[1:],'exhaustive_factor_witnesses':witnesses,
            'combined_factor_counts':[len(s) for s in seen[1:]],
            'minimal_missing_in_combined_sample':minimal_missing(seen),
            'scope':'finite sample only; no SFT, regularity, or all-length characterization inference'}


def cone_cnf(word, rule=30, alternating=True):
    """Exact cone: all 2*len(word)-1 initial right cells independent.

    Every satisfying finite cone extends with zero exterior to a physical
    forced-centre right half. No finite-row support cap is imposed.
    """
    if not word or any(c not in '01' for c in word):
        raise ValueError('nonempty binary word required')
    T = 2*(len(word)-1)
    rows = [[j+1 for j in range(T+1)]]
    nv = T+1
    clauses = []
    for t in range(T):
        new = []
        for j in range(T-t):
            nv += 1
            new.append(nv)
            inputs = rows[-1][max(j-1,0):j+2] if j else rows[-1][:2]
            for values in itertools.product((0,1),repeat=len(inputs)):
                l,c,r = values if j else ((t%2 if alternating else 0),*values)
                out = (rule >> ((l<<2)|(c<<1)|r)) & 1
                clauses.append([(-v if b else v) for v,b in zip(inputs,values)] + [nv if out else -nv])
        rows.append(new)
    for s,c in enumerate(word):
        clauses.append([rows[2*s][0] * (1 if c=='1' else -1)])
    return nv,clauses,rows[0]


def certify_factor(word, out, solver, checker, seconds=10, rule=30, alternating=True):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    stem = f'factor-{word}-rule{rule}' + ('' if alternating else '-zero-centre')
    cnf,proof = out/(stem+'.cnf'),out/(stem+'.drat')
    nv,clauses,initial = cone_cnf(word,rule,alternating)
    cnf.write_text(f'p cnf {nv} {len(clauses)}\n'+''.join(' '.join(map(str,c))+' 0\n' for c in clauses))
    p = subprocess.run([str(solver),'--no-binary','-t',str(seconds),str(cnf),str(proof)],capture_output=True,text=True,timeout=seconds+10)
    (out/(stem+'.solver.log')).write_text(p.stdout+p.stderr)
    status = {10:'SAT',20:'UNSAT'}.get(p.returncode,'UNKNOWN')
    result = {'word':word,'rule':rule,'alternating_centre':alternating,'status':status,
              'independent_initial_bits':2*len(word)-1,'purpose':'exact-exclusion',
              'variables':nv,'clauses':len(clauses),'cnf_sha256':hashlib.sha256(cnf.read_bytes()).hexdigest()}
    if status == 'UNSAT':
        q = subprocess.run([str(checker),str(cnf),str(proof)],capture_output=True,text=True,timeout=30)
        (out/(stem+'.checker.log')).write_text(q.stdout+q.stderr)
        assert q.returncode == 0 and 'VERIFIED' in q.stdout
        result.update(drat_verified=True,drat_sha256=hashlib.sha256(proof.read_bytes()).hexdigest())
    elif status == 'SAT':
        model = {int(v) for line in p.stdout.splitlines() if line.startswith('v ') for v in line[2:].split() if int(v)>0}
        r = sum((1 << j) for j,v in enumerate(initial) if v in model)
        if rule==30 and alternating:
            assert ''.join(map(str,boundary(r,len(word))))==word
        result['initial_right_integer']=r
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--bits',type=int,default=20)
    p.add_argument('--samples',type=int,default=20000)
    p.add_argument('--solver',type=Path)
    p.add_argument('--ladder',action='store_true')
    p.add_argument('--verify',action='store_true')
    p.add_argument('--encoder',type=Path,default=Path(__file__).with_name('free_u_density.py'))
    p.add_argument('--reference',type=Path,default=Path(__file__).with_name('period2_column_reduction.py'))
    p.add_argument('--max-N',type=int,default=300)
    p.add_argument('--query-seconds',type=float,default=.05)
    p.add_argument('--checker',type=Path,required=True)
    a=p.parse_args()
    if a.verify:
        print(json.dumps(verify_saved(a.out,a.checker,a.encoder,a.reference),indent=2))
        return
    if a.ladder:
        ladder(a.encoder,a.reference,a.out,a.checker,a.max_N,a.query_seconds)
        return
    if a.solver is None:
        p.error('--solver is required for factor certification')
    a.out.mkdir(parents=True,exist_ok=True)
    result=enumerate_physical(a.bits,a.samples)
    missing=[w for words in result['minimal_missing_in_combined_sample'].values() for w in words]
    result['exact_cones']=[certify_factor(w,a.out,a.solver,a.checker) for w in missing]
    result['all_minimal_absences_certified']=all(q['status']=='UNSAT' and q.get('drat_verified') for q in result['exact_cones'])
    result['controls']=[certify_factor('11',a.out,a.solver,a.checker,rule=90),certify_factor('00000',a.out,a.solver,a.checker,alternating=False)]
    (a.out/'language.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))



def language_clauses(u_vars, forbidden):
    return [[(-v if b=='1' else v) for v,b in zip(u_vars[i:i+len(word)],word)]
            for word in forbidden for i in range(len(u_vars)-len(word)+1)]


def language_size(length, forbidden):
    """Exact count of independent u strings in this finite relaxation."""
    width=max(map(len,forbidden),default=1)-1
    counts={'':1}
    for _ in range(length):
        new={}
        for suffix,count in counts.items():
            for b in '01':
                word=suffix+b
                if not any(word.endswith(w) for w in forbidden):
                    key=word[-width:] if width else ''
                    new[key]=new.get(key,0)+count
        counts=new
    return sum(counts.values())


def ladder(encoder_path, reference_path, out, checker, max_N=300, seconds=.1, budget=300):
    """Threshold queries only; UNKNOWN never establishes a minimum or bound.

    Glucose's retained ASCII DRUP is checked by drat-trim. The proof system
    is a subset of DRAT. One global budget bounds exploratory solver work.
    """
    import importlib.util
    import threading
    import time
    from pysat.solvers import Solver
    def load(path,name):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    encmod=load(encoder_path,'free_u_encoder')
    ref=load(reference_path,'reference_columns')
    assert ref.completion_from_u(tuple(map(int,'01'*17+'0'*6)),79).bit_count()==10
    assert ref.completion_from_u(tuple(map(int,'01010100110101101101010010010001100100010')),81).bit_count()==9
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    langs={'L1':['11'],'L2':['11','00000'],'L3':['11','00000','101001','0100101']}
    rows=[];start=time.monotonic()
    result={'purpose':'exact-exclusion; no complexity inference','forbidden':langs,
            'threshold':'N-7W_N > 8','solver':'glucose4 with proof; interrupt budget per query',
            'query_seconds':seconds,'rows':rows,'reference_sha256':hashlib.sha256(Path(reference_path).read_bytes()).hexdigest(),
            'encoder_sha256':hashlib.sha256(Path(encoder_path).read_bytes()).hexdigest()}
    for N in range(1,max_N+1):
        enc=encmod.encode(N)
        # deficit >=9 iff W <= floor((N-9)/7).
        threshold=(N-9)//7
        bound=enc.bound(threshold)
        for name,bans in langs.items():
            row={'N':N,'language':name,'independent_u_bits':len(enc.u_vars),
                 'finite_language_size':language_size(len(enc.u_vars),bans),'weight_cap':threshold}
            if time.monotonic()-start>=budget:
                row['status']='NOT_RUN_BUDGET';rows.append(row);continue
            clauses=bound+language_clauses(enc.u_vars,bans)
            with Solver(name='g4',bootstrap_with=clauses,with_proof=True) as solver:
                timer=threading.Timer(seconds,solver.interrupt);timer.start()
                try:
                    answer=solver.solve_limited(expect_interrupt=True)
                finally:
                    timer.cancel()
                row['status']='SAT' if answer else ('UNSAT' if answer is False else 'UNKNOWN')
                if answer:
                    witness=encmod.replay(enc,solver.get_model(),ref)
                    assert witness['weight']<=threshold
                    assert all(b not in witness['u'] for b in bans)
                    row.update(witness)
                elif answer is False:
                    stem=out/f'{name}-N{N:03}'
                    cnf=stem.with_suffix('.cnf');proof=stem.with_suffix('.drat')
                    cnf.write_text(enc.dimacs(clauses))
                    proof.write_text('\n'.join(solver.get_proof())+'\n')
                    q=subprocess.run([str(checker),str(cnf),str(proof)],capture_output=True,text=True,timeout=30)
                    stem.with_suffix('.check.log').write_text(q.stdout+q.stderr)
                    assert q.returncode==0 and 'VERIFIED' in q.stdout
                    row.update(drat_verified=True,cnf_sha256=hashlib.sha256(cnf.read_bytes()).hexdigest(),drat_sha256=hashlib.sha256(proof.read_bytes()).hexdigest())
            rows.append(row)
        if N%10==0:
            (out/'ladder.json').write_text(json.dumps(result,indent=2)+'\n')
            print(N,[(r['language'],r['status']) for r in rows[-3:]],flush=True)
    u='01'*45+'000101'
    e=ref.completion_from_u(tuple(map(int,u)),191)
    assert e.bit_count()==26 and all(b not in u for b in langs['L3'])
    result['explicit_L3_counterexample']={'N':191,'u':u,'weight':26,'deficit':9,'completion_hex':hex(e),
        'scope':'relaxed L1/L2/L3 only; physical right realizability not asserted'}
    result['elapsed_seconds']=time.monotonic()-start
    (out/'ladder.json').write_text(json.dumps(result,indent=2)+'\n')
    summarize_ladder(result,out)
    return result


def summarize_ladder(result,out):
    summary={'threshold':result['threshold'],'languages':{}}
    for name in result['forbidden']:
        rows=[r for r in result['rows'] if r['language']==name]
        statuses={r['status'] for r in rows}
        prefix=0
        for r in rows:
            if r['N']==prefix+1 and r['status']=='UNSAT' and r.get('drat_verified'):
                prefix+=1
            else:
                break
        summary['languages'][name]={'max_N_queried':max(r['N'] for r in rows),
            'status_counts':{status:sum(r['status']==status for r in rows) for status in sorted(statuses)},
            'all_N_verified_through':prefix,
            'explicit_counterexample_N':191,'explicit_counterexample_deficit':9,
            'scope':'UNKNOWN is unresolved; relaxed-language counterexample is not physical'}
    (Path(out)/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def verify_saved(out, checker, encoder_path=None, reference_path=None):
    """Replay saved cones/proofs/witnesses; never repeat the million-row scan."""
    import importlib.util
    import tarfile
    import tempfile
    out=Path(out)
    result=json.loads((out/'language.json').read_text())
    assert result['exhaustive_rows']==1<<result['independent_initial_bits']
    assert result['exhaustive_u_observations_per_row']==32
    assert len(result['combined_factor_counts'])==12
    if 'verified_by_source_sha256' in result:
        assert result['verified_by_source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    seen=[set()]
    for k,count in enumerate(result['combined_factor_counts'],1):
        witnesses=result['exhaustive_factor_witnesses'][str(k)]
        assert len(witnesses)==count==result['exhaustive_factor_counts'][k-1]
        seen.append(set(witnesses))
        for word,w in witnesses.items():
            assert len(word)==k and set(word)<=set('01')
            assert 0<=w['r']<1<<result['independent_initial_bits']
            offset=w['macro_offset']
            assert 0<=offset<=result['exhaustive_u_observations_per_row']-k
            assert ''.join(map(str,boundary(w['r'],offset+k)[offset:]))==word
    missing=minimal_missing(seen)
    assert missing==result['minimal_missing_in_combined_sample']
    assert {q['word'] for q in result['exact_cones']}=={w for ws in missing.values() for w in ws}
    assert len(result['exact_cones'])==sum(map(len,missing.values()))
    checked=0
    for q in result['exact_cones']+result['controls']:
        word=q['word'];rule=q['rule'];alternating=q['alternating_centre']
        assert rule in (30,90) and isinstance(alternating,bool)
        if q in result['exact_cones']:
            assert rule==30 and alternating and q['status']=='UNSAT' and q['drat_verified']
        nv,clauses,_=cone_cnf(word,rule,alternating)
        assert q['independent_initial_bits']==2*len(word)-1
        assert q['variables']==nv and q['clauses']==len(clauses)
        stem='factor-'+word+'-rule'+str(rule)+('' if alternating else '-zero-centre')
        cnf=out/(stem+'.cnf');proof=out/(stem+'.drat')
        expected=f'p cnf {nv} {len(clauses)}\n'+''.join(' '.join(map(str,c))+' 0\n' for c in clauses)
        assert cnf.read_text()==expected
        assert hashlib.sha256(cnf.read_bytes()).hexdigest()==q['cnf_sha256']
        if q['status']=='UNSAT':
            assert hashlib.sha256(proof.read_bytes()).hexdigest()==q['drat_sha256']
            p=subprocess.run([str(checker),str(cnf),str(proof)],capture_output=True,text=True,timeout=30)
            assert p.returncode==0 and 'VERIFIED' in p.stdout
            checked+=1
        elif q['status']=='SAT':
            r=q['initial_right_integer'];cells=[(r>>j)&1 for j in range(2*len(word)-1)]
            observed=[]
            for t in range(2*len(word)-1):
                if t%2==0:
                    observed.append(str(cells[0]))
                cells=[(rule>>(((t%2 if alternating else 0) if j==0 else cells[j-1])*4+cells[j]*2+cells[j+1]))&1 for j in range(len(cells)-1)]
            assert ''.join(observed)==word
        else:
            raise AssertionError('local factor certificate unresolved')
    ladder_count=0
    lp=out/'ladder'/'ladder.json'
    if lp.exists():
        def load(path,name):
            spec=importlib.util.spec_from_file_location(name,path)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
        encmod=load(encoder_path,'verify_encoder');ref=load(reference_path,'verify_reference')
        lr=json.loads(lp.read_text());cache={}
        with tempfile.TemporaryDirectory() as td:
            archive=lp.parent/'proofs.tar.xz'
            if archive.exists():
                with tarfile.open(archive,'r:xz') as tar:
                    for item in tar.getmembers():
                        assert item.isfile() and Path(item.name).name==item.name
                        Path(td,item.name).write_bytes(tar.extractfile(item).read())
            for row in lr['rows']:
                N=row['N'];name=row['language'];bans=lr['forbidden'][name]
                assert row['independent_u_bits']==(N+1)//2
                assert row['finite_language_size']==language_size((N+1)//2,bans)
                assert row['weight_cap']==(N-9)//7
                if row['status']=='UNSAT':
                    assert row['drat_verified']
                    if N not in cache:
                        cache={N:encmod.encode(N)}
                    enc=cache[N]
                    clauses=enc.bound(row['weight_cap'])+language_clauses(enc.u_vars,bans)
                    stem=f'{name}-N{N:03}'
                    base=Path(td) if archive.exists() else lp.parent
                    cnf=base/(stem+'.cnf');proof=base/(stem+'.drat')
                    assert cnf.read_text()==enc.dimacs(clauses)
                    assert hashlib.sha256(cnf.read_bytes()).hexdigest()==row['cnf_sha256']
                    assert hashlib.sha256(proof.read_bytes()).hexdigest()==row['drat_sha256']
                    p=subprocess.run([str(checker),str(cnf),str(proof)],capture_output=True,text=True,timeout=30)
                    assert p.returncode==0 and 'VERIFIED' in p.stdout
                    ladder_count+=1
                elif row['status']=='SAT':
                    u=row['u'];e=ref.completion_from_u(tuple(map(int,u)),N)
                    assert all(b not in u for b in bans)
                    assert e.bit_count()==row['weight']<=row['weight_cap']
                else:
                    assert row['status'] in ('UNKNOWN','NOT_RUN_BUDGET')
            w=lr['explicit_L3_counterexample'];u=w['u'];e=ref.completion_from_u(tuple(map(int,u)),w['N'])
            assert e.bit_count()==w['weight'] and w['N']-7*w['weight']==w['deficit']
            assert all(b not in u for b in lr['forbidden']['L3'])
    l4_verified=[]
    l4path=out/'l4-targeted'/'results.json'
    if l4path.exists():
        import lzma
        l4=json.loads(l4path.read_text())
        assert set(l4['forbidden'])=={q['word'] for q in result['exact_cones']}
        assert len(l4['forbidden'])==len(result['exact_cones'])
        for row in l4['rows']:
            N=row['N'];enc=encmod.encode(N)
            enc.clauses+=language_clauses(enc.u_vars,l4['forbidden'])
            assert row['bound']==(N-9)//7
            path=l4path.parent/Path(row['cnf']).name
            payload=path.read_bytes() if path.exists() else lzma.open(str(path)+'.xz','rb').read()
            assert payload.decode()==enc.dimacs(enc.bound(row['bound']))
            assert hashlib.sha256(payload).hexdigest()==row['cnf_sha256']
            if row['status']=='SAT':
                u=row['u'];e=ref.completion_from_u(tuple(map(int,u)),N)
                assert all(b not in u for b in l4['forbidden'])
                assert e.bit_count()==row['weight']<=row['bound']
            elif row['status']=='UNSAT':
                proof=l4path.parent/Path(row['proof']).name
                assert hashlib.sha256(proof.read_bytes()).hexdigest()==row['proof_sha256']
                with tempfile.TemporaryDirectory() as td:
                    cnf=Path(td)/'query.cnf';cnf.write_bytes(payload)
                    p=subprocess.run([str(checker),str(cnf),str(proof)],capture_output=True,text=True,timeout=30)
                    assert p.returncode==0 and 'VERIFIED' in p.stdout
            else:
                assert row['status']=='UNKNOWN'
            l4_verified.append({'N':N,'status':row['status']})
    checkpoint=out/'l1-checkpoint'/'results.json'
    checkpoint_verified=False
    if checkpoint.exists():
        import lzma
        q=json.loads(checkpoint.read_text())
        assert q['N']==79 and q['forbidden']==['11'] and q['bound']==9
        assert q['status']=='UNSAT' and q['drat_verified'] and q['claimed_minimum']==10
        enc=encmod.encode(79);enc.clauses+=language_clauses(enc.u_vars,['11'])
        cnf=checkpoint.parent/Path(q['cnf']).name
        payload=cnf.read_bytes() if cnf.exists() else lzma.open(str(cnf)+'.xz','rb').read()
        assert payload.decode()==enc.dimacs(enc.bound(9))
        assert hashlib.sha256(payload).hexdigest()==q['cnf_sha256']
        proof=checkpoint.parent/Path(q['proof']).name
        assert hashlib.sha256(proof.read_bytes()).hexdigest()==q['proof_sha256']
        with tempfile.TemporaryDirectory() as td:
            cnf=Path(td)/'query.cnf';cnf.write_bytes(payload)
            p=subprocess.run([str(checker),str(cnf),str(proof)],capture_output=True,text=True,timeout=30)
            assert p.returncode==0 and 'VERIFIED' in p.stdout
        u=q['upper_witness_u']
        assert len(u)==40 and '11' not in u
        assert ref.completion_from_u(tuple(map(int,u)),79).bit_count()==10
        checkpoint_verified=True
    return {'local_proofs_verified':checked,'realized_factor_witnesses':sum(len(s) for s in seen),
            'ladder_proofs_verified':ladder_count,'L1_N79_min10_verified':checkpoint_verified,'L4_queries_verified':l4_verified,'scope':'exact factor sets only through length12; no all-length language claim'}


if __name__=='__main__':
    main()
