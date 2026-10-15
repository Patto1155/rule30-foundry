#!/usr/bin/env python3
"""Physical finite-right-seed SAT; explicit disjunction over every prefix horizon."""
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


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(N,K,threshold,base,rule=30,target='alternating',fixed_right=None):
    """All r<2**K, actual infinite zero right exterior, some 1<=n<=N.

    Left completion is triangular. Right physical causal cone feeds only the
    even column bits in the Rule30 alternating case, all column bits otherwise.
    Initial right bits beyond N cannot influence any observation, but remain
    allocated when K>N so the seed class is explicitly exactly K bits.
    """
    if K<0 or N<1: raise ValueError('K>=0, N>=1 required')
    enc=base.encode(N,rule=rule,target=target)
    one=enc.new_var(); enc.add_clause([one]); zero=-one
    right={}; seeds=[enc.new_var() for _ in range(K)]
    for i in range(1,N+1): right[0,i]=seeds[i-1] if i<=K else zero
    if fixed_right is not None:
        if fixed_right<0 or fixed_right >= 1<<K: raise ValueError('seed outside K bits')
        for j,v in enumerate(seeds): enc.add_clause([v if (fixed_right>>j)&1 else -v])
    reduced=rule==30 and target=='alternating'
    for t in range(N):
        if t>0:
            for i in range(1,N-t+1):
                z=enc.new_var(); right[t,i]=z
                left=right[t-1,i-1] if i>1 else (one if target=='alternating' and (t-1)%2 else zero)
                inputs=[left,right[t-1,i],right[t-1,i+1]]
                for bits in itertools.product((0,1),repeat=3):
                    value=(rule>>(4*bits[0]+2*bits[1]+bits[2]))&1
                    enc.add_clause([(-v if b else v) for v,b in zip(inputs,bits)]+[z if value else -z])
        if not reduced or t%2==0:
            u=enc.columns[-1,t]; v=right[t,1]
            enc.add_clause([-u,v]); enc.add_clause([u,-v])
    # Exact unary prefix counters: C[j,k] iff at least k of e_1..e_j.
    # C[j,k] = C[j-1,k] OR (e_j AND C[j-1,k-1]).
    cap=max(0,(N-threshold)//7+1); prev={0:one}; triggers=[]
    for j,e in enumerate(enc.e_vars,1):
        cur={0:one}
        for k in range(1,min(j,cap)+1):
            a=prev.get(k,zero); b=prev.get(k-1,zero); z=enc.new_var(); cur[k]=z
            enc.add_clause([-a,z]); enc.add_clause([-e,-b,z])
            enc.add_clause([-z,a,e]); enc.add_clause([-z,a,b])
        if j>=threshold:
            bound=(j-threshold)//7
            triggers.append(-cur.get(bound+1,zero))
        prev=cur
    enc.add_clause(triggers)  # empty clause if no horizon can attain threshold
    enc.seed_vars=seeds
    return enc


def replay(enc,model,N,threshold,ref,rule=30,target='alternating'):
    yes={v for v in model if v>0}
    r=sum(int(v in yes)<<i for i,v in enumerate(enc.seed_vars))
    e=sum(int(v in yes)<<i for i,v in enumerate(enc.e_vars))
    if rule==30 and target=='alternating':
        assert e==ref.completion(r,N)
        assert ref.forward_check(r,e,N)
        u=''.join(map(str,ref.column1(r,N-1)[::2]))
    else:
        from experiments.support_growth import completion,forward_check
        assert e==completion(r,N,target,rule); forward_check(r,e,N,target,rule); u=None
    witnesses=[(j,j-7*(e&((1<<j)-1)).bit_count()) for j in range(1,N+1)]
    n,d=max(witnesses,key=lambda x:x[1]); assert d>=threshold
    return {'right':hex(r),'N':n,'W':(e&((1<<n)-1)).bit_count(),'deficit':d,
            'completion_hex':hex(e),'completion_horizon':N,'forward_simulation':'PASS','u':u,
            'L2':None if u is None else not any(x in u for x in ('11','00000')),
            'L3':None if u is None else not any(x in u for x in ('11','00000','101001','0100101'))}


def query(enc,N,K,threshold,out,cadical,drat,timeout,ref,rule=30,target='alternating',fixed_right=None):
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    cnf=out.with_suffix('.cnf'); proof=out.with_suffix('.drat')
    cnf.write_text(enc.dimacs()); started=time.monotonic()
    row={'K':K,'horizon_max':N,'threshold':threshold,'rule':rule,'target':target,
         'quantifier':('exists r<2^K' if fixed_right is None else f'r={fixed_right}')+', exists 1<=n<=horizon_max: n-7W_n>=threshold',
         'counting':{'initial_seed_bits':K,'free_input_bits':K if fixed_right is None else 0,'class_size':f'2^{K}' if fixed_right is None else '1','purpose':'exact-exclusion','complexity_inference':False},
         'cnf':cnf.name,'cnf_sha256':sha(cnf),'variables':enc.nv,'clauses':len(enc.clauses)}
    try:
        p=subprocess.run([str(cadical),'-q','--no-binary','-t',str(timeout),str(cnf),str(proof)],capture_output=True,text=True,timeout=timeout+10)
        out.with_suffix('.solver.txt').write_text(p.stdout+p.stderr)
        row['status']={10:'SAT',20:'UNSAT'}.get(p.returncode,'UNKNOWN')
        if row['status']=='UNKNOWN': row['reason']='solver-timeout-or-interruption'
    except subprocess.TimeoutExpired:
        row['status']='UNKNOWN'; row['reason']='wall-timeout'
    if row['status']=='SAT':
        model=[int(v) for line in p.stdout.splitlines() if line.startswith('v ') for v in line.split()[1:] if v!='0']
        row['witness']=replay(enc,model,N,threshold,ref,rule,target)
    elif row['status']=='UNSAT':
        try:
            checked=subprocess.run([str(drat),str(cnf),str(proof)],capture_output=True,text=True,timeout=60)
            out.with_suffix('.check.txt').write_text(checked.stdout+checked.stderr)
            if checked.returncode!=0 or 'VERIFIED' not in checked.stdout: raise RuntimeError('DRAT verification failed')
            row.update(drat_checked=True,proof=proof.name,proof_sha256=sha(proof))
        except (subprocess.TimeoutExpired,RuntimeError) as ex:
            row.update(status='UNKNOWN',reason=str(ex),solver_result='UNSAT-unverified')
    if row['status']!='UNSAT': proof.unlink(missing_ok=True)
    row['seconds']=time.monotonic()-started
    return row



def verify_results(path,drat,ref,base):
    """Recreate semantic CNFs, replay witnesses, and externally check ASCII DRAT."""
    path=Path(path); result=json.loads(path.read_text())
    assert sha(ref.__file__)==result['reference_sha256'], 'reference source changed'
    assert sha(base.__file__)==result['encoding_sha256'], 'encoding source changed'
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        if 'archive' in result:
            archive=path.parent/result['archive']
            assert sha(archive)==result['archive_sha256']
            with tarfile.open(archive) as tar: tar.extractall(root,filter='data')
        else: root=path.parent
        for row in result['queries']:
            K=row['K']; N=row['horizon_max']; fixed=row['fixed_right']
            assert row['status'] in ('SAT','UNSAT','UNKNOWN')
            assert sha(root/row['cnf'])==row['cnf_sha256']
            enc=encode(N,K,row['threshold'],base,row['rule'],row['target'],fixed)
            assert enc.dimacs().encode()==(root/row['cnf']).read_bytes(), 'semantic CNF mismatch'
            assert enc.nv==row['variables'] and len(enc.clauses)==row['clauses']
            del enc
            count=row['counting']; free=K if fixed is None else 0
            assert count['initial_seed_bits']==K and count['free_input_bits']==free
            assert count['class_size']==(f'2^{K}' if fixed is None else '1')
            if row['status']=='UNKNOWN':
                assert not any(key in row for key in ('proof','proof_sha256','drat_checked','witness'))
            elif row['status']=='UNSAT':
                assert row['drat_checked'] and sha(root/row['proof'])==row['proof_sha256']
                checked=subprocess.run([str(drat),str(root/row['cnf']),str(root/row['proof'])],capture_output=True,text=True,timeout=60)
                assert checked.returncode==0 and 'VERIFIED' in checked.stdout
            else:
                w=row['witness']; r=int(w['right'],16); n=w['completion_horizon']; e=int(w['completion_hex'],16)
                assert 0<=r<1<<K and (fixed is None or r==fixed)
                assert n==N and 1<=w['N']<=N and 0<=e<1<<N
                if row['rule']==30 and row['target']=='alternating':
                    assert ref.completion(r,n)==e and ref.forward_check(r,e,n)
                    u=''.join(map(str,ref.column1(r,N-1)[::2]))
                    assert w['u']==u
                    assert w['L2']==(not any(x in u for x in ('11','00000')))
                    assert w['L3']==(not any(x in u for x in ('11','00000','101001','0100101')))
                else:
                    from experiments.support_growth import completion,forward_check
                    assert completion(r,n,row['target'],row['rule'])==e
                    forward_check(r,e,n,row['target'],row['rule'])
                weights=[(j,(e&((1<<j)-1)).bit_count()) for j in range(1,N+1)]
                j,weight=max(weights,key=lambda item:item[0]-7*item[1])
                assert (w['N'],w['W'],w['deficit'])==(j,weight,j-7*weight)
                assert w['deficit']>=row['threshold'] and w['forward_simulation']=='PASS'
    return {'verified_queries':len(result['queries']),'UNKNOWN':sum(q['status']=='UNKNOWN' for q in result['queries'])}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoding',type=Path,default=Path(__file__).with_name('free_u_density.py'))
    p.add_argument('--reference',type=Path,default=Path(__file__).with_name('period2_column_reduction.py'))
    p.add_argument('--cadical',type=Path,required=True); p.add_argument('--drat-trim',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('runs/p2-deficit-sat-2026-10-15'))
    p.add_argument('--verify',type=Path)
    p.add_argument('--N',type=int,default=300); p.add_argument('--K',default='64,128,256'); p.add_argument('--timeout',type=int,default=15)
    args=p.parse_args(); base=load(args.encoding,'free_u'); ref=load(args.reference,'reference')
    if args.verify:
        print(json.dumps(verify_results(args.verify,args.drat_trim,ref,base))); return
    assert ref.completion(0x98f9d8a0,190).bit_count()==26
    args.output.mkdir(parents=True,exist_ok=True)
    result={'purpose':'exact-exclusion','tools':{x.name:sha(x) for x in (args.cadical,args.drat_trim)},
            'reference_sha256':sha(args.reference),'encoding_sha256':sha(args.encoding),'queries':[]}
    cases=[(190,64,8,30,'alternating',0x98f9d8a0),(12,0,1,30,'zero',0),(12,2,1,90,'zero',1),(300,0,9,30,'alternating',0)]
    cases += [(args.N,int(K),9,30,'alternating',None) for K in args.K.split(',')]
    for i,(N,K,threshold,rule,target,fixed) in enumerate(cases):
        enc=encode(N,K,threshold,base,rule,target,fixed)
        row=query(enc,N,K,threshold,args.output/f'q{i:02d}',args.cadical,args.drat_trim,args.timeout,ref,rule,target,fixed)
        row['control']=i<4; row['fixed_right']=fixed
        result['queries'].append(row); (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(row),flush=True)
if __name__=='__main__': main()
