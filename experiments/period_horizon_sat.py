"""Certified finite-support trace instrument; T=0, never an eventuality theorem.

python experiments/period_horizon_sat.py --self-test
python experiments/period_horizon_sat.py --run --cadical third_party/cadical --drat-trim third_party/drat-trim
python experiments/period_horizon_sat.py --verify runs/period-horizon-2026-10-14/results.json --drat-trim third_party/drat-trim

Lengths count bits (not repetitions). Nested support makes satisfiability
monotone in radius. SAT at R and checked UNSAT at R-1 establish minimum R.
UNKNOWN is neither UNSAT nor a minimum. Every UNSAT includes a checked DRAT.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import lzma
import tempfile
import tarfile
import io
import re
import hashlib
import itertools
import json
import random
import subprocess
import sys
import time
from pathlib import Path

# Allow both module and direct script execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.counting_bound import finite_configuration_verdict


def cell(left, center, right, rule):
    return (rule >> (4 * left + 2 * center + right)) & 1


def simulate(initial, length, rule=30):
    """Independent unbounded sparse-cell simulation, no CNF or packed kernel."""
    alive = {int(i) for i, x in initial.items() if x}
    result = []
    for _ in range(length):
        result.append(int(0 in alive))
        positions = {i + d for i in alive for d in (-1, 0, 1)}
        alive = {i for i in positions if cell(int(i-1 in alive), int(i in alive), int(i+1 in alive), rule)}
    return result


def encode(radius, word, rule=30, nonzero=True, return_variables=False):
    """Zero exterior on infinite line, using only the backward observation cone.

    All initial support variables are allocated, including those outside the
    observation cone: they can satisfy nonzero without affecting the trace.
    At later times positions outside the forward support cone are literal 0.
    """
    if radius < 0 or not word or any(x not in (0, 1) for x in word):
        raise ValueError('radius >= 0 and nonempty binary word required')
    next_var = 1
    variables = {}
    clauses = []
    def alloc(t, i):
        nonlocal next_var
        variables[t, i] = next_var
        next_var += 1
        return variables[t, i]
    initial = {i: alloc(0, i) for i in range(-radius, radius+1)}
    if nonzero:
        clauses.append(list(initial.values()))
    def at(t, i):
        return variables.get((t, i), 0)
    for t in range(1, len(word)):
        bound = min(radius + t, len(word)-1-t)
        for i in range(-bound, bound+1):
            output = alloc(t, i)
            inputs = [at(t-1, i-1), at(t-1, i), at(t-1, i+1)]
            for bits in itertools.product((0, 1), repeat=3):
                if any(v == 0 and b for v, b in zip(inputs, bits)):
                    continue
                expected = cell(*bits, rule)
                # Forbid this input assignment together with the wrong output.
                clause = [v if b == 0 else -v for v, b in zip(inputs, bits) if v]
                clause.append(output if expected else -output)
                clauses.append(clause)
    for t, bit in enumerate(word):
        v = at(t, 0)
        clauses.append([v if bit else -v] if v else ([] if bit else [next_var]))
        if not v and not bit:  # tautological dummy variable, only radius0 special case
            next_var += 1
    header = f'p cnf {next_var-1} {len(clauses)}\n'
    text = header + ''.join(' '.join(map(str, c)) + (' ' if c else '') + '0\n' for c in clauses)
    return text, variables if return_variables else initial


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_drat(cnf, proof, binary, timeout=60):
    p = subprocess.run([str(binary), str(cnf), str(proof)], text=True,
                       capture_output=True, timeout=timeout)
    if p.returncode != 0 or 'VERIFIED' not in p.stdout:
        raise RuntimeError('DRAT check failed: ' + p.stdout[-1000:] + p.stderr[-1000:])
    return p.stdout


def self_test():
    # Exhaustive one-step truth table including the OR latch.
    for rule in (30, 90):
        for bits in itertools.product((0,1), repeat=3):
            text, init = encode(1, [bits[1], cell(*bits, rule)], rule, False)
            clauses = [[int(v) for v in line.split()[:-1]] for line in text.splitlines()[1:]]
            assignment = {init[-1]:bits[0], init[0]:bits[1], init[1]:bits[2], 4:cell(*bits,rule)}
            assert all(any(assignment[abs(v)] == int(v>0) for v in c) for c in clauses)
            assignment[4] ^= 1
            assert not all(any(assignment[abs(v)] == int(v>0) for v in c) for c in clauses)
    # Whole-cone boundary check: every small initial row has its independently
    # evolved assignment admitted by CNF at ten steps, including radius zero.
    for rule in (30,90):
        for radius in range(4):
            for bits in itertools.product((0,1),repeat=2*radius+1):
                alive={i for i,b in zip(range(-radius,radius+1),bits) if b}
                rows=[]
                for t in range(10):
                    rows.append(set(alive))
                    positions={i+d for i in alive for d in (-1,0,1)}
                    alive={i for i in positions if cell(int(i-1 in alive),int(i in alive),int(i+1 in alive),rule)}
                word=[int(0 in row) for row in rows]
                text,variables=encode(radius,word,rule,False,True)
                assignment={v:int(i in rows[t]) for (t,i),v in variables.items()}
                clauses=[[int(v) for v in line.split()[:-1]] for line in text.splitlines()[1:]]
                assert all(any(assignment[abs(v)]==int(v>0) for v in clause) for clause in clauses)
    # Independent exhaustive horizon maxima and extremizer counts.
    for radius in range(1, 7):
        maxima = {0:-1, 1:-1}
        counts = {0:0, 1:0}
        for bits in itertools.product((0,1), repeat=2*radius+1):
            if not any(bits):
                continue
            trace = simulate(dict(zip(range(-radius,radius+1),bits)), radius+4)
            bit = trace[0]
            horizon = next(t for t,x in enumerate(trace) if x != bit)
            if horizon > maxima[bit]:
                maxima[bit], counts[bit] = horizon, 1
            elif horizon == maxima[bit]:
                counts[bit] += 1
        assert maxima == {0:2*((radius+1)//2)+1, 1:2*(radius//2)+2}, (radius,maxima)
        assert counts == {0:2**radius-1, 1:2**radius}, (radius,counts)
    assert simulate({0:1},64,90) == [1]+[0]*63
    assert simulate({},64,30) == [0]*64
    print('PASS self-test: truth table, independent enumeration R1..6, Rule90 and zero')


class Runner:
    def __init__(self, args):
        self.args = args
        self.out = args.output
        self.out.mkdir(parents=True,exist_ok=True)
        (self.out/'certificates').mkdir(exist_ok=True)
        self.started = time.monotonic()
        self.queries = []
        self.cache = {}
        self.toolchain={'cadical_version':subprocess.check_output([str(args.cadical),'--version'],text=True).strip(),
                        'cadical_sha256':digest(args.cadical),'drat_trim_sha256':digest(args.drat_trim),
                        'proof_format':'ASCII DRAT','query_timeout_seconds':args.timeout,'search_budget_seconds':args.budget}
    def query(self, radius, word, rule=30, nonzero=True):
        key = (radius, tuple(word),rule,nonzero)
        if key in self.cache:
            return self.cache[key]
        counting(radius,len(word),nonzero)  # mandatory before EACH search
        base = f'certificates/q{len(self.queries):04d}'
        cnf = self.out/(base+'.cnf')
        proof = self.out/(base+'.drat')
        text, init = encode(radius,word,rule,nonzero)
        cnf.write_text(text)
        rec = {'id':len(self.queries),'radius':radius,'word':word,'rule':rule,
               'nonzero':nonzero,'cnf':base+'.cnf','cnf_sha256':digest(cnf),
               'counting':counting(radius,len(word),nonzero)}
        begin=time.monotonic()
        if begin-self.started >= self.args.budget:
            rec.update(status='UNKNOWN', reason='total-budget')
        else:
            limit=max(1,min(self.args.timeout,int(self.args.budget-(begin-self.started))))
            p = subprocess.run([str(self.args.cadical),'-q','--no-binary','-t',str(limit),str(cnf),str(proof)], text=True,capture_output=True,timeout=limit+10)
            (self.out/(base+'.solver.txt')).write_text(p.stdout+p.stderr)
            if p.returncode==10:
                assignments={abs(int(v)):int(int(v)>0) for line in p.stdout.splitlines() if line.startswith('v ') for v in line.split()[1:] if int(v)}
                row={str(i):assignments[v] for i,v in init.items()}
                assert (not nonzero or any(row.values())) and simulate(row,len(word),rule)==word
                rec.update(status='SAT',initial=row)
                proof.unlink(missing_ok=True)
            elif p.returncode==20:
                verified=check_drat(cnf,proof,self.args.drat_trim)
                (self.out/(base+'.drat-check.txt')).write_text(verified)
                rec.update(status='UNSAT',proof=base+'.drat',proof_sha256=digest(proof),drat_checked=True)
            elif p.returncode==0:
                rec.update(status='UNKNOWN',reason='solver-timeout')
                proof.unlink(missing_ok=True)
            else:
                raise RuntimeError('solver failed '+str(p.returncode)+p.stdout[-1000:]+p.stderr[-1000:])
        rec['elapsed_seconds']=round(time.monotonic()-begin,4)
        self.queries.append(rec)
        self.cache[key]=rec
        print(f"q{rec['id']} rule{rule} R{radius} L{len(word)} {rec['status']} {rec['elapsed_seconds']}s",flush=True)
        return rec
    def minimum(self,word,rule=30):
        lower=-1  # radius domain is >=0
        upper=self.args.max_radius
        qs=[]
        high=self.query(upper,word,rule)
        qs.append(high['id'])
        if high['status']!='SAT':
            return {'status':'censored','minimum_radius':None,'lower_excluded':upper if high['status']=='UNSAT' else lower,'sat_upper':None,'query_ids':qs}
        while upper-lower>1:
            mid=(upper+lower)//2
            q=self.query(mid,word,rule)
            qs.append(q['id'])
            if q['status']=='SAT':
                upper=mid
            elif q['status']=='UNSAT':
                lower=mid
            else:
                return {'status':'censored','minimum_radius':None,'lower_excluded':lower,'sat_upper':upper,'query_ids':qs}
        return {'status':'exact','minimum_radius':upper,'lower_excluded':lower,'sat_upper':upper,'query_ids':qs}


def run(args):
    runner=Runner(args)
    gate={'T':0,'length_range':[8,64], 'max_radius':args.max_radius,'purpose':'exact-exclusion',
          'arithmetic':'log2(2^(2R+1)-1), upper bound on distinct trace words',
          'not_complexity_evidence':True,'negative_guaranteed':False}
    (args.output/'counting-design.json').write_text(json.dumps(gate,indent=2)+'\n')
    controls=[]
    for radius in range(1,7):
        for bit, horizon in ((0,2*((radius+1)//2)+1),(1,2*(radius//2)+2)):
            for length,status in ((horizon,'SAT'),(horizon+1,'UNSAT')):
                q=runner.query(radius,[bit]*length)
                controls.append(q['id'])
                if q['status']!=status:
                    raise RuntimeError(f'Control FAILED R{radius} bit{bit} L{length}: expected{status} STOP')
    # Changing the initial condition is detected; zero is deliberately excluded.
    zero=runner.query(0,[0]*64,30,False)
    assert zero['status']=='SAT'
    rule90=runner.query(0,[1]+[0]*63,90)
    assert rule90['status']=='SAT'
    records=[]
    for length in range(8,65):
        for target in ('alternating','zero','one'):
            word=[i%2 for i in range(length)] if target=='alternating' else [int(target=='one')]*length
            records.append({'length':length,'target':target,'rule':30,**runner.minimum(word)})
        save(args.output,gate,runner,controls,records,[zero['id'],rule90['id']])
    rng=random.Random(20261014)
    nulls=[]
    for length in (8,16,32,64):
        word=[rng.randrange(2) for _ in range(length)]
        nulls.append({'length':length,'word':word,**runner.minimum(word)})
    save(args.output,gate,runner,controls,records,[zero['id'],rule90['id']],nulls)
    plot(args.output/'results.json')


def save(out,gate,runner,controls,records,sanity,nulls=None):
    result={'artifact_type':'rule30.period-horizon-sat','design':gate,'controls':controls,
            'sanity_controls':sanity,'queries':runner.queries,'records':records,'random_controls':nulls or [],
            'total_elapsed_seconds':round(time.monotonic()-runner.started,3),'toolchain':runner.toolchain}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')


def counting(radius, length, nonzero):
    if nonzero:
        return finite_configuration_verdict(radius,length)
    return {"radius":radius,"n_bits":length,"initial_configuration_count":str(1 << (2*radius+1)),
            "log2_class_size_upper":2*radius+1,"log2_expected_random_fits_upper":2*radius+1-length,
            "discriminating":2*radius+1>=length,"purpose":"exact-exclusion",
            "negative_guaranteed":False,"exclusion_would_still_be_true":True,"includes_zero":True}


def validate_metadata(result):
    design=result['design']
    assert design['T']==0 and design['length_range']==[8,64]
    assert design['purpose']=='exact-exclusion' and design['not_complexity_evidence']
    assert not design['negative_guaranteed']
    queries=result['queries']
    for i,q in enumerate(queries):
        assert q['id']==i and 0<=q['radius']<=design['max_radius']
        assert q['counting']==counting(q['radius'],len(q['word']),q['nonzero'])
        assert q['rule'] in (30,90) and q['status'] in ('SAT','UNSAT','UNKNOWN')
        if q['status']=='SAT':
            assert set(q['initial'])=={str(j) for j in range(-q['radius'],q['radius']+1)}
            assert all(v in (0,1) and type(v) is int for v in q['initial'].values())
            assert not q['nonzero'] or any(q['initial'].values())
            assert simulate(q['initial'],len(q['word']),q['rule'])==q['word']
    records=result['records']
    assert len(records)==171
    assert {(r['target'],r['length']) for r in records}=={(target,length) for target in ('alternating','zero','one') for length in range(8,65)}
    for r in records+result['random_controls']:
        word=r.get('word')
        if 'target' in r:
            assert r['rule']==30
            word=[i%2 for i in range(r['length'])] if r['target']=='alternating' else [int(r['target']=='one')]*r['length']
        assert len(word)==r['length']
        assert len(set(r['query_ids']))==len(r['query_ids']) and r['query_ids']
        qs=[queries[i] for i in r['query_ids']]
        assert all(q['word']==word and q['rule']==30 and q['nonzero'] for q in qs)
        lo,hi=r['lower_excluded'],r['sat_upper']
        assert lo==-1 or any(q['status']=='UNSAT' and q['radius']==lo for q in qs)
        assert hi is None or any(q['status']=='SAT' and q['radius']==hi for q in qs)
        if r['status']=='exact':
            radius=r['minimum_radius']
            assert hi==radius and lo==radius-1
            if r.get('target')=='zero':assert radius==2*(r['length']//2)-1
            if r.get('target')=='one':assert radius==2*((r['length']-1)//2)
            assert any(q['status']=='SAT' and q['radius']==radius for q in qs)
            assert radius==0 or any(q['status']=='UNSAT' and q['radius']==radius-1 for q in qs)
        else:
            assert r['status']=='censored' and r['minimum_radius'] is None
            assert hi is None or lo<hi
            assert any(q['status']=='UNKNOWN' for q in qs) or (hi is None and lo==design['max_radius'])
    assert len(result['random_controls'])==4
    rng=random.Random(20261014)
    for r,length in zip(result['random_controls'],(8,16,32,64)):
        assert r['length']==length and r['word']==[rng.randrange(2) for _ in range(length)]
    controls=[queries[i] for i in result['controls']]
    expected={(radius,bit,length,status) for radius in range(1,7) for bit in (0,1)
              for h in [2*((radius+1)//2)+1 if bit==0 else 2*(radius//2)+2]
              for length,status in ((h,'SAT'),(h+1,'UNSAT'))}
    assert len(controls)==24 and len(set(result['controls']))==24
    assert {(q['radius'],q['word'][0],len(q['word']),q['status']) for q in controls}==expected
    assert all(q['rule']==30 and q['nonzero'] and q['word']==[q['word'][0]]*len(q['word']) for q in controls)
    sanity=[queries[i] for i in result['sanity_controls']]
    assert len(sanity)==2
    assert any(q['radius']==0 and q['rule']==30 and not q['nonzero'] and q['word']==[0]*64 and q['status']=='SAT' for q in sanity)
    assert any(q['radius']==0 and q['rule']==90 and q['nonzero'] and q['word']==[1]+[0]*63 and q['status']=='SAT' for q in sanity)


def raw_bytes(path):
    return gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()


def check_query(q,base,drat):
    text,_=encode(q['radius'],q['word'],q['rule'],q['nonzero'])
    encoded=text.encode()
    assert hashlib.sha256(encoded).hexdigest()==q['cnf_sha256']
    if 'cnf' in q:
        assert raw_bytes(base/q['cnf'])==encoded
    if q['status']=='UNSAT':
        proof=raw_bytes(base/q['proof'])
        assert hashlib.sha256(proof).hexdigest()==q['proof_sha256']
        with tempfile.TemporaryDirectory() as directory:
            cnf_path=Path(directory)/'input.cnf'; proof_path=Path(directory)/'proof.drat'
            cnf_path.write_bytes(encoded); proof_path.write_bytes(proof)
            check_drat(cnf_path,proof_path,drat)


def split_archive(result,path):
    archive=path.parent/result['certificates_archive']
    if archive.stat().st_size<=20_000_000:
        return
    chunks=[]
    with archive.open('rb') as stream:
        while True:
            content=stream.read(5_000_000)
            if not content:break
            name=archive.name+f'.part{len(chunks):03d}'
            (path.parent/name).write_bytes(content)
            chunks.append({'path':name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
    result['certificates_chunks']=chunks
    archive.unlink()


def compact(path):
    result=json.loads(path.read_text())
    archive=path.parent/'certificates.tar.xz'
    members=[]
    for q in result['queries']:
        q['counting']=counting(q['radius'],len(q['word']),q['nonzero'])
        if q['status']=='UNSAT':
            q['drat_check']=q['proof'].replace('.drat','.drat-check.txt')
            q['drat_check_sha256']=digest(path.parent/q['drat_check'])
            members.extend(q[field] for field in ('cnf','proof','drat_check'))
        else:
            (path.parent/q.pop('cnf')).unlink(missing_ok=True)
    # Stream the archive: the checked DRATs can exceed one gigabyte raw.
    with lzma.open(archive,'wb',preset=3) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|') as tar:
            for name in sorted(members):
                source=path.parent/name
                info=tarfile.TarInfo(name); info.size=source.stat().st_size; info.mtime=0; info.mode=0o644
                with source.open('rb') as data:tar.addfile(info,data)
    result['certificates_archive']=archive.name
    result['certificates_archive_sha256']=digest(archive)
    split_archive(result,path)
    for p in (path.parent/'certificates').iterdir():
        p.unlink()
    (path.parent/'certificates').rmdir()
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(f'PASS compact: {len(members)} UNSAT CNF/DRAT/checklog members retained in deterministic {archive.name}')


def extract_certificates(result,path,directory):
    name=result['certificates_archive']
    assert Path(name).name==name
    archive=path.parent/name
    if 'certificates_chunks' in result:
        pieces=[]
        for chunk in result['certificates_chunks']:
            assert re.fullmatch(r'certificates\.tar\.xz\.part[0-9]+',chunk['path'])
            content=(path.parent/chunk['path']).read_bytes()
            if content.startswith(b'version https://git-lfs'):
                raise SystemExit(f"{chunk['path']} is a Git LFS pointer; run `git lfs pull` first")
            assert len(content)==chunk['bytes'] and hashlib.sha256(content).hexdigest()==chunk['sha256']
            pieces.append(content)
        content=b''.join(pieces)
        assert hashlib.sha256(content).hexdigest()==result['certificates_archive_sha256']
        archive_source=io.BytesIO(content)
    else:
        assert digest(archive)==result['certificates_archive_sha256']
        archive_source=None
    expected={q[field] for q in result['queries'] if q['status']=='UNSAT' for field in ('cnf','proof','drat_check')}
    with tarfile.open(archive if archive_source is None else None,'r:*',fileobj=archive_source) as tar:
        members=tar.getmembers()
        assert len(members)==len(expected) and {m.name for m in members}==expected
        for member in members:
            assert member.isfile() and re.fullmatch(r'certificates/q[0-9]+\.(cnf|drat|drat-check\.txt)',member.name)
            destination=directory/member.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(tar.extractfile(member).read())
    for q in result['queries']:
        if q['status']=='UNSAT':
            assert digest(directory/q['drat_check'])==q['drat_check_sha256']


def verify(path,drat):
    result=json.loads(path.read_text())
    validate_metadata(result)
    with tempfile.TemporaryDirectory() as directory:
        base=Path(directory) if 'certificates_archive' in result else path.parent
        if 'certificates_archive' in result:extract_certificates(result,path,base)
        for q in result['queries']:
            check_query(q,base,drat)
    print(f"PASS verify {len(result['queries'])} regenerated CNFs, SAT witnesses, all UNSAT DRAT proofs, minimum evidence, controls")


def tamper_test(path,drat):
    result=json.loads(path.read_text()); validate_metadata(result)
    mutations=[('target',lambda r:r['records'][0].update(target='zero')),
               ('length',lambda r:r['records'][0].update(length=9)),
               ('minimum',lambda r:r['records'][0].update(minimum_radius=999)),
               ('query identity',lambda r:r['queries'][0].update(id=999)),
               ('witness',lambda r:r['queries'][0]['initial'].update({'0':2})),
               ('censored bounds',lambda r:r['records'][0].update(status='censored',minimum_radius=None,lower_excluded=64))]
    for name,mutate in mutations:
        corrupted=copy.deepcopy(result); mutate(corrupted)
        try:validate_metadata(corrupted)
        except (AssertionError,KeyError,IndexError):pass
        else:raise AssertionError('tamper accepted: '+name)
    q=copy.deepcopy(next(q for q in result['queries'] if q['status']=='UNSAT'))
    q['proof_sha256']='0'*64
    with tempfile.TemporaryDirectory() as directory:
        base=Path(directory) if 'certificates_archive' in result else path.parent
        if 'certificates_archive' in result:extract_certificates(result,path,base)
        try:check_query(q,base,drat)
        except AssertionError:pass
        else:raise AssertionError('tamper accepted: proof digest')
    print('PASS tamper rejection: target, length, minimum, query identity, witness, censored bounds, proof digest')


def plot(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads(path.read_text())
    fig,ax=plt.subplots(figsize=(8,5))
    for target in ('alternating','zero','one'):
        rows=[r for r in data['records'] if r['target']==target and r['status']=='exact']
        ax.plot([r['length'] for r in rows],[r['minimum_radius'] for r in rows],label=target)
        censored=[r for r in data['records'] if r['target']==target and r['status']!='exact' and r['sat_upper'] is not None]
        if censored:
            ax.scatter([r['length'] for r in censored],[r['sat_upper'] for r in censored],marker='x',label=target+' censored SAT upper')
    ax.set(xlabel='Observed prefix length L (T=0)',ylabel='Minimum support radius R',title='Rule 30 finite horizon instrument; no eventuality claim')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(path.parent/'plot.png',dpi=140)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true'); ap.add_argument('--run',action='store_true')
    ap.add_argument('--verify',type=Path); ap.add_argument('--compact',type=Path); ap.add_argument('--split-certificates',type=Path); ap.add_argument('--tamper-test',type=Path); ap.add_argument('--output',type=Path,default=Path('runs/period-horizon-2026-10-14'))
    ap.add_argument('--cadical',type=Path,default=Path('third_party/cadical'))
    ap.add_argument('--drat-trim',type=Path,default=Path('third_party/drat-trim'))
    ap.add_argument('--budget',type=int,default=900); ap.add_argument('--timeout',type=int,default=10)
    ap.add_argument('--max-radius',type=int,default=64)
    args=ap.parse_args()
    args.cadical=args.cadical.resolve(); args.drat_trim=args.drat_trim.resolve()
    if args.self_test:self_test()
    if args.compact:compact(args.compact)
    if args.split_certificates:
        result=json.loads(args.split_certificates.read_text())
        split_archive(result,args.split_certificates)
        args.split_certificates.write_text(json.dumps(result,indent=2)+'\n')
    if args.verify:verify(args.verify,args.drat_trim)
    if args.tamper_test:tamper_test(args.tamper_test,args.drat_trim)
    if args.run:run(args)
    if not (args.self_test or args.verify or args.run or args.compact or args.tamper_test or args.split_certificates):ap.error('choose --self-test, --run or --verify')

if __name__=='__main__':main()
