#!/usr/bin/env python3
"""Exact packed-time triangular completion; finite-row lemma discovery only.

Bit j-1 of completion(r,N) is the initial cell at coordinate -j.
No finite measurement settles the existential asymptotic density conjecture.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

LABEL = 'lemma discovery over finite rows; not a single-seed measurement'


def target_word(depth, target):
    if target not in ('zero', 'one', 'alternating'):
        raise ValueError('unknown target')
    if target == 'zero':
        return 0
    if target == 'one':
        return (1 << (depth + 1)) - 1
    # finite alternating time word; t=0 is zero
    return sum(1 << t for t in range(1, depth + 1, 2))


def completion(right, depth, target='alternating', rule=30):
    """Sideways solve a_t=c_(t+1) XOR (c_t OR d_t) for Rule30.

    Positive-half updates are driven by the prescribed centre. The temporal
    columns then shorten one bit per negative depth, exactly the triangular
    light cone. Rule90 instead has a_t=c_(t+1) XOR d_t.
    """
    if right < 0 or depth < 0 or rule not in (30, 90):
        raise ValueError('nonnegative finite seed/depth and rule 30 or 90 required')
    c = target_word(depth, target)
    d = 0
    r = right
    for t in range(depth):
        d |= (r & 1) << t
        b = (c >> t) & 1
        r = ((r << 1) | b) ^ ((r | (r >> 1)) if rule == 30 else (r >> 1))
    # One packed bit for each time: solving the local left-permutive equation
    # creates a temporal column one bit shorter. Its bit0 is the next initial
    # negative cell. Only the triangular light cone is retained, so discarded
    # high-time bits cannot influence any claimed initial completion bit.
    result = 0
    mask = (1 << depth) - 1
    for j in range(depth):
        a = ((c >> 1) ^ ((c | d) if rule == 30 else d)) & mask
        result |= (a & 1) << j
        d, c = c, a
        mask >>= 1
    return result


def forward_check(right, left, depth, target='alternating', rule=30):
    """Independent cell dictionary truth-table simulation (small depths)."""
    table = (0, 1, 1, 1, 1, 0, 0, 0) if rule == 30 else (0, 1, 0, 1, 1, 0, 1, 0)
    row = {0: target_word(depth, target) & 1}
    row.update({j + 1: (right >> j) & 1 for j in range(right.bit_length())})
    row.update({-j - 1: (left >> j) & 1 for j in range(depth)})
    for t in range(depth + 1):
        assert row.get(0, 0) == ((target_word(depth, target) >> t) & 1), (right, depth, t)
        lo, hi = min(row), max(row)
        row = {i: table[(row.get(i-1, 0) << 2) | (row.get(i, 0) << 1) | row.get(i+1, 0)]
               for i in range(lo-1, hi+2)}


def self_test():
    try:
        from .period2_identity_check import forced_left, zero_completion
    except ImportError:
        from period2_identity_check import forced_left, zero_completion
    count = 0
    for rule in (30, 90):
        for target in ('zero', 'one', 'alternating'):
            for r in range(32):
                n = 24
                word = completion(r, n, target, rule)
                forward_check(r, word, n, target, rule)
                assert completion(r, 13, target, rule) == word & ((1 << 13)-1)
                if rule == 30:
                    expected = forced_left(r, [(target_word(n, target) >> t)&1 for t in range(n+1)], n)
                    assert word == sum(x << j for j, x in enumerate(expected))
                    if target == 'zero':
                        assert expected == zero_completion(r, n)
                count += 1
    # Guard across packed-word boundaries with independent cell simulation.
    forward_check(0x12345, completion(0x12345, 80), 80)
    assert completion(0, 256, 'zero') == 0
    assert completion(1, 256, 'zero', 90) == 1  # finite two-cell constant-zero fibre
    return {'cases': count, 'depth': 24, 'independent_forward': 'PASS',
            'reference_inversion': 'PASS', 'prefix_consistency': 'PASS',
            'zero_control': 'PASS', 'rule90_finite_constant_zero_completion': 'r=1, left=1 at depth1 only'}


def record(r, n, source):
    start = time.perf_counter()
    word = completion(r, n)
    checkpoints = sorted({x for x in (16, 32, 64, 128, 256, 512, 1024, 4096, 10000, 25000, 50000, 100000, n) if x<=n})
    weights = [[x, (word & ((1<<x)-1)).bit_count()] for x in checkpoints]
    raw = word.to_bytes((n+7)//8, 'little')
    return {'right': str(r), 'depth': n, 'source': source, 'weight': word.bit_count(),
            'slope': word.bit_count()/n, 'checkpoints': weights,
            'increments': [[weights[i][0]-weights[i-1][0],weights[i][1]-weights[i-1][1]] for i in range(1,len(weights))],
            'completion_hex_little_endian': raw.hex(), 'sha256': hashlib.sha256(raw).hexdigest(),
            'elapsed_seconds': time.perf_counter()-start}


def witnesses(path):
    qs = json.loads(Path(path).read_text())['queries']
    out = set()
    for q in qs:
        if q['status'] == 'SAT' and len(q['word']) > 3 and q['word'] == [t%2 for t in range(len(q['word']))]:
            out.add(sum(int(v) << (int(k)-1) for k,v in q['initial'].items() if int(k)>0))
    return sorted(out)


def run(args):
    checks = self_test()
    start = time.perf_counter()
    seeds = witnesses(args.witnesses)
    original_seed_count = len(seeds)
    converted = []
    if args.extra_witnesses:
        for q in json.loads(Path(args.extra_witnesses).read_text()):
            r = int(q['right_half'])
            if q['target'] == 'alternating-one':
                r = ((r << 1) | 1) ^ (r | (r >> 1))
            elif q['target'] != 'alternating':
                raise ValueError('extra witness phase unsupported')
            converted.append({'original_right': q['right_half'], 'phase': q['target'],
                              'length': q['length'], 'converted_right': str(r)})
        seeds = sorted(set(seeds) | {int(q['converted_right']) for q in converted})
    rows = [record(r,args.witness_depth,'SAT alternating witness') for r in seeds]
    # Enumeration is weight measurement, not a no-fit model-class search.
    best = []
    for r in range(1<<args.enumerate_bits):
        w = completion(r,args.enum_depth).bit_count()
        best.append((w,r))
    best.sort()
    enum_worst = best[:16]
    rng = random.Random(30)
    sampled = [rng.randrange(1<<24) for _ in range(args.samples)]
    sample_best = sorted((completion(r,args.search_depth).bit_count(),r) for r in sampled)
    # Deterministic coordinate descent: strictly improving, bounded evaluations.
    adversarial=[]
    for _,initial in sample_best[:8]+enum_worst[:8]:
        r=initial
        w=completion(r,args.search_depth).bit_count()
        for _ in range(8):
            candidates=[(completion(r^(1<<b),args.search_depth).bit_count(),r^(1<<b)) for b in range(24)]
            nw,nr=min(candidates)
            if nw>=w: break
            w,r=nw,nr
        adversarial.append((w,r))
    witness_slowest = sorted((row['weight'],int(row['right'])) for row in rows)[:8]
    long_seeds=sorted(set(r for _,r in enum_worst[:4]+sample_best[:4]+sorted(adversarial)[:8]+witness_slowest))
    rows += [record(r,args.long_depth,'enumeration/sample/adversarial finalist') for r in long_seeds]
    controls={'rule30_constant_zero': [{'right':r,'depth':256,'weight':completion(r,256,'zero').bit_count()} for r in (0,1,2,7,65535)],
              'rule90_constant_zero': {'right':1,'depth':256,'weight':completion(1,256,'zero',90).bit_count(),'finite_completion_depth':1}}
    result={'label':LABEL,'instrument':'packed temporal sideways inversion','self_test':checks,
            'scope':{'enumerated_right_seeds':1<<args.enumerate_bits,'enumeration_depth':args.enum_depth,'sampled_24bit_seeds':args.samples,'sample_depth':args.search_depth,'random_seed':30,'original_SAT_witness_seed_count':original_seed_count,'combined_witness_seed_count':len(seeds),'extra_witness_input_count':len(converted),'long_prefix_count':len(long_seeds),'long_depth':args.long_depth,'long_seed_selection':'eight worst short-horizon SAT witnesses plus eight adversarial finalists plus four enumeration and sample finalists (deduplicated); deep global minimum not claimed','adversarial_bit_width':24,'adversarial_max_rounds':8},
            'enumeration_worst':enum_worst,'sample_worst':sample_best[:16],'adversarial_finalists':sorted(adversarial),'rows':rows,'controls':controls,'new_horizon_witnesses':converted,
            'elapsed_seconds':time.perf_counter()-start,
            'interpretation':'Exact bounded prefixes only. Finite low slopes do not refute forall r exists alpha(r)>0,C(r) forall N. No period-two exclusion.',
            'not_run':['infinite support-growth proof','single-seed prize measurement','GPU','external-family audit']}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('scope','elapsed_seconds','enumeration_worst')},indent=2))


def verify(path):
    data=json.loads(Path(path).read_text())
    for row in data['rows']:
        replay=record(int(row['right']),row['depth'],row['source'])
        for k in ('weight','slope','checkpoints','increments','sha256','completion_hex_little_endian'):
            assert replay[k]==row[k],(row['right'],k)
    for q in data.get('new_horizon_witnesses', []):
        r=int(q['original_right'])
        if q['phase']=='alternating-one':
            r=((r<<1)|1)^(r|(r>>1))
        assert str(r)==q['converted_right']
    assert data['label'] == LABEL
    scope = data['scope']
    original = set(witnesses('runs/period-horizon-2026-10-14/results.json'))
    combined = original | {int(q['converted_right']) for q in data.get('new_horizon_witnesses', [])}
    assert scope['original_SAT_witness_seed_count'] == len(original)
    assert scope['combined_witness_seed_count'] == len(combined)
    assert scope['extra_witness_input_count'] == len(data.get('new_horizon_witnesses', []))
    assert combined <= {int(row['right']) for row in data['rows']}
    assert scope['long_prefix_count'] == sum(row['depth']==scope['long_depth'] and row['source']=='enumeration/sample/adversarial finalist' for row in data['rows'])
    enum = sorted((completion(r,scope['enumeration_depth']).bit_count(),r) for r in range(scope['enumerated_right_seeds']))
    assert [list(x) for x in enum[:16]] == data['enumeration_worst']
    rng = random.Random(scope['random_seed'])
    sample = sorted((completion(r,scope['sample_depth']).bit_count(),r) for r in [rng.randrange(1<<24) for _ in range(scope['sampled_24bit_seeds'])])
    assert [list(x) for x in sample[:16]] == data['sample_worst']
    adversarial = []
    for _,initial in sample[:8]+enum[:16][:8]:
        r=initial; w=completion(r,scope['sample_depth']).bit_count()
        for _ in range(scope['adversarial_max_rounds']):
            nw,nr=min((completion(r^(1<<b),scope['sample_depth']).bit_count(),r^(1<<b)) for b in range(scope['adversarial_bit_width']))
            if nw>=w: break
            w,r=nw,nr
        adversarial.append((w,r))
    assert [list(x) for x in sorted(adversarial)] == data['adversarial_finalists']
    for control in data['controls']['rule30_constant_zero']:
        assert completion(control['right'],control['depth'],'zero').bit_count()==control['weight']
    control=data['controls']['rule90_constant_zero']
    assert completion(control['right'],control['depth'],'zero',90)==1
    assert control['weight']==1 and control['finite_completion_depth']==1
    print(json.dumps({'result':'PASS','rows_replayed':len(data['rows']),'self_test':self_test()}))


def main():
    p=argparse.ArgumentParser(description=LABEL)
    p.add_argument('--self-test',action='store_true');p.add_argument('--verify')
    p.add_argument('--witnesses',default='runs/period-horizon-2026-10-14/results.json')
    p.add_argument('--extra-witnesses')
    p.add_argument('--output',default='runs/support-growth-2026-10-14/results.json')
    p.add_argument('--enumerate-bits',type=int,default=16);p.add_argument('--enum-depth',type=int,default=128)
    p.add_argument('--samples',type=int,default=256);p.add_argument('--search-depth',type=int,default=1024)
    p.add_argument('--witness-depth',type=int,default=4096);p.add_argument('--long-depth',type=int,default=100000)
    args=p.parse_args()
    if args.self_test: print(json.dumps(self_test()))
    elif args.verify: verify(args.verify)
    else: run(args)

if __name__=='__main__': main()
