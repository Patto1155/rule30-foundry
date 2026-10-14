#!/usr/bin/env python3
"""Replay finite kill tests; surviving finite checks are never proofs."""
import argparse
import copy
from functools import lru_cache
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def engine(path=None):
    sys.path.insert(0, str(Path(path) if path else ROOT / 'experiments'))
    from support_growth import completion, self_test
    return lru_cache(maxsize=4096)(completion), self_test


def right_events(r,n,rule):
    out=[]
    for t in range(n):
        out.append(r&1)
        r=((r<<1)|(t&1))^((r|(r>>1)) if rule==30 else r>>1)
    return out


@lru_cache(maxsize=1024)
def stats(r,n,rule,completion):
    w=completion(r,n,rule=rule)
    prefix=[0]
    for j in range(n):prefix.append(prefix[-1]+((w>>j)&1))
    return {'prefix':prefix,'r':r,'n':n,'h':r.bit_length(),'word':w,
            'bits':[(w>>j)&1 for j in range(n)],'weight':w.bit_count(),
            's':right_events(r,n,rule)[::2]}


def evaluate(kind,x,completion,rule=30):
    r,n,h,b,w,s=x['r'],x['n'],x['h'],x['bits'],x['weight'],x['s']
    W=lambda k:x['prefix'][k]
    other=lambda q:completion(q,n,rule=rule).bit_count()
    start=min(n,2*h)
    if kind.startswith('exact7:'):
        seed=int(kind.split(':')[1])
        return (True if r!=seed else x['word']==sum(1<<j for j in range(0,n,7))), {'seed':seed,'support_residue':1,'modulus':7}
    if kind.startswith('right-alt:'):
        seed=int(kind.split(':')[1])
        return (True if r!=seed else all(z==(k&1) for k,z in enumerate(s))), {'seed':seed,'macrotime_count':len(s)}
    if kind=='alias32':return (True if r%64!=32 else x['word']==completion(32,n,rule=rule)), {'baseline_right':32,'required_residue':32,'modulus':64}
    if kind=='base32':return w>=other(32)-h, {'baseline':other(32),'credit':h}
    if kind=='density7nocredit':
        bad=next((j for j in range(1,n+1) if W(j)<(j+6)//7),None)
        return bad is None, {'bad_N':bad,'credit':0}
    if kind=='pair2': return (True if r%4!=1 else x['word']==completion(r^2,n,rule=rule)), {'other':r^2}
    if kind=='pair8': return (True if r%8!=2 else x['word']==completion(r^8,n,rule=rule)), {'other':r^8}
    if kind=='odd-base': return (True if not r&1 else w>=other(1)-h), {'baseline':other(1),'credit':h}
    if kind=='truncate': return abs(w-other(r&((1<<max(1,h//2))-1)))<=2*h, {'credit':2*h}
    if kind=='reset-charge': return 2*w>=sum(s)-2*h, {'reset_count':sum(s),'credit':2*h}
    if kind=='reset-window':
        bad=next((k for k,z in enumerate(s) if z and 2*k>=start and 2*k+8<=n and sum(b[2*k:2*k+8])==0),None)
        return bad is None, {'bad_macrotime':bad}
    if kind in ('gap3','gap5'):
        positions=[i for i,z in enumerate(s) if z];gap=int(kind[-1]);ct=sum(v-u==gap for u,v in zip(positions,positions[1:]))
        return w>=ct-h, {'gap':gap,'event_count':ct,'credit':h}
    if kind.startswith('zero-run'):
        limit=int(kind.split(':')[1]);bad=next((j for j in range(start,n-limit+1) if sum(b[j:j+limit])==0),None)
        return bad is None, {'forbidden_length':limit,'bad_depth':None if bad is None else bad+1}
    if kind=='pair-cover':
        bad=next((j for j in range(start,n-1) if b[j]+b[j+1]==0),None)
        return bad is None, {'bad_depth':None if bad is None else bad+1}
    if kind=='odd-cover':
        bad=next((j for j in range(start,n-3) if b[j]+b[j+2]==0),None)
        return bad is None, {'bad_depth':None if bad is None else bad+1}
    if kind.startswith('prefix:'):
        den=int(kind.split(':')[1]);bad=next((j for j in range(1,n+1) if den*W(j)<j-den*h),None)
        return bad is None, {'denominator':den,'bad_N':bad,'credit':h}
    if kind=='annulus':
        bad=next((j for j in range(1,n//2+1) if 8*(W(2*j)-W(j))<j-8*h),None)
        return bad is None, {'bad_N':bad,'credit':h,'denominator':8}
    if kind=='gap-credit':
        limit=2*h+8;bad=next((j for j in range(start,n-limit+1) if sum(b[j:j+limit])==0),None)
        return bad is None, {'gap_bound':limit,'bad_depth':None if bad is None else bad+1}
    if kind.startswith('coverage:'):
        k=int(kind.split(':')[1]);seen={tuple(b[j:j+k]) for j in range(start,n-k+1)}
        return len(seen)==1<<k, {'block_length':k,'distinct':len(seen),'required':1<<k}
    if kind=='period4':
        bad=next((j for j in range(start,n-31) if all(b[j+t]==b[j+t%4] for t in range(32))),None)
        return bad is None, {'bad_depth':None if bad is None else bad+1,'period':4,'window':32}
    if kind=='period8':
        bad=next((j for j in range(start,n-31) if all(b[j+t]==b[j+t%8] for t in range(32))),None)
        return bad is None, {'bad_depth':None if bad is None else bad+1,'period':8,'window':32}
    raise ValueError(kind)


def specs(round_number=1):
    fields=[('2-adic dynamics',[
        ('pair2','For every r=1 mod4 and N>=1, E_N(r)=E_N(r xor2).','A seed quotient could reduce the finite-right induction; equality alone does not imply growth.'),
        ('pair8','For every r=2 mod8 and N>=1, E_N(r)=E_N(r xor8).','Test a second OR-shielded seed quotient before generalizing it.'),
        ('odd-base','For every odd finite r and N>=1, W_N(r)>=W_N(1)-bitlength(r).','Transfers a separately proved base-seed bound to odd seeds.'),
        ('truncate','For every finite r,N>=1, |W_N(r)-W_N(r mod2^max(1,floor(h/2)))|<=2h, h=bitlength(r).','Would support induction on seed width; requires a separate base-seed theorem.')]),
      ('transfer operators and renewal theory',[
        ('reset-charge','For every finite r,N>=1, 2W_N(r)>=sum_{k<ceil(N/2)} bit0(H^k(r))-2h.','Verified NOR return forbids00000: this charging inequality would imply positive density after finite seed credit.'),
        ('reset-window','For every finite r and k>=h, if bit0(H^k(r))=1, at least one of e_(2k+1)..e_(2k+8) is1.','Bounded-overlap ordered event charging would imply a density lower bound using the NOR forbidden word.'),
        ('gap3','For every finite r,N, W_N(r)>=number of consecutive right-output1 gaps3 in first ceil(N/2) macrotimes-h.','If physical entry plus enough gap3 events is proved, transfers event count to left weight.'),
        ('gap5','For every finite r,N, W_N(r)>=number of consecutive right-output1 gaps5 in first ceil(N/2) macrotimes-h.','A complementary charge route; physical entry and positive gap5 density are separate obligations.')]),
      ('automata and regular languages',[
        ('zero-run:4','For every finite r, no 0000 occurs in e_j for j>=2h+1.','A language exclusion would give density at least1/4, conditional on a genuine OR-based proof.'),
        ('zero-run:8','For every finite r, no 00000000 occurs in e_j for j>=2h+1.','A weaker bounded-gap language gives density at least1/8.'),
        ('pair-cover','For every finite r,j>=2h+1, e_j+e_(j+1)>=1.','Tests whether the period-one checkerboard language survives in any tail form.'),
        ('odd-cover','For every finite r,j>=2h+1, e_j+e_(j+2)>=1.','Tests a parity-separated local cover, giving a density lower bound if proved.')]),
      ('potential or charge functions',[
        ('prefix:3','For every finite r,N>=1, 3W_N(r)>=N-3h.','An explicit seed-credit positive drift; numerical constant is a hypothesis, not fitted evidence.'),
        ('prefix:4','For every finite r,N>=1, 4W_N(r)>=N-4h.','Weaker positive drift; exact prefix counterexample kills this stated pair only.'),
        ('annulus','For every finite r,N>=1, 8(W_(2N)(r)-W_N(r))>=N-8h.','Dyadic charges imply density by summing annuli; avoids the refuted width3 cut histogram potential.'),
        ('gap-credit','For every finite r,j>=2h+1, the next2h+8 completion bits contain a1.','Seed-dependent bounded gaps imply alpha(r)>=1/(2h+8), with a separately proved OR mechanism.')]),
      ('symbolic dynamics and entropy',[
        ('coverage:3','For every finite r,N>=max(64,4h+16), the tail e_(2h+1)..e_N contains all eight binary length3 words.','A recurrent language target; coverage alone is not support growth, and no entropy theorem follows.'),
        ('coverage:4','For every finite r,N>=max(64,4h+16), the tail e_(2h+1)..e_N contains all sixteen binary length4 words.','Sharper recurrent language test, stronger than needed and independently falsifiable.'),
        ('period4','For every finite r,j>=2h+1, the length32 completion block at j is not4-periodic.','A finite window sufficient for aperiodic completion, stronger than needed; bounded findings only.'),
        ('period8','For every finite r,j>=2h+1, the length32 completion block at j is not8-periodic.','Tests a larger periodic sublanguage; it cannot justify positive entropy from finite data.')])]
    out=[]
    for field,items in fields:
        for kind,statement,unlock in items:
            out.append({'id':'p2-r%d-%s'%(round_number,kind.replace(':','-')),'round':round_number,'field':field,'predicate':kind,'statement':statement,'unlocks':unlock,'status':'untested','filter':{'Rule90':'Paired same-seed Rule90 replay. A Rule90 survivor is inadmissible until a separate OR-specific proof is supplied.','zero':'Target is exactly alternating from centre0; identically zero target/right seed is excluded. Constants depend on the finite right seed h; no random-IC conclusion.','OR_mechanism':'Sideways inversion uses a=c_next xor(c OR d) and physical H=R1 composed R0. Event candidates additionally charge the verified NOR reset; other candidates still require an OR-specific derivation.'},'kill_test':'python experiments/period2_candidates.py --verify queue/theory/candidates.json; predicate '+kind})
    return out


def round_two():
    items=[('prefix:6','For every finite r,N>=1, 6W_N(r)>=N-6h.','Test how1/7 slow rows constrain any universal density.'),
           ('prefix:7','For every finite r,N>=1, 7W_N(r)>=N-7h.','A candidate sharp uniform slope with seed-width credit; much stronger than existential seed-specific density.'),
           ('density7nocredit','For every finite r,N>=1, W_N(r)>=ceil(N/7).','Kill the tempting extrapolation of the exact minimum checkpoint curve without initial credit.'),
           ('alias32','For every finite r=32 mod64,N>=1, E_N(r)=E_N(32).','Test whether the observed slow seeds form a complete residue cylinder.'),
           ('base32','For every finite r,N>=1, W_N(r)>=W_N(32)-h.','A minimum-seed comparison would reduce universal growth to the slow periodic completion plus finite credit.'),
           ('zero-run:7','For every finite r, no0000000 occurs after depth2h.','Test a stronger bounded-gap explanation of the observed1/7 density.'),
          ]
    for seed in (32,96,160):
        items.append(('exact7:'+str(seed),'For every N>=1, e_j('+str(seed)+')=1 iff j=1 mod7 for1<=j<=N.','An exact seed-conditioned lemma, not a universal exclusion; needs a finite right grammar proof.'))
        items.append(('right-alt:'+str(seed),'For every k>=0, bit0(H^k('+str(seed)+'))=k mod2.','This explicit physical boundary lemma could couple the period7 left ring to the alternating centre and prove its exact infinite completion.'))
    template=specs()[0];out=[]
    for kind,statement,unlock in items:
        c=copy.deepcopy(template);c.update(id='p2-r2-'+kind.replace(':','-'),round=2,field='data-conditioned recombination',predicate=kind,statement=statement,unlocks=unlock,kill_test='python experiments/period2_candidates.py --verify queue/theory/candidates.json; predicate '+kind)
        out.append(c)
    return out


def test_candidate(c,seeds,n,completion):
    for r in seeds:
        x=stats(r,n,30,completion)
        if c['predicate'].startswith('coverage') and n<max(64,4*x['h']+16):continue
        ok,evidence=evaluate(c['predicate'],x,completion)
        if not ok:
            c.update(status='killed',counterexample={'right':str(r),'depth':n,'completion_hex':hex(x['word']),'weight':x['weight'],'evidence':evidence})
            return
    c['status']='survived'
    c['scope']={'seed_count':len(seeds),'depth':n,'meaning':'Passed only these bounded kill tests; no universal claim established.'}
    fail90=None
    for r in seeds:
        x=stats(r,n,90,completion)
        if c['predicate'].startswith('coverage') and n<max(64,4*x['h']+16):continue
        if not evaluate(c['predicate'],x,completion,90)[0]:fail90=str(r);break
    c['rule90_replay']={'counterexample_right':fail90,'meaning':'Distinct finite control response' if fail90 is not None else 'Filter unresolved: survives Rule90 too; do not promote'}


def verify(data,completion):
    assert completion(0,256,target='zero',rule=30)==0
    assert completion(1,256,target='zero',rule=90)==1
    assert completion(1,256,target='zero',rule=30).bit_count()>100
    assert len({c['id'] for c in data['candidates']})==len(data['candidates']), 'duplicate ids'
    for c in data['candidates']:
        if c['status']=='killed':
            z=c['counterexample'];x=stats(int(z['right']),z['depth'],30,completion)
            assert hex(x['word'])==z['completion_hex'] and x['weight']==z['weight']
            ok,evidence=evaluate(c['predicate'],x,completion)
            assert not ok and evidence==z['evidence'],c['id']
        elif c['status']=='survived':
            for r in data['rounds'][str(c['round'])]['seeds']:
                x=stats(int(r),c['scope']['depth'],30,completion)
                if c['predicate'].startswith('coverage') and x['n']<max(64,4*x['h']+16):continue
                assert evaluate(c['predicate'],x,completion)[0],c['id']
            control=c.get('rule90_replay',{}).get('counterexample_right')
            if control is not None:
                x=stats(int(control),c['scope']['depth'],90,completion)
                assert not evaluate(c['predicate'],x,completion,90)[0],c['id']
        else: assert c['status']=='untested'
    # An altered retained witness is rejected; not merely re-evaluated loosely.
    killed=next((c for c in data['candidates'] if c['status']=='killed'),None)
    if killed:
        tampered=copy.deepcopy(killed);tampered['counterexample']['completion_hex']=hex(int(tampered['counterexample']['completion_hex'],16)^1)
        try:verify({'candidates':[tampered],'rounds':data['rounds']},completion)
        except AssertionError:pass
        else:raise AssertionError('tampered certificate accepted')


def main():
    p=argparse.ArgumentParser();p.add_argument('--support-dir');p.add_argument('--verify');p.add_argument('--output',default='queue/theory/candidates.json');p.add_argument('--depth',type=int,default=256);p.add_argument('--support-data');p.add_argument('--round2',action='store_true');args=p.parse_args()
    completion,self_test=engine(args.support_dir)
    if args.verify:
        data=json.loads(Path(args.verify).read_text());verify(data,completion)
    else:
        seeds=list(range(64))
        if args.support_data:
            b=json.loads(Path(args.support_data).read_text());seeds=sorted(set(seeds+[int(x['right']) for x in b['rows']]))
        if args.round2:
            data=json.loads(Path(args.output).read_text())
            candidates=round_two()
            seeds=sorted(set(seeds+[32,96,160,224]))
            for c in candidates:test_candidate(c,seeds,args.depth,completion)
            data['rounds']['2']={'seeds':[str(r) for r in seeds],'depth':args.depth,'motivation':'B minimum density1/7 at32,96,160 and C plateau witnesses with these right halves. Hypotheses selected after seeing those data.'}
            data['candidates']=[c for c in data['candidates'] if c['round']!=2]+candidates
            verify(data,completion);Path(args.output).write_text(json.dumps(data,indent=2)+'\n')
            print(json.dumps({'result':'PASS','round2_counts':{s:sum(c['status']==s for c in candidates) for s in ('killed','survived','untested')}}));return
        candidates=specs()
        for c in candidates:test_candidate(c,seeds,args.depth,completion)
        data={'schema':1,'label':'lemma discovery over finite rows; not a single-seed measurement','threshold_policy':'All constants are explicit prospective conjectures, not empirical fits. A killed pair does not refute existential alpha(r),C(r).','rounds':{'1':{'seeds':[str(r) for r in seeds],'depth':args.depth}},'candidates':candidates,'not_run':['infinite completion proof','positive entropy proof','recurrence or automaton fitting/no-fit search','outside-family audit']}
        verify(data,completion);Path(args.output).write_text(json.dumps(data,indent=2)+'\n')
    counts={s:sum(c['status']==s for c in data['candidates']) for s in ('killed','survived','untested')}
    print(json.dumps({'result':'PASS','counts':counts,'tamper_rejection':'PASS'}))
if __name__=='__main__':main()
