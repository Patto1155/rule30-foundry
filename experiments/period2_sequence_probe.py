"""Exact finite-sequence diagnostics, never proof of an infinite recurrence."""
import argparse,json,math
from pathlib import Path
from fractions import Fraction

def rref(rows):
    a=[[Fraction(x) for x in row] for row in rows]; pivots=[]; r=0
    for c in range(len(a[0])):
        pivot=next((i for i in range(r,len(a)) if a[i][c]),None)
        if pivot is None:continue
        a[r],a[pivot]=a[pivot],a[r];scale=a[r][c];a[r]=[x/scale for x in a[r]]
        for i in range(len(a)):
            if i!=r and a[i][c]:
                scale=a[i][c];a[i]=[x-scale*y for x,y in zip(a[i],a[r])]
        pivots.append(c);r+=1
        if r==len(a):break
    return a,pivots

def nullspace(rows):
    a,p=rref(rows);out=[];dim=len(rows[0])
    for j in range(dim):
        if j in p:continue
        v=[Fraction(0)]*dim;v[j]=Fraction(1)
        for i,c in enumerate(p):v[c]=-a[i][j]
        out.append(v)
    return out

def solve(rows,b):
    a,p=rref([row+[y] for row,y in zip(rows,b)]);dim=len(rows[0])
    if dim in p:raise ValueError()
    v=[Fraction(0)]*dim
    for i,c in enumerate(p):v[c]=a[i][-1]
    return v

def bounded_design(dim):
    # Evaluate this before each no-fit search, not after interpreting its result.
    return {'purpose':'exact-exclusion','coefficient_box':[-8,8],
            'log2_class_upper':dim*math.log2(17),'not_complexity_evidence':True,
            'rational_coefficient_class':'infinite; handled by exact finite-design rank, not counting'}

def recurrence(values,start):
    split=max(12,len(values)*2//3); train=values[:split]; output=[]
    for order in range(1,6):
        for degree in range(4):
            dim=(order+1)*(degree+1)
            if len(train)-order<dim: continue
            gate=bounded_design(dim)
            def row(i):return [(start+i)**d*values[i+j] for j in range(order+1) for d in range(degree+1)]
            basis=nullspace([row(i) for i in range(split-order)]); alive=[];killed=[]
            for v in basis:
                # Reject relations missing the latest term.
                if not any(v[order*(degree+1)+d] for d in range(degree+1)): continue
                violation=next((start+i for i in range(split-order,len(values)-order) if sum(x*y for x,y in zip(row(i),v))!=0),None)
                rec={'coefficients':[str(x) for x in v], 'first_heldout_violation':violation}
                (alive if violation is None else killed).append(rec)
            full_basis=nullspace([row(i) for i in range(len(values)-order)])
            full_relations=[[str(x) for x in v] for v in full_basis if any(v[order*(degree+1)+d] for d in range(degree+1))]
            output.append({'full_data_relations':full_relations,'basiswise_heldout_is_not_family_exclusion':True,'order':order,'degree':degree,'training_terms':split,'dimension':dim,'nullity':len(basis),'survived':alive,'killed':killed,
                'counting':gate})
    return output

def digital(values,start):
    a=dict(enumerate(values,start));out=[]
    for k in (2,3):
        for j in range(k):
            pairs=[(n,k*n+j) for n in a if k*n+j in a]
            split=max(3,len(pairs)*2//3);train=pairs[:split]
            if len(train)<3:continue
            gate=bounded_design(3)
            m=[[a[n],n,1] for n,t in train]; b=[a[t] for n,t in train]
            try:
                solution=solve(m,b)
                fail=next((t for n,t in pairs[split:] if a[t]!=sum(x*y for x,y in zip([a[n],n,1],solution))),None)
                status='survived' if fail is None else 'killed'
                coefficients=[str(c) for c in solution]
            except ValueError:status='no_training_fit';coefficients=None;fail=None
            out.append({'k':k,'digit':j,'family':'a(kn+j)=c*a(n)+d*n+e','status':status,'training_pairs':len(train),'heldout_pairs':len(pairs)-split,'coefficients':coefficients,'first_heldout_violation':fail,'counting':gate})
    return out

def finite_kernel(values):
    out=[]
    for k in (2,3):
        for depth in (1,2):
            scale=k**depth; width=len(values)//scale
            if width<2:continue
            rows=[[values[k**e*n+j] for n in range(width)] for e in range(depth+1) for j in range(k**e)]
            _,pivots=rref(rows)
            out.append({"k":k,"depth":depth,"common_terms":width,"subsequences":len(rows),"rank":len(pivots),"scope":"rank of finite sampled kernel only; cannot exclude infinite k-regularity"})
    return out

def interpolate_plateaus(by):
    # Prefix feasibility is monotone: equal exact endpoint minima fix any gap.
    for target in {t for t,L in by}:
        rows=sorted([v for (t,L),v in by.items() if t==target],key=lambda r:r['length'])
        for low,high in zip(rows,rows[1:]):
            if low['radius']!=high['radius']:continue
            for length in range(low['length']+1,high['length']):
                by[target,length]={**high,'length':length,'inference':'prefix monotonicity between equal certified endpoint minima',
                    'inference_sources':[{k:r[k] for k in ('length','radius','source','query_id')} for r in (low,high)]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path);p.add_argument('--verify',type=Path);args=p.parse_args()
    if not(args.output or args.verify):p.error('choose --output or --verify')
    by={};witness=[]
    for path in args.inputs:
        d=json.loads(path.read_text())
        for r in d['records']:
            if r['status']!='exact' or not r['target'].startswith('alternating'):continue
            key=(r['target'],r['length'])
            if key in by: assert by[key]['radius']==r['minimum_radius']
            q=next(d['queries'][i] for i in r['query_ids'] if d['queries'][i]['status']=='SAT' and d['queries'][i]['radius']==r['minimum_radius'])
            by[key]={'length':r['length'],'radius':r['minimum_radius'],'source':str(path),'query_id':q['id'],'initial':q['initial'],'right_half':str(sum(v<< (int(i)-1) for i,v in q['initial'].items() if int(i)>0))}
    interpolate_plateaus(by)
    result={'scope':'bounded integer sequences with explicitly sourced prefix-monotonic plateau interpolation; no infinite k-regularity inference','targets':{}}
    for target in sorted({t for t,L in by}):
        rows=sorted([v for (t,L),v in by.items() if t==target],key=lambda v:v['length']);plateaus=[]
        for r in rows:
            if plateaus and r['length']==plateaus[-1]['end']+1 and r['radius']==plateaus[-1]['radius']:plateaus[-1]['end']=r['length']
            else:plateaus.append({'radius':r['radius'],'start':r['length'],'end':r['length']})
        edge={p[e] for p in plateaus for e in ('start','end')};witness.extend({'target':target,**r} for r in rows if r['length'] in edge)
        # Only contiguous initial block used by guessers; holes are never silently joined.
        block=[]
        for r in rows:
            if block and r['length']!=block[-1]['length']+1:break
            block.append(r)
        values=[r['radius'] for r in block];start=block[0]['length']
        result['targets'][target]={'exact_rows':[{k:v for k,v in r.items() if k!='initial'} for r in rows],'contiguous_start':start,'minima':values,'first_differences':[b-a for a,b in zip(values,values[1:])],'plateaus':plateaus,'plateau_starts':[p['start'] for p in plateaus],'plateau_ends':[p['end'] for p in plateaus],'recurrences':recurrence(values,start),'digital_affine':digital(values,start),'finite_kernel_ranks':finite_kernel(values)}
    if args.verify:
        assert json.loads(args.verify.read_text())==result
        assert json.loads(args.verify.with_name('edge-witnesses.json').read_text())==witness
        print('PASS exact regeneration: sourced plateau interpolation, sequence diagnostics, witnesses')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');args.output.with_name('edge-witnesses.json').write_text(json.dumps(witness,indent=2)+'\n')
    print({t:len(v['exact_rows']) for t,v in result['targets'].items()})
if __name__=='__main__': main()
