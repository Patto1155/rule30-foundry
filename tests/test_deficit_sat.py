"""Small exhaustive model validation, including all-prefix disjunction."""
import itertools
import copy
import json
import tempfile
from types import SimpleNamespace
import os
from pathlib import Path
import unittest
from pysat.solvers import Solver
from experiments import deficit_sat as d
from experiments.support_growth import completion


class DeficitSATTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=d.load(Path(os.environ.get('DEFICIT_ENCODING',Path(d.__file__).with_name('free_u_density.py'))),'free_u_test')

    def test_exhaustive_physical_and_prefix_query(self):
        for rule,target in itertools.product((30,90),('alternating','zero')):
            for r in range(8):
                N=12; K=3; e=completion(r,N,target,rule)
                maximum=max(j-7*(e&((1<<j)-1)).bit_count() for j in range(1,N+1))
                for threshold in (1,maximum,maximum+1):
                    enc=d.encode(N,K,threshold,self.base,rule,target,r)
                    with Solver(name='glucose3',bootstrap_with=enc.clauses) as solver:
                        sat=solver.solve()
                        self.assertEqual(sat,maximum>=threshold,(rule,target,r,threshold))
                        if sat:
                            yes={v for v in solver.get_model() if v>0}
                            actual=sum(int(v in yes)<<i for i,v in enumerate(enc.e_vars))
                            self.assertEqual(actual,e)

    def test_replay_rejects_semantic_and_witness_tampering(self):
        enc=d.encode(12,0,1,self.base,target='zero',fixed_right=0)
        with Solver(name='glucose3',bootstrap_with=enc.clauses) as solver:
            self.assertTrue(solver.solve()); model=solver.get_model()
        ref=SimpleNamespace(__file__=d.__file__)
        row={'K':0,'horizon_max':12,'threshold':1,'rule':30,'target':'zero',
             'fixed_right':0,'status':'SAT','variables':enc.nv,'clauses':len(enc.clauses),
             'counting':{'initial_seed_bits':0,'free_input_bits':0,'class_size':'1'},
             'cnf':'q.cnf','witness':d.replay(enc,model,12,1,ref,target='zero')}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'q.cnf').write_text(enc.dimacs())
            row['cnf_sha256']=d.sha(root/'q.cnf')
            record={'reference_sha256':d.sha(ref.__file__),
                    'encoding_sha256':d.sha(self.base.__file__),'queries':[row]}
            path=root/'results.json'
            def verify(data):
                path.write_text(json.dumps(data)); return d.verify_results(path,None,ref,self.base)
            self.assertEqual(verify(record)['verified_queries'],1)
            for field,value in [('K',1),('horizon_max',13)]:
                changed=copy.deepcopy(record); changed['queries'][0][field]=value
                with self.assertRaises(AssertionError): verify(changed)
            changed=copy.deepcopy(record); changed['queries'][0]['witness']['W']=1
            with self.assertRaises(AssertionError): verify(changed)
            changed=copy.deepcopy(record); changed['reference_sha256']='0'*64
            with self.assertRaises(AssertionError): verify(changed)
            changed=copy.deepcopy(record); changed['encoding_sha256']='0'*64
            with self.assertRaises(AssertionError): verify(changed)
            changed=copy.deepcopy(record); changed['queries'][0]['status']='UNKNOWN'
            with self.assertRaises(AssertionError): verify(changed)

    def test_support_limit(self):
        with self.assertRaises(ValueError): d.encode(12,3,1,self.base,fixed_right=8)

if __name__=='__main__': unittest.main()
