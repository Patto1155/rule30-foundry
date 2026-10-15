"""Independent truth-table and reference checks of finite boundary encodings."""
import importlib.util
import itertools
import json
import tempfile
from pathlib import Path
import random
import unittest
from experiments import u_language as ul
from pysat.solvers import Solver


class ULanguageTests(unittest.TestCase):
    def test_packed_against_naive_boundary(self):
        rng=random.Random(30)
        for _ in range(40):
            r=rng.getrandbits(100)
            row=[(r>>j)&1 for j in range(100)] + [0]*65
            observed=[]
            for t in range(63):
                if t%2==0:
                    observed.append(row[0])
                row=[(t%2 if j==0 else row[j-1]) ^ (row[j]|row[j+1])
                     for j in range(len(row)-1)]
            self.assertEqual(tuple(observed),ul.boundary(r,32))

    def test_cone_against_all_small_truth_tables(self):
        for length in range(1,5):
            T=2*length-1
            for bits in itertools.product('01',repeat=length):
                word=''.join(bits)
                _,clauses,initial=ul.cone_cnf(word)
                with Solver(name='g4',bootstrap_with=clauses) as solver:
                    for r in range(1<<T):
                        assumptions=[v if (r>>j)&1 else -v for j,v in enumerate(initial)]
                        self.assertEqual(solver.solve(assumptions=assumptions),
                                         ''.join(map(str,ul.boundary(r,length)))==word)

    def test_known_cones_and_controls(self):
        for word in ('11','00000','101001','0100101','0101000101'):
            _,clauses,_=ul.cone_cnf(word)
            with Solver(name='g4',bootstrap_with=clauses) as solver:
                self.assertFalse(solver.solve(),word)
        for word,rule,alternating in [('11',90,True),('00000',30,False)]:
            _,clauses,_=ul.cone_cnf(word,rule,alternating)
            with Solver(name='g4',bootstrap_with=clauses) as solver:
                self.assertTrue(solver.solve())

    def test_language_count_and_clauses(self):
        bans=['11','00000','101001','0100101']
        for n in range(1,10):
            words=[''.join(x) for x in itertools.product('01',repeat=n)]
            expected=[x for x in words if all(b not in x for b in bans)]
            self.assertEqual(ul.language_size(n,bans),len(expected))
            with Solver(name='g4',bootstrap_with=ul.language_clauses(list(range(1,n+1)),bans)) as solver:
                for word in words:
                    self.assertEqual(solver.solve(assumptions=[j+1 if b=='1' else -(j+1) for j,b in enumerate(word)]),word in expected)

    def test_numpy_enumeration_matches_python(self):
        result=ul.enumerate_physical(bits=5,samples=0,max_length=8)
        expected=ul.factor_sets((ul.boundary(r,32) for r in range(32)),8)
        self.assertEqual(result['combined_factor_counts'],[len(x) for x in expected[1:]])

    def test_saved_verifier_rejects_tampered_horizon_and_witness(self):
        path=Path(__file__).resolve().parents[1]/'runs/p2-u-language-2026-10-15/language.json'
        for kind in ('horizon','witness'):
            data=json.loads(path.read_text())
            if kind=='horizon':
                data['exact_cones'][0]['independent_initial_bits']+=2
            else:
                data['exhaustive_factor_witnesses']['12'].pop(next(iter(data['exhaustive_factor_witnesses']['12'])))
            with tempfile.TemporaryDirectory() as td:
                Path(td,'language.json').write_text(json.dumps(data))
                with self.assertRaises(AssertionError):
                    ul.verify_saved(td,'/unused/checker')

    def test_minimal_absence_is_finite_scope(self):
        seen=ul.factor_sets(['0010010'],3)
        absent=ul.minimal_missing(seen)
        self.assertEqual(absent['2'],['11'])
        self.assertEqual(absent['3'],['000','101'])
        self.assertEqual(ul.boundary(0,5),(0,1,0,0,0))

if __name__=='__main__':
    unittest.main()
