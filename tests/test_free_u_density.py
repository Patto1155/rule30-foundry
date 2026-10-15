"""Independent finite truth-table and cardinality checks of the SAT instrument."""
import itertools
import os
import unittest
from pysat.solvers import Solver
from experiments.free_u_density import encode, replay, reference

class FreeDensityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('COLUMN_REFERENCE')
        cls.ref=reference(path)

    def test_exhaustive_small_inputs(self):
        for n in range(1,13):
            enc=encode(n)
            with Solver(name='cadical195',bootstrap_with=enc.clauses) as solver:
                for bits in itertools.product((0,1),repeat=len(enc.u_vars)):
                    assumptions=[v if b else -v for v,b in zip(enc.u_vars,bits)]
                    self.assertTrue(solver.solve(assumptions=assumptions))
                    record=replay(enc,solver.get_model(),self.ref)
                    self.assertEqual(int(record['completion_hex'],16),self.ref.completion_from_u(bits,n))
                    expected=int(record['completion_hex'],16)
                    # Uniqueness: each opposite initial completion bit is impossible.
                    for j,v in enumerate(enc.e_vars):
                        self.assertFalse(solver.solve(assumptions=assumptions+[-v if (expected>>j)&1 else v]))

    def test_exact_small_minima(self):
        for n in range(1,17):
            enc=encode(n)
            minimum=min(self.ref.completion_from_u(bits,n).bit_count() for bits in itertools.product((0,1),repeat=len(enc.u_vars)))
            for k,want in ((minimum,True),(minimum-1,False)):
                with Solver(name='cadical195',bootstrap_with=enc.bound(k)) as solver:
                    self.assertEqual(solver.solve(),want,(n,k))

    def test_checkpoint_witnesses(self):
        for n,text,weight in [(81,'01010100110101101101010010010001100100010',9),(79,'01'*17+'0'*6,10)]:
            enc=encode(n)
            with Solver(name='cadical195',bootstrap_with=enc.bound(weight)) as solver:
                self.assertTrue(solver.solve(assumptions=[v if b=='1' else -v for v,b in zip(enc.u_vars,text)]))
                self.assertEqual(replay(enc,solver.get_model(),self.ref)['weight'],weight)

    def test_controls(self):
        for rule,target in ((30,'zero'),(90,'zero'),(90,'alternating')):
            enc=encode(18,rule,target)
            self.assertEqual(len(enc.u_vars),18)
            with Solver(name='cadical195',bootstrap_with=enc.clauses) as solver:
                self.assertTrue(solver.solve(assumptions=[-v for v in enc.u_vars]))
                record=replay(enc,solver.get_model(),self.ref)
                if target=='zero': self.assertEqual(record['weight'],0)

if __name__=='__main__': unittest.main()
