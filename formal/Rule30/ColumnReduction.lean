import Rule30.Basic

namespace Rule30

/-- Rule 30's left-permutive inversion, without any orbit assumption. -/
theorem column_left_inversion (a b c : Bool) :
    a = localRule a b c ^^ (b || c) := by
  cases a <;> cases b <;> cases c <;> rfl

/-- The erased odd column is the LEFT neighbor. -/
theorem odd_left_latch {x : Configuration}
    (hc : x 0 = true) (hn : step x 0 = false) : x (-1) = true := by
  have h : localRule (x (-1)) true (x 1) = false := by
    simpa [step, hc] using hn
  cases hl : x (-1) <;> simp_all [localRule]

/-- Right-neighbor two-step return under centres zero then one. -/
theorem column_right_return {x : Configuration}
    (h0 : x 0 = false) (h1 : step x 0 = true) :
    step (step x) 1 = (!(x 1) && !(x 2) && !(x 3)) := by
  change localRule (step x 0) (step x 1) (step x 2) = _
  rw [h1]
  change localRule true (localRule (x 0) (x 1) (x 2))
    (localRule (x 1) (x 2) (x 3)) = _
  rw [h0]
  exact alternating_return (x 1) (x 2) (x 3)

/-- Local two-step form of forbidden factor 11 in even right output. -/
theorem column_no11 {x : Configuration}
    (h0 : x 0 = false) (h1 : step x 0 = true) (hu : x 1 = true) :
    step (step x) 1 = false := by
  rw [column_right_return h0 h1, hu]
  rfl

/-- Corrected odd-left assertion along a normalized alternating orbit. -/
theorem alternating_odd_left {x : Configuration}
    (h : ∀ n : Nat, trace x n = decide (n % 2 = 1)) (s : Nat) :
    evolve x (2*s+1) (-1) = true := by
  apply odd_left_latch
  · simpa [trace, Nat.add_mod, Nat.mul_mod] using h (2*s+1)
  · change evolve x ((2*s+1)+1) 0 = false
    simpa [trace, Nat.add_mod, Nat.mul_mod] using h (2*s+2)

/-- No two consecutive even outputs can both be one. -/
theorem alternating_u_no11 {x : Configuration}
    (h : ∀ n : Nat, trace x n = decide (n % 2 = 1)) (s : Nat)
    (hu : evolve x (2*s) 1 = true) : evolve x (2*(s+1)) 1 = false := by
  have h0 : evolve x (2*s) 0 = false := by
    simpa [trace, Nat.mul_mod] using h (2*s)
  have h1 : step (evolve x (2*s)) 0 = true := by
    change evolve x (2*s+1) 0 = true
    simpa [trace, Nat.add_mod, Nat.mul_mod] using h (2*s+1)
  have hr := column_no11 h0 h1 hu
  change evolve x (2*s+1+1) 1 = false at hr
  have he : 2*(s+1) = 2*s+1+1 := by omega
  rw [he]
  exact hr

#print axioms alternating_odd_left
#print axioms alternating_u_no11
#print axioms column_left_inversion
#print axioms odd_left_latch
#print axioms column_right_return
#print axioms column_no11

end Rule30
