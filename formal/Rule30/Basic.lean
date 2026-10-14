import Lean

namespace Rule30
abbrev Configuration := Int → Bool

def localRule (left centre right : Bool) : Bool := left ^^ (centre || right)
def step (x : Configuration) : Configuration := fun i => localRule (x (i-1)) (x i) (x (i+1))
def evolve (x : Configuration) : Nat → Configuration
  | 0 => x
  | t+1 => step (evolve x t)
def trace (x : Configuration) (t : Nat) : Bool := evolve x t 0
def Finite (x : Configuration) : Prop :=
  ∃ R : Nat, ∀ i : Int, i < -(R : Int) ∨ (R : Int) < i → x i = false
def Nonzero (x : Configuration) : Prop := ∃ i, x i = true

theorem zero_latch (s u : Bool) : localRule false s u = (s || u) := by
  cases s <;> cases u <;> rfl

/-- Boolean form of `(1-s)(1-u)(1-v)`, using Rule 30's OR. -/
theorem alternating_return (s u v : Bool) :
    localRule true (localRule false s u) (localRule s u v) = (!s && !u && !v) := by
  cases s <;> cases u <;> cases v <;> rfl

theorem step_finite {x : Configuration} (h : Finite x) : Finite (step x) := by
  obtain ⟨R, hR⟩ := h
  refine ⟨R+1, ?_⟩
  intro i hi
  have hl : x (i-1) = false := hR _ (by rcases hi with hi | hi <;> omega)
  have hc : x i = false := hR _ (by rcases hi with hi | hi <;> omega)
  have hr : x (i+1) = false := hR _ (by rcases hi with hi | hi <;> omega)
  simp [step, localRule, hl, hc, hr]

theorem evolve_finite {x : Configuration} (h : Finite x) (t : Nat) : Finite (evolve x t) := by
  induction t with
  | zero => exact h
  | succ t ih => exact step_finite ih

theorem evolve_add (x : Configuration) (a b : Nat) : evolve x (a+b) = evolve (evolve x a) b := by
  induction b with
  | zero => rfl
  | succ b ih => simp [Nat.add_succ, evolve, ih]

theorem trace_shift (x : Configuration) (T n : Nat) : trace (evolve x T) n = trace x (T+n) := by
  simp [trace, evolve_add]

/-- A nonconstant alternating tail, allowing either initial phase. -/
def AlternatingTail (x : Configuration) : Prop :=
  ∃ T : Nat, ∃ phase : Bool, ∀ n : Nat, trace x (T+n) = (phase ^^ decide (n % 2 = 1))

theorem onset_reduction {x : Configuration} (hf : Finite x) (ha : AlternatingTail x) :
    ∃ y : Configuration, Finite y ∧ Nonzero y ∧
      ∀ n : Nat, trace y n = decide (n % 2 = 1) := by
  obtain ⟨T₀, phase, hp⟩ := ha
  have normalized : ∃ T : Nat, ∀ n : Nat, trace x (T+n) = decide (n % 2 = 1) := by
    cases phase with
    | false => exact ⟨T₀, by intro n; simpa using hp n⟩
    | true =>
      refine ⟨T₀+1, ?_⟩
      intro n
      have hn : T₀+1+n = T₀+(n+1) := by omega
      rw [hn, hp]
      have parity : n % 2 = 0 ∨ n % 2 = 1 := by omega
      rcases parity with h | h
      · have hnext : (n+1) % 2 = 1 := by omega
        simp [h, hnext]
      · have hnext : (n+1) % 2 = 0 := by omega
        simp [h, hnext]
  obtain ⟨T, hT⟩ := normalized
  refine ⟨evolve x T, evolve_finite hf T, ?_, ?_⟩
  · apply Classical.byContradiction
    intro hn
    have hz : ∀ i, evolve x T i = false := by
      intro i
      cases hi : evolve x T i with
      | false => rfl
      | true => exact False.elim (hn ⟨i, hi⟩)
    have hone := hT 1
    rw [← trace_shift] at hone
    simp [trace, evolve, step, localRule, hz] at hone
  · intro n
    rw [trace_shift]
    exact hT n

/-- Checkerboard used by Condrey's one-fiber construction. This is only
its local invariant; completeness/exclusion is not asserted. -/
def oneLeft (j : Nat) : Bool := decide (j % 2 = 0)

theorem oneLeft_adjacent (j : Nat) : oneLeft j || oneLeft (j+1) = true := by
  simp only [oneLeft]
  have : j % 2 = 0 ∨ (j+1) % 2 = 0 := by omega
  rcases this with h | h <;> simp [h]

theorem oneLeft_invariant (j : Nat) :
    localRule (oneLeft (j+2)) (oneLeft (j+1)) (oneLeft j) = oneLeft (j+1) := by
  have he : (j+2) % 2 = j % 2 := by omega
  have ho : (j+1) % 2 = 0 ↔ j % 2 ≠ 0 := by omega
  unfold localRule oneLeft
  simp only [he]
  by_cases h : j % 2 = 0 <;> simp [h, ho]

/-- Condrey Theorem 3's existence half. Uniqueness is a separate obligation. -/
def LeftOne (x : Configuration) : Prop := ∀ j : Nat, x (-(j : Int)) = oneLeft j

theorem leftOne_step {x : Configuration} (h : LeftOne x) : LeftOne (step x) := by
  intro j
  cases j with
  | zero =>
    have h0 := h 0
    have h1 := h 1
    simp [oneLeft] at h0 h1
    simp [step, localRule, h0, h1, oneLeft]
  | succ j =>
    have hl : -(↑(j+1) : Int)-1 = -(↑(j+2) : Int) := by omega
    have hr : -(↑(j+1) : Int)+1 = -(↑j : Int) := by omega
    simp only [step, hl, hr]
    rw [h (j+2), h (j+1), h j]
    exact oneLeft_invariant j

theorem leftOne_evolve {x : Configuration} (h : LeftOne x) (t : Nat) : LeftOne (evolve x t) := by
  induction t with
  | zero => exact h
  | succ t ih => exact leftOne_step ih

theorem leftOne_constant_trace {x : Configuration} (h : LeftOne x) (t : Nat) : trace x t = true := by
  have h0 := leftOne_evolve h t 0
  simpa [trace, oneLeft] using h0

theorem leftOne_not_finite {x : Configuration} (h : LeftOne x) : ¬ Finite x := by
  intro hf
  obtain ⟨R, hR⟩ := hf
  have ht := h (2*(R+1))
  have hz := hR (-(↑(2*(R+1)) : Int)) (by left; omega)
  have he : (2*(R+1)) % 2 = 0 := by omega
  simp only [oneLeft, he, decide_true] at ht
  have : x (-((2 : Int)*((R : Int)+1))) = false := by simpa using hz
  simp [this] at ht

/-- A finite-ring control, not a bridge to any finite physical right half. -/
def ringStep7 (x : Fin 7 → Bool) (i : Fin 7) : Bool :=
  localRule (x (i+1)) (x i) (x (i-1))
def ring0 (i : Fin 7) : Bool := decide (i.val = 0)
def ring1 (i : Fin 7) : Bool := decide (i.val = 0 ∨ i.val = 1 ∨ i.val = 6)
def ring2 (i : Fin 7) : Bool := decide (i.val = 1 ∨ i.val = 2 ∨ i.val = 5)
def ring3 (i : Fin 7) : Bool := decide (i.val = 0 ∨ 2 ≤ i.val)

theorem ring7_step0 : ∀ i, ringStep7 ring0 i = ring1 i := by decide
theorem ring7_step1 : ∀ i, ringStep7 ring1 i = ring2 i := by decide
theorem ring7_step2 : ∀ i, ringStep7 ring2 i = ring3 i := by decide
theorem ring7_step3 : ∀ i, ringStep7 ring3 i = ring0 i := by decide

theorem ring7_boundary :
    ring0 0 = true ∧ ring0 1 = false ∧ ring2 0 = false ∧ ring2 1 = true ∧
    ring1 0 = true ∧ ring3 0 = true := by decide

/-- Zero-pair indicator used in the recovered depth-seven temporal formula. -/
def zeroPair (x y : Bool) : Int := if x || y then 0 else 1

/-- Exact coboundary of the recovered depth-seven OR-specific temporal
formulas, under adjacent-11-free hypotheses. This is an algebraic identity;
its interpretation as a column formula and any spatial transport are
separate obligations. -/
theorem temporal_phase_slip (b c d : Bool)
    (hbc : (b && c) = false) (hcd : (c && d) = false) :
    ((!(c || d)).toNat : Int) + ((!d || (b && d)).toNat : Int) - 1 =
      zeroPair c d - zeroPair b c + zeroPair b c * zeroPair c d := by
  cases b <;> cases c <;> cases d <;> simp_all [zeroPair, Bool.toNat]

#print axioms temporal_phase_slip

#print axioms ring7_boundary
#print axioms ring7_step0
#print axioms ring7_step1
#print axioms ring7_step2
#print axioms ring7_step3

#print axioms leftOne_step
#print axioms leftOne_evolve
#print axioms leftOne_constant_trace
#print axioms leftOne_not_finite

#print axioms zero_latch
#print axioms alternating_return
#print axioms step_finite
#print axioms evolve_finite
#print axioms evolve_add
#print axioms trace_shift
#print axioms onset_reduction
#print axioms oneLeft_adjacent
#print axioms oneLeft_invariant
end Rule30
