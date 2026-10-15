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

theorem onset_reduction_evolve {x : Configuration} (hf : Finite x) (ha : AlternatingTail x) :
    ∃ T : Nat, ∃ y : Configuration, y = evolve x T ∧ Finite y ∧ Nonzero y ∧
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
  refine ⟨T, evolve x T, rfl, evolve_finite hf T, ?_, ?_⟩
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

/-- Compatibility form; the stronger theorem also records the time-shift witness. -/
theorem onset_reduction {x : Configuration} (hf : Finite x) (ha : AlternatingTail x) :
    ∃ y : Configuration, Finite y ∧ Nonzero y ∧
      ∀ n : Nat, trace y n = decide (n % 2 = 1) := by
  obtain ⟨_, y, _, hy⟩ := onset_reduction_evolve hf ha
  exact ⟨y, hy⟩

/-- Left permutivity is infrastructure, also valid for Rule 90. -/
theorem localRule_left_injective (a b c d : Bool)
    (h : localRule a c d = localRule b c d) : a = b := by
  cases a <;> cases b <;> cases c <;> cases d <;> simp_all [localRule]

/-- A prescribed centre trace and initial nonnegative half determine the
entire evolving nonnegative half. No finiteness hypothesis is needed. -/
theorem right_half_evolve_unique {x y : Configuration}
    (hr : ∀ i : Int, 0 ≤ i → x i = y i)
    (hc : ∀ t : Nat, trace x t = trace y t) :
    ∀ t : Nat, ∀ i : Int, 0 ≤ i → evolve x t i = evolve y t i := by
  intro t
  induction t with
  | zero => exact hr
  | succ t ih =>
    intro i hi
    by_cases hz : i = 0
    · subst i
      exact hc (t+1)
    · have hl := ih (i-1) (by omega)
      have hm := ih i hi
      have hh := ih (i+1) (by omega)
      simp only [evolve, step, hl, hm, hh]

/-- Triangular uniqueness (Condrey Lemma 1): fixing the whole initial
right half, including centre, and the whole centre trace fixes the row.
This uses only left permutivity, not Rule 30-specific exclusion. -/
theorem triangular_uniqueness {x y : Configuration}
    (hr : ∀ i : Int, 0 ≤ i → x i = y i)
    (hc : ∀ t : Nat, trace x t = trace y t) : x = y := by
  have strip : ∀ n : Nat, ∀ t : Nat, ∀ i : Int,
      -(n : Int) ≤ i → evolve x t i = evolve y t i := by
    intro n
    induction n with
    | zero => simpa using right_half_evolve_unique hr hc
    | succ n ih =>
      intro t i hi
      by_cases hinner : -(n : Int) ≤ i
      · exact ih t i hinner
      · have hmid := ih t (i+1) (by omega)
        have hright := ih t ((i+1)+1) (by omega)
        have hnext := ih (t+1) (i+1) (by omega)
        simp only [evolve, step] at hnext
        have hleft : (i+1)-1 = i := by omega
        rw [hleft, hmid, hright] at hnext
        exact localRule_left_injective _ _ _ _ hnext
  funext i
  have h := strip i.natAbs 0 i (by omega)
  exact h

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

/-- Completion with the given positive half and the constant-one checkerboard. -/
def oneCompletion (x : Configuration) : Configuration := fun i =>
  if 0 < i then x i else oneLeft i.natAbs

theorem oneCompletion_left (x : Configuration) : LeftOne (oneCompletion x) := by
  intro j
  have h : ¬ (0 : Int) < -(j : Int) := by omega
  simp [oneCompletion, h]

/-- Condrey Theorem 3, completeness: every constant-one fibre has the
checkerboard left half, without a finiteness restriction. -/
theorem constant_one_complete {x : Configuration}
    (hc : ∀ t : Nat, trace x t = true) : LeftOne x := by
  have hr : ∀ i : Int, 0 ≤ i → x i = oneCompletion x i := by
    intro i hi
    by_cases hp : 0 < i
    · simp [oneCompletion, hp]
    · have hz : i = 0 := by omega
      subst i
      simpa [trace, evolve, oneCompletion, oneLeft] using hc 0
  have heq : x = oneCompletion x := triangular_uniqueness hr (by
    intro t
    rw [hc t, leftOne_constant_trace (oneCompletion_left x) t])
  rw [heq]
  exact oneCompletion_left x

theorem finite_not_constant_one {x : Configuration} (hf : Finite x) :
    ¬ (∀ t : Nat, trace x t = true) := by
  intro hc
  exact leftOne_not_finite (constant_one_complete hc) hf

/-- Condrey's zero fibre; the zero row is treated separately below.
These formulas use Rule 30's OR, unlike triangular uniqueness alone. -/
def zeroLeft (m j : Nat) : Bool :=
  if j < m then false else if j = m then true else decide (j % 2 = 1)

theorem zeroLeft_update (m j : Nat) (hm : 0 < m) (hj : 0 < j) :
    localRule (zeroLeft m (j+1)) (zeroLeft m j) (zeroLeft m (j-1)) =
      zeroLeft (max 1 (m-1)) j := by
  have hjp : (j+1) % 2 = 1 ↔ j % 2 = 0 := by omega
  have hjm : (j-1) % 2 = 1 ↔ j % 2 = 0 := by omega
  have hjpar : j % 2 = 0 ∨ j % 2 = 1 := by omega
  by_cases hlow : j+1 < m
  · have h1 : j < m := by omega
    have h2 : j-1 < m := by omega
    have h3 : j < max 1 (m-1) := by omega
    simp [zeroLeft, localRule, hlow, h1, h2, h3]
  · by_cases heq : j+1 = m
    · have h1 : j < m := by omega
      have h2 : j-1 < m := by omega
      have h3 : j = max 1 (m-1) := by omega
      rw [show zeroLeft (max 1 (m-1)) j = true by simp [zeroLeft, h3]]
      simp [zeroLeft, localRule, heq, h1, h2]
    · by_cases heq2 : j = m
      · subst j
        rcases hjpar with hp | hp <;>
          simp [zeroLeft, localRule, show ¬ m+1<m by omega,
            show m+1≠m by omega, show m-1<m by omega,
            show ¬ m < max 1 (m-1) by omega,
            hjp, hp] <;> omega
      · by_cases heq3 : j = m+1
        · subst j
          rcases hjpar with hp | hp <;>
            simp [zeroLeft, localRule, show ¬ m+1+1<m by omega,
              show m+1+1≠m by omega, show ¬ m+1<m by omega,
              show m+1≠m by omega, show ¬ m+1 < max 1 (m-1) by omega,
              show m+1 ≠ max 1 (m-1) by omega, hjp, hp]
        · have h1 : ¬ j+1<m := by omega
          have h2 : j+1≠m := by omega
          have h3 : ¬ j<m := by omega
          have h4 : ¬ j-1<m := by omega
          have h5 : j-1≠m := by omega
          have h6 : ¬ j<max 1 (m-1) := by omega
          have h7 : j≠max 1 (m-1) := by omega
          rcases hjpar with hp | hp <;>
            simp [zeroLeft, localRule, h1,h2,h3,heq2,h4,h5,h6,h7,hjp,hjm,hp]

def ZeroClass (m : Nat) (x : Configuration) : Prop :=
  0 < m ∧ (∀ j : Nat, j < m → x (j : Int) = false) ∧
    x (m : Int) = true ∧ ∀ j : Nat, x (-(j : Int)) = zeroLeft m j

theorem zeroClass_centre {m : Nat} {x : Configuration} (h : ZeroClass m x) :
    step x 0 = false := by
  obtain ⟨hm, hz, hone, hl⟩ := h
  have hc : x 0 = false := hz 0 hm
  have he : x (-1) = x 1 := by
    have hh := hl 1
    by_cases hm1 : m = 1
    · subst m
      simpa [zeroLeft] using hh.trans hone.symm
    · have hr : x 1 = false := hz 1 (by omega)
      rw [hr]
      simpa [zeroLeft, show 1<m by omega] using hh
  simp only [step, localRule]
  simp only [show (0:Int)-1 = -1 from rfl, show (0:Int)+1 = 1 from rfl]
  rw [he, hc]
  cases x 1 <;> rfl

theorem zeroClass_step {m : Nat} {x : Configuration} (h : ZeroClass m x) :
    ZeroClass (max 1 (m-1)) (step x) := by
  have hc := zeroClass_centre h
  obtain ⟨hm, hz, hone, hl⟩ := h
  refine ⟨by omega, ?_, ?_, ?_⟩
  · intro j hj
    by_cases hj0 : j = 0
    · subst j; exact hc
    · have ha : (j : Int)-1 = ((j-1 : Nat) : Int) := by omega
      have hb : (j : Int)+1 = ((j+1 : Nat) : Int) := by omega
      have h1 := hz (j-1) (by omega)
      have h2 := hz j (by omega)
      have h3 := hz (j+1) (by omega)
      rw [step, ha, hb, h1, h2, h3]
      rfl
  · by_cases hm1 : m = 1
    · subst m
      have hc0 : x 0 = false := hz 0 (by omega)
      have h1 : x 1 = true := hone
      simp [step, localRule, hc0, h1]
    · have hmax : max 1 (m-1) = m-1 := by omega
      rw [hmax]
      have ha : ((m-1 : Nat) : Int)-1 = ((m-2 : Nat) : Int) := by omega
      have hb : ((m-1 : Nat) : Int)+1 = (m : Int) := by omega
      have h1 := hz (m-2) (by omega)
      have h2 := hz (m-1) (by omega)
      rw [step, ha, hb, h1, h2, hone]
      rfl
  · intro j
    by_cases hj0 : j = 0
    · subst j
      simpa [zeroLeft, show 0 < max 1 (m-1) by omega] using hc
    · have ha : -(j : Int)-1 = -((j+1 : Nat) : Int) := by omega
      have hb : -(j : Int)+1 = -((j-1 : Nat) : Int) := by omega
      simp only [step, ha, hb, hl]
      exact zeroLeft_update m j hm (by omega)

theorem zeroClass_trace {m : Nat} {x : Configuration} (h : ZeroClass m x) :
    ∀ t : Nat, trace x t = false := by
  have hinv : ∀ t : Nat, ∃ m : Nat, ZeroClass m (evolve x t) := by
    intro t
    induction t with
    | zero => exact ⟨m, h⟩
    | succ t ih =>
      obtain ⟨m, hm⟩ := ih
      exact ⟨max 1 (m-1), zeroClass_step hm⟩
  intro t
  obtain ⟨m, hm, hz, _, _⟩ := hinv t
  exact hz 0 hm

/-- Preserve the given nonnegative half; complete its left half for a
constant-zero trace when its first positive one is at m. -/
def zeroCompletion (x : Configuration) (m : Nat) : Configuration := fun i =>
  if i < 0 then zeroLeft m i.natAbs else x i

theorem zeroCompletion_class {x : Configuration} {m : Nat} (hm : 0 < m)
    (hz : ∀ j : Nat, j < m → x (j : Int) = false) (hone : x (m : Int) = true) :
    ZeroClass m (zeroCompletion x m) := by
  refine ⟨hm, ?_, ?_, ?_⟩
  · intro j hj
    simpa [zeroCompletion, show ¬ (j : Int)<0 by omega] using hz j hj
  · simpa [zeroCompletion, show ¬ (m : Int)<0 by omega] using hone
  · intro j
    by_cases hj : j = 0
    · subst j
      simpa [zeroCompletion, zeroLeft, hm] using hz 0 hm
    · simp [zeroCompletion, show -(j : Int)<0 by omega]

/-- Condrey Theorem 2 construction, with arbitrary exterior positive tail. -/
theorem zeroCompletion_trace {x : Configuration} {m : Nat} (hm : 0 < m)
    (hz : ∀ j : Nat, j < m → x (j : Int) = false) (hone : x (m : Int) = true) :
    ∀ t : Nat, trace (zeroCompletion x m) t = false :=
  zeroClass_trace (zeroCompletion_class hm hz hone)

/-- Condrey Theorem 2 completeness for a right half with first positive
one at m: triangular uniqueness forces precisely the stated left pattern. -/
theorem constant_zero_complete {x : Configuration} {m : Nat} (hm : 0 < m)
    (hz : ∀ j : Nat, j < m → x (j : Int) = false) (hone : x (m : Int) = true)
    (hc : ∀ t : Nat, trace x t = false) :
    ∀ j : Nat, x (-(j : Int)) = zeroLeft m j := by
  have hr : ∀ i : Int, 0 ≤ i → x i = zeroCompletion x m i := by
    intro i hi
    simp [zeroCompletion, show ¬ i<0 by omega]
  have heq : x = zeroCompletion x m := triangular_uniqueness hr (by
    intro t
    rw [hc t, zeroCompletion_trace hm hz hone t])
  rw [heq]
  exact (zeroCompletion_class hm hz hone).2.2.2

theorem zeroClass_not_finite {x : Configuration} {m : Nat}
    (hl : ∀ j : Nat, x (-(j : Int)) = zeroLeft m j) : ¬ Finite x := by
  intro hf
  obtain ⟨R, hR⟩ := hf
  let j := 2*(R+m+1)+1
  have hjm : m < j := by omega
  have hjR : (R : Int) < (j : Int) := by omega
  have hodd : j % 2 = 1 := by omega
  have ht := hl j
  have hz := hR (-(j : Int)) (by left; omega)
  simp [zeroLeft, show ¬ j<m by omega, show j≠m by omega, hodd, hz] at ht

theorem zero_evolve (t : Nat) :
    evolve (fun _ : Int => false) t = (fun _ : Int => false) := by
  induction t with
  | zero => rfl
  | succ t ih =>
    funext i
    simp [evolve, ih, step, localRule]

/-- The empty-positive-half case of Condrey Theorem 2 includes the zero row. -/
theorem constant_zero_empty_right {x : Configuration}
    (hr : ∀ i : Int, 0 ≤ i → x i = false)
    (hc : ∀ t : Nat, trace x t = false) : x = (fun _ => false) := by
  apply triangular_uniqueness hr
  intro t
  simpa [trace, zero_evolve] using hc t

theorem first_right_one {x : Configuration} (h : ∃ j : Nat, x (j : Int) = true) :
    ∃ m : Nat, x (m : Int) = true ∧ ∀ j : Nat, j < m → x (j : Int) = false := by
  obtain ⟨j, hj⟩ := h
  induction j using Nat.strongRecOn with
  | ind j ih =>
    by_cases hs : ∃ k : Nat, k < j ∧ x (k : Int) = true
    · obtain ⟨k, hk, hone⟩ := hs
      exact ih k hk hone
    · refine ⟨j, hj, ?_⟩
      intro k hk
      cases he : x (k : Int) with
      | false => rfl
      | true => exact False.elim (hs ⟨k, hk, he⟩)

theorem finite_nonzero_not_constant_zero {x : Configuration}
    (hf : Finite x) (hn : Nonzero x) : ¬ (∀ t : Nat, trace x t = false) := by
  intro hc
  have hc0 : x 0 = false := hc 0
  by_cases hp : ∃ j : Nat, x (j : Int) = true
  · obtain ⟨m, hone, hz⟩ := first_right_one hp
    have hm : 0 < m := by
      apply Classical.byContradiction
      intro h
      have : m = 0 := by omega
      subst m
      simp [hc0] at hone
    exact zeroClass_not_finite (constant_zero_complete hm hz hone hc) hf
  · have hr : ∀ i : Int, 0 ≤ i → x i = false := by
      intro i hi
      cases he : x i with
      | false => rfl
      | true =>
        apply False.elim
        apply hp
        exact ⟨i.toNat, by simpa [Int.toNat_of_nonneg hi] using he⟩
    have heq := constant_zero_empty_right hr hc
    obtain ⟨i, hi⟩ := hn
    simp [heq] at hi

/-- Finite nonzero rows cannot disappear in one step. -/
theorem step_nonzero {x : Configuration} (hf : Finite x) (hn : Nonzero x) :
    Nonzero (step x) := by
  apply Classical.byContradiction
  intro hzero
  have hs : ∀ i : Int, step x i = false := by
    intro i
    cases hi : step x i with
    | false => rfl
    | true => exact False.elim (hzero ⟨i, hi⟩)
  obtain ⟨R, hR⟩ := hf
  have strip : ∀ n : Nat, ∀ i : Int, (R : Int)+1-(n : Int) ≤ i → x i = false := by
    intro n
    induction n with
    | zero =>
      intro i hi
      exact hR i (by right; omega)
    | succ n ih =>
      intro i hi
      by_cases hinner : (R : Int)+1-(n : Int) ≤ i
      · exact ih i hinner
      · have h1 := ih (i+1) (by omega)
        have h2 := ih ((i+1)+1) (by omega)
        have hstep := hs (i+1)
        have hl : (i+1)-1 = i := by omega
        simpa [step, hl, localRule, h1, h2] using hstep
  obtain ⟨i, hi⟩ := hn
  have hx := strip (R+1+i.natAbs) i (by omega)
  simp [hx] at hi

theorem evolve_nonzero {x : Configuration} (hf : Finite x) (hn : Nonzero x)
    (t : Nat) : Nonzero (evolve x t) := by
  induction t with
  | zero => exact hn
  | succ t ih => exact step_nonzero (evolve_finite hf t) ih

/-- Condrey Corollary 5: neither eventual constant phase is possible for a
finite nonzero row; the all-zero row is intentionally excluded. -/
theorem finite_nonzero_not_eventually_constant {x : Configuration}
    (hf : Finite x) (hn : Nonzero x) :
    ¬ ∃ T : Nat, ∃ phase : Bool, ∀ n : Nat, trace x (T+n) = phase := by
  intro ⟨T, phase, hc⟩
  have hshift : ∀ n : Nat, trace (evolve x T) n = phase := by
    intro n
    rw [trace_shift]
    exact hc n
  cases phase with
  | false =>
    exact finite_nonzero_not_constant_zero (evolve_finite hf T) (evolve_nonzero hf hn T) hshift
  | true => exact finite_not_constant_one (evolve_finite hf T) hshift

/-- A finite-ring control, not a bridge to any finite physical right half. -/
def ringStep7 (x : Fin 7 → Bool) (i : Fin 7) : Bool :=
  localRule (x (i+1)) (x i) (x (i-1))
def ring0 (i : Fin 7) : Bool := decide (i.val = 0)
def ring1 (i : Fin 7) : Bool := decide (i.val = 0 ∨ i.val = 1 ∨ i.val = 6)
def ring2 (i : Fin 7) : Bool := decide (i.val = 1 ∨ i.val = 2 ∨ i.val = 5)
def ring3 (i : Fin 7) : Bool := decide (i.val = 0 ∨ 2 ≤ i.val)

/-- Ring indices increase towards physical left: i maps to position -i.
The modulo lift is spatially periodic and has no finite-support assertion. -/
def ringIndex7 (position : Int) : Fin 7 :=
  ⟨((-position) % 7).toNat, by omega⟩

def ringLift7 (x : Fin 7 → Bool) : Configuration := fun i => x (ringIndex7 i)

theorem ringIndex7_left (i : Int) : ringIndex7 (i-1) = ringIndex7 i + 1 := by
  apply Fin.ext
  simp [ringIndex7, Fin.add_def]
  omega

theorem ringIndex7_right (i : Int) : ringIndex7 (i+1) = ringIndex7 i - 1 := by
  apply Fin.ext
  simp [ringIndex7, Fin.sub_def]
  omega

theorem lift_step (x : Fin 7 → Bool) : step (ringLift7 x) = ringLift7 (ringStep7 x) := by
  funext i
  simp only [step, ringLift7, ringStep7, ringIndex7_left, ringIndex7_right]

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

#print axioms ringIndex7_left
#print axioms ringIndex7_right
#print axioms lift_step
#print axioms ring7_boundary
#print axioms ring7_step0
#print axioms ring7_step1
#print axioms ring7_step2
#print axioms ring7_step3

#print axioms oneCompletion_left
#print axioms constant_one_complete
#print axioms finite_not_constant_one
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
#print axioms onset_reduction_evolve
#print axioms localRule_left_injective
#print axioms right_half_evolve_unique
#print axioms triangular_uniqueness
#print axioms onset_reduction
#print axioms oneLeft_adjacent
#print axioms oneLeft_invariant
end Rule30
