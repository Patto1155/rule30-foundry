/-
Rule 30: no column of a nonzero finite-support orbit is eventually constant.

Self-contained Lean 4 (core only, no Mathlib, no axioms beyond Lean's own).
Check with:  lean formal/Rule30/PeriodOne.lean

The argument is the elementary one reconstructed in
`docs/theory/period2_attempt.md` section 2, simplified so that the
eventually-zero case reduces to the eventually-one case:

* eventually ONE at column c  ==> columns c-1, c-2, ... are forced to the
  constant checkerboard 0,1,0,1,... at every time, so the support is infinite;
* eventually ZERO at column c ==> the right neighbour obeys the monotone OR
  latch s' = s OR u and the left neighbour equals it. Either the latch never
  fires, and then the whole row is zero, or it fires, and then the left
  neighbour column is eventually one, which is the first case.
-/

namespace Rule30

abbrev Configuration := Int → Bool

/-- Rule 30: new = left XOR (centre OR right). -/
def localRule (l c r : Bool) : Bool := l ^^ (c || r)

def step (x : Configuration) : Configuration :=
  fun i => localRule (x (i - 1)) (x i) (x (i + 1))

def evolve (x : Configuration) : Nat → Configuration
  | 0 => x
  | t + 1 => step (evolve x t)

/-- Finite support: zero outside a bounded interval. -/
def Finite (x : Configuration) : Prop :=
  ∃ R : Int, ∀ i : Int, (i < -R ∨ R < i) → x i = false

/-- The single-black-cell initial condition of the Wolfram prize problems. -/
def seed : Configuration := fun i => decide (i = 0)

/-! ### Truth-table facts -/

theorem rule_one_centre : ∀ a q : Bool, localRule a true q = true → a = false := by decide
theorem rule_checker : ∀ a b : Bool, localRule a (!b) b = (!b) → a = b := by decide
theorem rule_zero_left : ∀ a : Bool, localRule a false false = false → a = false := by decide
theorem rule_zero_right : ∀ r : Bool, localRule false false r = false → r = false := by decide
theorem rule_zero_mid : ∀ p q : Bool, localRule false p q = false → p = false := by decide
theorem rule_zero_centre : ∀ a q : Bool, localRule a false q = false → a = q := by decide
theorem rule_latch : ∀ q : Bool, localRule false true q = true := by decide

/-! ### Basic dynamics -/

theorem evolve_succ (x : Configuration) (t : Nat) (i : Int) :
    evolve x (t + 1) i
      = localRule (evolve x t (i - 1)) (evolve x t i) (evolve x t (i + 1)) := rfl

theorem step_finite {x : Configuration} (h : Finite x) : Finite (step x) := by
  obtain ⟨R, hR⟩ := h
  refine ⟨R + 1, fun i hi => ?_⟩
  have h1 : x (i - 1) = false := hR _ (by omega)
  have h2 : x i = false := hR _ (by omega)
  have h3 : x (i + 1) = false := hR _ (by omega)
  simp [step, localRule, h1, h2, h3]

theorem evolve_finite {x : Configuration} (h : Finite x) (t : Nat) :
    Finite (evolve x t) := by
  induction t with
  | zero => exact h
  | succ t ih => exact step_finite ih

theorem evolve_add (x : Configuration) (a b : Nat) :
    evolve x (a + b) = evolve (evolve x a) b := by
  induction b with
  | zero => rfl
  | succ b ih => show step (evolve x (a + b)) = step (evolve (evolve x a) b); rw [ih]

/-! ### Eventually one is impossible -/

/-- Checkerboard parity, defined by recursion so the successor law is `rfl`. -/
def par : Nat → Bool
  | 0 => true
  | j + 1 => !par j

theorem par_double (k : Nat) : par (2 * k) = true := by
  induction k with
  | zero => rfl
  | succ k ih =>
    show (!(!par (2 * k))) = true
    rw [ih]; rfl

/-- If column `c` is one at every time, column `c - j` is constantly `par j`. -/
theorem checkerboard {z : Configuration} {c : Int} (h : ∀ t, evolve z t c = true) :
    ∀ j : Nat, (∀ t, evolve z t (c - (j : Int)) = par j) ∧
               (∀ t, evolve z t (c - ((j + 1 : Nat) : Int)) = par (j + 1)) := by
  intro j
  induction j with
  | zero =>
    refine ⟨fun t => ?_, fun t => ?_⟩
    · have e : c - ((0 : Nat) : Int) = c := by omega
      rw [e]; exact h t
    · have e : c - ((0 + 1 : Nat) : Int) = c - 1 := by omega
      rw [e]
      have h1 := h (t + 1)
      rw [evolve_succ, h t] at h1
      exact rule_one_centre _ _ h1
  | succ j ih =>
    refine ⟨ih.2, fun t => ?_⟩
    have eq := evolve_succ z t (c - ((j + 1 : Nat) : Int))
    have e1 : c - ((j + 1 : Nat) : Int) - 1 = c - ((j + 1 + 1 : Nat) : Int) := by omega
    have e2 : c - ((j + 1 : Nat) : Int) + 1 = c - (j : Int) := by omega
    rw [e1, e2, ih.2 (t + 1), ih.2 t, ih.1 t] at eq
    -- eq : par (j+1) = localRule (evolve z t (c - (j+2))) (par (j+1)) (par j)
    have := rule_checker (evolve z t (c - ((j + 1 + 1 : Nat) : Int))) (par j) eq.symm
    rw [this]
    show par j = (!(!par j))
    cases par j <;> rfl

theorem no_always_one {z : Configuration} (hf : Finite z) (c : Int)
    (h : ∀ t, evolve z t c = true) : False := by
  obtain ⟨R, hR⟩ := hf
  have hk := (checkerboard h (2 * (c + R + 1).toNat)).1 0
  rw [par_double] at hk
  have hz : z (c - ((2 * (c + R + 1).toNat : Nat) : Int)) = false :=
    hR _ (Or.inl (by omega))
  have : evolve z 0 (c - ((2 * (c + R + 1).toNat : Nat) : Int)) = false := hz
  rw [this] at hk
  exact Bool.noConfusion hk

/-! ### Eventually zero is impossible for a nonzero row -/

theorem zero_left {z : Configuration} {c : Int}
    (h0 : ∀ t, evolve z t c = false) (h1 : ∀ t, evolve z t (c + 1) = false) :
    ∀ j : Nat, (∀ t, evolve z t (c - (j : Int)) = false) ∧
               (∀ t, evolve z t (c - (j : Int) + 1) = false) := by
  intro j
  induction j with
  | zero =>
    have e : c - ((0 : Nat) : Int) = c := by omega
    rw [e]; exact ⟨h0, h1⟩
  | succ j ih =>
    have e2 : c - ((j + 1 : Nat) : Int) + 1 = c - (j : Int) := by omega
    refine ⟨fun t => ?_, fun t => by rw [e2]; exact ih.1 t⟩
    have eq := evolve_succ z t (c - (j : Int))
    have e1 : c - (j : Int) - 1 = c - ((j + 1 : Nat) : Int) := by omega
    rw [e1, ih.1 (t + 1), ih.1 t, ih.2 t] at eq
    exact rule_zero_left _ eq.symm

theorem zero_right {z : Configuration} {c : Int}
    (h0 : ∀ t, evolve z t c = false) (h1 : ∀ t, evolve z t (c + 1) = false) :
    ∀ k : Nat, (∀ t, evolve z t (c + (k : Int)) = false) ∧
               (∀ t, evolve z t (c + (k : Int) + 1) = false) := by
  intro k
  induction k with
  | zero =>
    have e : c + ((0 : Nat) : Int) = c := by omega
    rw [e]; exact ⟨h0, h1⟩
  | succ k ih =>
    have e2 : c + ((k + 1 : Nat) : Int) = c + (k : Int) + 1 := by omega
    refine ⟨fun t => by rw [e2]; exact ih.2 t, fun t => ?_⟩
    have eq := evolve_succ z t (c + (k : Int) + 1)
    have e1 : c + (k : Int) + 1 - 1 = c + (k : Int) := by omega
    have e3 : c + (k : Int) + 1 + 1 = c + ((k + 1 : Nat) : Int) + 1 := by omega
    rw [e1, e3, ih.2 (t + 1), ih.1 t, ih.2 t] at eq
    exact rule_zero_right _ eq.symm

theorem latch {z : Configuration} {c : Int} (h0 : ∀ t, evolve z t c = false)
    (τ : Nat) (hτ : evolve z τ (c + 1) = true) :
    ∀ n : Nat, evolve z (τ + n) (c + 1) = true := by
  intro n
  induction n with
  | zero => exact hτ
  | succ n ih =>
    show evolve z (τ + n + 1) (c + 1) = true
    rw [evolve_succ]
    have e : c + 1 - 1 = c := by omega
    rw [e, h0 (τ + n), ih]
    exact rule_latch _

theorem left_eq_right {z : Configuration} {c : Int} (h0 : ∀ t, evolve z t c = false)
    (t : Nat) : evolve z t (c - 1) = evolve z t (c + 1) := by
  have eq := evolve_succ z t c
  rw [h0 (t + 1), h0 t] at eq
  exact rule_zero_centre _ _ eq.symm

/-- A finite row with a column that is zero at every time is the zero row. -/
theorem always_zero_forces_zero {z : Configuration} (hf : Finite z) (c : Int)
    (h0 : ∀ t, evolve z t c = false) : ∀ i, z i = false := by
  by_cases hs : ∃ τ, evolve z τ (c + 1) = true
  · obtain ⟨τ, hτ⟩ := hs
    exfalso
    refine no_always_one (evolve_finite hf τ) (c - 1) (fun n => ?_)
    rw [← evolve_add, left_eq_right h0]
    exact latch h0 τ hτ n
  · have h1 : ∀ t, evolve z t (c + 1) = false := by
      intro t
      cases hv : evolve z t (c + 1) with
      | false => rfl
      | true => exact absurd ⟨t, hv⟩ hs
    intro i
    by_cases hi : i ≤ c
    · have := (zero_left h0 h1 (c - i).toNat).1 0
      have e : c - (((c - i).toNat : Nat) : Int) = i := by omega
      rw [e] at this; exact this
    · have := (zero_right h0 h1 (i - c).toNat).1 0
      have e : c + (((i - c).toNat : Nat) : Int) = i := by omega
      rw [e] at this; exact this

/-- A finite row whose image is zero is zero. -/
theorem step_zero {x : Configuration} (hf : Finite x)
    (h : ∀ i, step x i = false) : ∀ i, x i = false := by
  obtain ⟨R, hR⟩ := hf
  have key : ∀ n : Nat, x (-R - 1 + (n : Int)) = false := by
    intro n
    induction n with
    | zero => exact hR _ (Or.inl (by omega))
    | succ n ih =>
      have hs := h (-R - 1 + ((n + 1 : Nat) : Int))
      have e1 : -R - 1 + ((n + 1 : Nat) : Int) - 1 = -R - 1 + (n : Int) := by omega
      unfold step at hs
      rw [e1, ih] at hs
      exact rule_zero_mid _ _ hs
  intro i
  by_cases hi : i < -R
  · exact hR _ (Or.inl hi)
  · have := key (i + R + 1).toNat
    have e : -R - 1 + (((i + R + 1).toNat : Nat) : Int) = i := by omega
    rw [e] at this; exact this

theorem evolve_zero {x : Configuration} (hf : Finite x) (T : Nat)
    (h : ∀ i, evolve x T i = false) : ∀ i, x i = false := by
  induction T with
  | zero => exact h
  | succ T ih => exact ih (step_zero (evolve_finite hf T) h)

/-! ### Main theorems -/

/-- **Period one is excluded.** No column of the Rule 30 orbit of a nonzero
finite-support configuration is eventually constant. -/
theorem no_eventually_constant_column {x : Configuration} (hf : Finite x)
    (hn : ∃ i, x i = true) (c : Int) (b : Bool) :
    ¬ ∃ T : Nat, ∀ t : Nat, T ≤ t → evolve x t c = b := by
  rintro ⟨T, hT⟩
  have hz : ∀ n, evolve (evolve x T) n c = b := by
    intro n
    rw [← evolve_add]
    exact hT _ (Nat.le_add_right T n)
  cases b with
  | true => exact no_always_one (evolve_finite hf T) c hz
  | false =>
    have hzero := evolve_zero hf T (always_zero_forces_zero (evolve_finite hf T) c hz)
    obtain ⟨i, hi⟩ := hn
    rw [hzero i] at hi
    exact Bool.noConfusion hi

theorem seed_finite : Finite seed :=
  ⟨0, fun i hi => by
    have : i ≠ 0 := by omega
    simp [seed, this]⟩

/-- **The prize object.** In the centre column of Rule 30 started from a single
black cell, both colours occur at arbitrarily late times. -/
theorem centre_column_not_eventually_constant (T : Nat) (b : Bool) :
    ∃ t : Nat, T ≤ t ∧ evolve seed t 0 = b := by
  apply Classical.byContradiction
  intro hno
  refine no_eventually_constant_column seed_finite ⟨0, rfl⟩ 0 (!b) ⟨T, fun t ht => ?_⟩
  cases hv : evolve seed t 0 <;> cases b <;> first
    | rfl
    | exact absurd ⟨t, ht, hv⟩ hno

/-! ### Quantitative version for the single seed

The same argument run inside a finite horizon gives an explicit window: from
any time `T`, the centre column shows both colours by time `4T + 2`.
-/

theorem checkerboard_fin {z : Configuration} {c : Int} {g : Nat}
    (h : ∀ t, t ≤ g → evolve z t c = true) :
    ∀ j : Nat, (∀ t, t + j ≤ g → evolve z t (c - (j : Int)) = par j) ∧
               (∀ t, t + (j + 1) ≤ g →
                  evolve z t (c - ((j + 1 : Nat) : Int)) = par (j + 1)) := by
  intro j
  induction j with
  | zero =>
    refine ⟨fun t ht => ?_, fun t ht => ?_⟩
    · have e : c - ((0 : Nat) : Int) = c := by omega
      rw [e]; exact h t (by omega)
    · have e : c - ((0 + 1 : Nat) : Int) = c - 1 := by omega
      rw [e]
      have h1 := h (t + 1) (by omega)
      rw [evolve_succ, h t (by omega)] at h1
      exact rule_one_centre _ _ h1
  | succ j ih =>
    refine ⟨ih.2, fun t ht => ?_⟩
    have eq := evolve_succ z t (c - ((j + 1 : Nat) : Int))
    have e1 : c - ((j + 1 : Nat) : Int) - 1 = c - ((j + 1 + 1 : Nat) : Int) := by omega
    have e2 : c - ((j + 1 : Nat) : Int) + 1 = c - (j : Int) := by omega
    rw [e1, e2, ih.2 (t + 1) (by omega), ih.2 t (by omega), ih.1 t (by omega)] at eq
    have := rule_checker (evolve z t (c - ((j + 1 + 1 : Nat) : Int))) (par j) eq.symm
    rw [this]
    show par j = (!(!par j))
    cases par j <;> rfl

theorem zero_left_fin {z : Configuration} {c : Int} {N : Nat}
    (h0 : ∀ t, t ≤ N → evolve z t c = false)
    (h1 : ∀ t, t ≤ N → evolve z t (c + 1) = false) :
    ∀ j : Nat, (∀ t, t + j ≤ N → evolve z t (c - (j : Int)) = false) ∧
               (∀ t, t + j ≤ N → evolve z t (c - (j : Int) + 1) = false) := by
  intro j
  induction j with
  | zero =>
    have e : c - ((0 : Nat) : Int) = c := by omega
    rw [e]
    exact ⟨fun t ht => h0 t (by omega), fun t ht => h1 t (by omega)⟩
  | succ j ih =>
    have e2 : c - ((j + 1 : Nat) : Int) + 1 = c - (j : Int) := by omega
    refine ⟨fun t ht => ?_, fun t ht => by rw [e2]; exact ih.1 t (by omega)⟩
    have eq := evolve_succ z t (c - (j : Int))
    have e1 : c - (j : Int) - 1 = c - ((j + 1 : Nat) : Int) := by omega
    rw [e1, ih.1 (t + 1) (by omega), ih.1 t (by omega), ih.2 t (by omega)] at eq
    exact rule_zero_left _ eq.symm

theorem latch_fin {z : Configuration} {c : Int} {g : Nat}
    (h0 : ∀ t, t ≤ g → evolve z t c = false)
    (τ : Nat) (hτ : evolve z τ (c + 1) = true) :
    ∀ n : Nat, τ + n ≤ g → evolve z (τ + n) (c + 1) = true := by
  intro n
  induction n with
  | zero => intro _; exact hτ
  | succ n ih =>
    intro hn
    show evolve z (τ + n + 1) (c + 1) = true
    rw [evolve_succ]
    have e : c + 1 - 1 = c := by omega
    rw [e, h0 (τ + n) (by omega), ih (by omega)]
    exact rule_latch _

theorem left_eq_right_fin {z : Configuration} {c : Int} {g : Nat}
    (h0 : ∀ t, t ≤ g → evolve z t c = false) (t : Nat) (ht : t + 1 ≤ g) :
    evolve z t (c - 1) = evolve z t (c + 1) := by
  have eq := evolve_succ z t c
  rw [h0 (t + 1) ht, h0 t (by omega)] at eq
  exact rule_zero_centre _ _ eq.symm

/-- Light cone: at time `T` the seed's orbit is zero outside `[-T, T]`. -/
theorem seed_support (T : Nat) :
    ∀ i : Int, (i < -(T : Int) ∨ (T : Int) < i) → evolve seed T i = false := by
  induction T with
  | zero =>
    intro i hi
    have : i ≠ 0 := by omega
    simp [evolve, seed, this]
  | succ T ih =>
    intro i hi
    rw [evolve_succ, ih (i - 1) (by omega), ih i (by omega), ih (i + 1) (by omega)]
    rfl

/-- The left edge of the light cone is always black. -/
theorem seed_left_edge (T : Nat) : evolve seed T (-(T : Int)) = true := by
  induction T with
  | zero => rfl
  | succ T ih =>
    rw [evolve_succ]
    have e : -((T + 1 : Nat) : Int) + 1 = -(T : Int) := by omega
    rw [e, ih, seed_support T (-((T + 1 : Nat) : Int) - 1) (by omega),
        seed_support T (-((T + 1 : Nat) : Int)) (by omega)]
    rfl

/-- **Explicit window.** In the centre column of Rule 30 from a single black
cell, every time window `[T, 4T + 2]` contains both colours. -/
theorem centre_column_gap (T : Nat) (b : Bool) :
    ∃ t : Nat, T ≤ t ∧ t ≤ 4 * T + 2 ∧ evolve seed t 0 = b := by
  apply Classical.byContradiction
  intro hno
  have hz : ∀ n, n ≤ 3 * T + 2 → evolve (evolve seed T) n 0 = !b := by
    intro n hn
    rw [← evolve_add]
    cases hv : evolve seed (T + n) 0 <;> cases b <;> first
      | rfl
      | exact absurd ⟨T + n, by omega, by omega, hv⟩ hno
  cases b with
  | false =>
    -- centre is one on the whole window: checkerboard reaches past the cone
    have hk := (checkerboard_fin (c := 0) hz (2 * (T + 1))).1 0 (by omega)
    rw [par_double] at hk
    have hs := seed_support T (0 - ((2 * (T + 1) : Nat) : Int)) (Or.inl (by omega))
    have : evolve (evolve seed T) 0 (0 - ((2 * (T + 1) : Nat) : Int)) = false := hs
    rw [this] at hk
    exact Bool.noConfusion hk
  | true =>
    have h0 : ∀ t, t ≤ 3 * T + 2 → evolve (evolve seed T) t 0 = false := hz
    by_cases hs : ∃ τ, τ ≤ T ∧ evolve (evolve seed T) τ (0 + 1) = true
    · obtain ⟨τ, hτT, hτ⟩ := hs
      have hone : ∀ n, n ≤ 2 * T + 1 →
          evolve (evolve (evolve seed T) τ) n (-1) = true := by
        intro n hn
        rw [← evolve_add]
        have e : (-1 : Int) = 0 - 1 := by omega
        rw [e, left_eq_right_fin h0 (τ + n) (by omega)]
        exact latch_fin h0 τ hτ n (by omega)
      have hk := (checkerboard_fin (c := -1) hone (2 * T)).1 0 (by omega)
      rw [par_double] at hk
      have hsup := seed_support (T + τ) (-1 - ((2 * T : Nat) : Int)) (Or.inl (by omega))
      rw [evolve_add] at hsup
      have : evolve (evolve (evolve seed T) τ) 0 (-1 - ((2 * T : Nat) : Int)) = false := hsup
      rw [this] at hk
      exact Bool.noConfusion hk
    · have h1 : ∀ t, t ≤ T → evolve (evolve seed T) t (0 + 1) = false := by
        intro t ht
        cases hv : evolve (evolve seed T) t (0 + 1) with
        | false => rfl
        | true => exact absurd ⟨t, ht, hv⟩ hs
      have hl := (zero_left_fin (c := 0) (N := T)
        (fun t ht => h0 t (by omega)) h1 T).1 0 (by omega)
      have e : (0 : Int) - (T : Int) = -(T : Int) := by omega
      rw [e] at hl
      have hedge : evolve (evolve seed T) 0 (-(T : Int)) = true := seed_left_edge T
      rw [hedge] at hl
      exact Bool.noConfusion hl

/-! ### Sanity: the formal object is the Rule 30 of OEIS A051023 -/

example : (List.range 15).map (fun t => evolve seed t 0)
    = [true, true, false, true, true, true, false, false,
       true, true, false, false, false, true, false] := by decide

end Rule30

#print axioms Rule30.no_eventually_constant_column
#print axioms Rule30.centre_column_not_eventually_constant
#print axioms Rule30.centre_column_gap
