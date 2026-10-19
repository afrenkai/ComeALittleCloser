import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse
import Mathlib.Data.Matrix.Notation
import Mathlib.Tactic

namespace ComeCloser

structure Coordinates where
  latitude : ℝ
  longitude : ℝ

noncomputable def dotProduct {Index : Type} [finiteIndex : Fintype Index]
    (left right : Index → ℝ) : ℝ :=
  ∑ coordinate, left coordinate * right coordinate

noncomputable def cosineSimilarity {Index : Type} [finiteIndex : Fintype Index]
    (left right : Index → ℝ) : ℝ :=
  dotProduct left right /
    (Real.sqrt (dotProduct left left) * Real.sqrt (dotProduct right right))

noncomputable def spatialVector (point : Coordinates) : Fin 3 → ℝ :=
  ![Real.cos point.latitude * Real.cos point.longitude,
    Real.cos point.latitude * Real.sin point.longitude,
    Real.sin point.latitude]

noncomputable def haversineFactor (left right : Coordinates) : ℝ :=
  Real.sin ((right.latitude - left.latitude) / 2) ^ 2 +
    Real.cos left.latitude * Real.cos right.latitude *
      Real.sin ((right.longitude - left.longitude) / 2) ^ 2

noncomputable def haversineDistance (radius : ℝ) (left right : Coordinates) : ℝ :=
  2 * radius * Real.arcsin (Real.sqrt (haversineFactor left right))

noncomputable def angularDistance (radius : ℝ) (left right : Fin 3 → ℝ) : ℝ :=
  radius * Real.arccos (dotProduct left right)

theorem spatialVectorUnit (point : Coordinates) :
    dotProduct (spatialVector point) (spatialVector point) = 1 := by
  calc
    dotProduct (spatialVector point) (spatialVector point) =
        Real.cos point.latitude ^ 2 *
          (Real.cos point.longitude ^ 2 + Real.sin point.longitude ^ 2) +
          Real.sin point.latitude ^ 2 := by
      simp only [dotProduct, spatialVector, Fin.sum_univ_succ,
        Matrix.cons_val_zero, Matrix.cons_val_succ, Fin.sum_univ_zero, add_zero]
      ring
    Real.cos point.latitude ^ 2 *
        (Real.cos point.longitude ^ 2 + Real.sin point.longitude ^ 2) +
        Real.sin point.latitude ^ 2 = 1 := by
      rw [Real.cos_sq_add_sin_sq, mul_one, Real.cos_sq_add_sin_sq]

theorem spatialDotFormula (left right : Coordinates) :
    dotProduct (spatialVector left) (spatialVector right) =
      Real.sin left.latitude * Real.sin right.latitude +
        Real.cos left.latitude * Real.cos right.latitude *
          Real.cos (left.longitude - right.longitude) := by
  simp [dotProduct, spatialVector, Fin.sum_univ_succ, Real.cos_sub]
  ring

theorem unitDotBounds (left right : Fin 3 → ℝ)
    (leftUnit : dotProduct left left = 1)
    (rightUnit : dotProduct right right = 1) :
    -1 ≤ dotProduct left right ∧ dotProduct left right ≤ 1 := by
  simp [dotProduct, Fin.sum_univ_succ] at leftUnit rightUnit ⊢
  constructor
  · nlinarith [sq_nonneg (left 0 + right 0),
      sq_nonneg (left 1 + right 1), sq_nonneg (left 2 + right 2)]
  · nlinarith [sq_nonneg (left 0 - right 0),
      sq_nonneg (left 1 - right 1), sq_nonneg (left 2 - right 2)]

theorem spatialCosineEqualsDot (left right : Coordinates) :
    cosineSimilarity (spatialVector left) (spatialVector right) =
      dotProduct (spatialVector left) (spatialVector right) := by
  simp [cosineSimilarity, spatialVectorUnit]

theorem haversineFactorIdentity (left right : Coordinates) :
    haversineFactor left right =
      (1 - dotProduct (spatialVector left) (spatialVector right)) / 2 := by
  rw [haversineFactor, spatialDotFormula]
  rw [Real.sin_sq_eq_half_sub, Real.sin_sq_eq_half_sub]
  have latitudeHalves : 2 * ((right.latitude - left.latitude) / 2) =
      right.latitude - left.latitude := by ring
  have longitudeHalves : 2 * ((right.longitude - left.longitude) / 2) =
      right.longitude - left.longitude := by ring
  rw [latitudeHalves, longitudeHalves, Real.cos_sub, Real.cos_sub, Real.cos_sub]
  ring

theorem haversineFactorBounds (left right : Coordinates) :
    0 ≤ haversineFactor left right ∧ haversineFactor left right ≤ 1 := by
  have bounds := unitDotBounds (spatialVector left) (spatialVector right)
    (spatialVectorUnit left) (spatialVectorUnit right)
  rw [haversineFactorIdentity]
  constructor <;> linarith [bounds.1, bounds.2]

theorem arccosHaversineIdentity (factor : ℝ)
    (factorNonnegative : 0 ≤ factor) (factorBound : factor ≤ 1) :
    Real.arccos (1 - 2 * factor) =
      2 * Real.arcsin (Real.sqrt factor) := by
  have rootNonnegative := Real.sqrt_nonneg factor
  have rootBound : Real.sqrt factor ≤ 1 := Real.sqrt_le_one.mpr factorBound
  have rootLower : -1 ≤ Real.sqrt factor := by linarith
  have sineValue := Real.sin_arcsin rootLower rootBound
  have squareValue := Real.sq_sqrt factorNonnegative
  have arcsineNonnegative : 0 ≤ Real.arcsin (Real.sqrt factor) :=
    Real.arcsin_nonneg.mpr rootNonnegative
  have angleNonnegative : 0 ≤ 2 * Real.arcsin (Real.sqrt factor) := by linarith
  have angleBound : 2 * Real.arcsin (Real.sqrt factor) ≤ Real.pi := by
    linarith [Real.arcsin_le_pi_div_two (Real.sqrt factor)]
  apply Real.arccos_eq_of_eq_cos angleNonnegative angleBound
  rw [Real.cos_two_mul]
  nlinarith [Real.sin_sq_add_cos_sq (Real.arcsin (Real.sqrt factor))]

theorem haversineEqualsAngular (radius : ℝ) (left right : Coordinates) :
    haversineDistance radius left right =
      angularDistance radius (spatialVector left) (spatialVector right) := by
  have bounds := haversineFactorBounds left right
  have dotIdentity : dotProduct (spatialVector left) (spatialVector right) =
      1 - 2 * haversineFactor left right := by
    rw [haversineFactorIdentity]
    ring
  rw [haversineDistance, angularDistance, dotIdentity,
    arccosHaversineIdentity (haversineFactor left right) bounds.1 bounds.2]
  ring

theorem sphericalRankingEquivalence (radius : ℝ) (radiusPositive : 0 < radius)
    (origin left right : Coordinates) :
    haversineDistance radius origin left ≤ haversineDistance radius origin right ↔
      cosineSimilarity (spatialVector origin) (spatialVector right) ≤
        cosineSimilarity (spatialVector origin) (spatialVector left) := by
  rw [haversineEqualsAngular, haversineEqualsAngular,
    spatialCosineEqualsDot, spatialCosineEqualsDot, angularDistance, angularDistance]
  have leftBounds := unitDotBounds (spatialVector origin) (spatialVector left)
    (spatialVectorUnit origin) (spatialVectorUnit left)
  have rightBounds := unitDotBounds (spatialVector origin) (spatialVector right)
    (spatialVectorUnit origin) (spatialVectorUnit right)
  constructor
  · intro distanceOrder
    by_contra reversedScore
    have strictScore := lt_of_not_ge reversedScore
    have strictAngle := Real.arccos_lt_arccos leftBounds.1 strictScore rightBounds.2
    have strictDistance := mul_lt_mul_of_pos_left strictAngle radiusPositive
    linarith
  · intro scoreOrder
    exact mul_le_mul_of_nonneg_left (Real.arccos_le_arccos scoreOrder)
      (le_of_lt radiusPositive)

end ComeCloser
