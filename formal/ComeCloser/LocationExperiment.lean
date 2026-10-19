import ComeCloser.Objectives

namespace ComeCloser

noncomputable def affineCoordinatePrediction (dimension : ℕ)
    (weights : Fin dimension → Fin 3 → ℝ) (intercept : Fin 3 → ℝ)
    (embedding : Fin dimension → ℝ) : Fin 3 → ℝ :=
  fun coordinate => intercept coordinate +
    ∑ feature, embedding feature * weights feature coordinate

noncomputable def affineRidgeObjective (sampleCount dimension : ℕ)
    (embeddings : Fin sampleCount → Fin dimension → ℝ)
    (targets : Fin sampleCount → Fin 3 → ℝ)
    (weights : Fin dimension → Fin 3 → ℝ) (intercept : Fin 3 → ℝ)
    (regularization : ℝ) : ℝ :=
  (∑ sample, ∑ coordinate,
    (affineCoordinatePrediction dimension weights intercept
      (embeddings sample) coordinate - targets sample coordinate) ^ 2) +
    regularization * ∑ feature, ∑ coordinate, weights feature coordinate ^ 2

noncomputable def unitCoordinatePrediction (prediction : Fin 3 → ℝ)
    (zeroFallback : Coordinates) : Fin 3 → ℝ :=
  if prediction = 0 then spatialVector zeroFallback
  else normalizedEmbedding 3 prediction

noncomputable def coordinatePredictionError (radius : ℝ)
    (prediction : Fin 3 → ℝ) (truth zeroFallback : Coordinates) : ℝ :=
  angularDistance radius (unitCoordinatePrediction prediction zeroFallback)
    (spatialVector truth)

def genreEligibleCandidates (candidates : Finset ℕ)
    (hasGenre : ℕ → Bool) : Finset ℕ :=
  eligibleCandidates candidates hasGenre

structure FiniteNeighborhood where
  stations : Finset ℕ
  nonempty : stations.Nonempty

noncomputable def meanDistance (distance : ℕ → ℝ)
    (neighbors : FiniteNeighborhood) : ℝ :=
  (∑ station ∈ neighbors.stations, distance station) /
    (neighbors.stations.card : ℝ)

noncomputable def meanGeographicDistance (radius : ℝ)
    (locations : ℕ → Coordinates) (origin : Coordinates)
    (neighbors : FiniteNeighborhood) : ℝ :=
  meanDistance (fun station => haversineDistance radius origin (locations station))
    neighbors

noncomputable def geographicRegret (distance : ℕ → ℝ)
    (reference retrieved : FiniteNeighborhood) : ℝ :=
  meanDistance distance retrieved - meanDistance distance reference

noncomputable def topKTurnover (neighborCount : ℕ)
    (left right : Finset ℕ) : ℝ :=
  1 - ((left ∩ right).card : ℝ) / (neighborCount : ℝ)

noncomputable def originResponsiveness (radius : ℝ)
    (locations : ℕ → Coordinates) (originA originB : Coordinates)
    (neighborsA neighborsB : FiniteNeighborhood) : ℝ :=
  ((meanGeographicDistance radius locations originB neighborsA -
      meanGeographicDistance radius locations originB neighborsB) +
    (meanGeographicDistance radius locations originA neighborsB -
      meanGeographicDistance radius locations originA neighborsA)) / 2

theorem affineRidgeObjectiveNonnegative (sampleCount dimension : ℕ)
    (embeddings : Fin sampleCount → Fin dimension → ℝ)
    (targets : Fin sampleCount → Fin 3 → ℝ)
    (weights : Fin dimension → Fin 3 → ℝ) (intercept : Fin 3 → ℝ)
    (regularization : ℝ) (regularizationPositive : 0 < regularization) :
    0 ≤ affineRidgeObjective sampleCount dimension embeddings targets
      weights intercept regularization := by
  unfold affineRidgeObjective
  positivity

theorem affineRidgeObjectiveZeroAtExactConstantFit (sampleCount dimension : ℕ)
    (embeddings : Fin sampleCount → Fin dimension → ℝ)
    (targets : Fin sampleCount → Fin 3 → ℝ)
    (intercept : Fin 3 → ℝ) (regularization : ℝ)
    (exactFit : ∀ sample coordinate, targets sample coordinate = intercept coordinate) :
    affineRidgeObjective sampleCount dimension embeddings targets
      0 intercept regularization = 0 := by
  simp [affineRidgeObjective, affineCoordinatePrediction, exactFit]

theorem unitCoordinatePredictionZero (zeroFallback : Coordinates) :
    unitCoordinatePrediction 0 zeroFallback = spatialVector zeroFallback := by
  simp [unitCoordinatePrediction]

theorem genreEligibleCandidatesMembership (candidates : Finset ℕ)
    (hasGenre : ℕ → Bool) (station : ℕ) :
    station ∈ genreEligibleCandidates candidates hasGenre ↔
      station ∈ candidates ∧ hasGenre station = true := by
  simp [genreEligibleCandidates, eligibleCandidates]

theorem geographicRegretNonnegative (distance : ℕ → ℝ)
    (reference retrieved : FiniteNeighborhood)
    (referenceMeanMinimal : meanDistance distance reference ≤
      meanDistance distance retrieved) :
    0 ≤ geographicRegret distance reference retrieved := by
  exact sub_nonneg.mpr referenceMeanMinimal

theorem topKTurnoverIdentical (neighborCount : ℕ)
    (neighborsA neighborsB : Finset ℕ)
    (neighborCountPositive : 0 < neighborCount)
    (cardinality : neighborsA.card = neighborCount)
    (identical : neighborsA = neighborsB) :
    topKTurnover neighborCount neighborsA neighborsB = 0 := by
  have denominatorNonzero : (neighborCount : ℝ) ≠ 0 := by
    exact_mod_cast Nat.ne_of_gt neighborCountPositive
  simp [topKTurnover, ← identical, cardinality, denominatorNonzero]

theorem originResponsivenessIdentical (radius : ℝ)
    (locations : ℕ → Coordinates) (originA originB : Coordinates)
    (neighborsA neighborsB : FiniteNeighborhood)
    (identical : neighborsA.stations = neighborsB.stations) :
    originResponsiveness radius locations originA originB neighborsA neighborsB = 0 := by
  simp [originResponsiveness, meanGeographicDistance, meanDistance, identical]

end ComeCloser
