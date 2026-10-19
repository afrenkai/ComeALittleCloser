import ComeCloser.Geometry

namespace ComeCloser

noncomputable def normalizedEmbedding (dimension : ℕ) (embedding : Fin dimension → ℝ) :
    Fin dimension → ℝ :=
  fun coordinate => embedding coordinate / Real.sqrt (dotProduct embedding embedding)

noncomputable def cosineDistance {Index : Type} [finiteIndex : Fintype Index]
    (left right : Index → ℝ) : ℝ :=
  1 - cosineSimilarity left right

noncomputable def projectedEmbedding (dimension outputDimension : ℕ)
    (weights : Fin outputDimension → Fin dimension → ℝ)
    (embedding : Fin dimension → ℝ) : Fin outputDimension → ℝ :=
  normalizedEmbedding outputDimension
    (fun output => ∑ feature, weights output feature * embedding feature)

def eligibleCandidates (candidates : Finset ℕ) (eligible : ℕ → Bool) : Finset ℕ :=
  candidates.filter (fun candidate => eligible candidate = true)

noncomputable def rankedCandidates (candidates : List ℕ) (score : ℕ → ℝ)
    (neighborCount : ℕ) : List ℕ :=
  (candidates.mergeSort (fun left right =>
    decide ((-score left, left) ≤ (-score right, right)))).take neighborCount

noncomputable def overlapAtK (reference retrieved : Finset ℕ) : ℝ :=
  ((reference ∩ retrieved).card : ℝ) / (reference.card : ℝ)

noncomputable def precisionAtK (eligible : ℕ → Bool) (retrieved : Finset ℕ) : ℝ :=
  ((eligibleCandidates retrieved eligible).card : ℝ) / (retrieved.card : ℝ)

noncomputable def excessKilometers (distance : ℕ → ℝ)
    (referenceFirst retrievedFirst : ℕ) : ℝ :=
  distance retrievedFirst - distance referenceFirst

noncomputable def spearmanCorrelation (candidateCount : ℕ)
    (referenceRank retrievedRank : Fin candidateCount → ℝ) : ℝ :=
  1 - 6 * (∑ candidate, (referenceRank candidate - retrievedRank candidate) ^ 2) /
    ((candidateCount : ℝ) * ((candidateCount : ℝ) ^ 2 - 1))

noncomputable def fusionScore (spatialWeight semanticScore geographicScore : ℝ) : ℝ :=
  (1 - spatialWeight) * semanticScore + spatialWeight * geographicScore

noncomputable def fusedVector (spatialWeight : ℝ) (dimension : ℕ)
    (semantic : Fin dimension → ℝ) (spatial : Fin 3 → ℝ) :
    Fin dimension ⊕ Fin 3 → ℝ :=
  Sum.elim (fun coordinate => Real.sqrt (1 - spatialWeight) * semantic coordinate)
    (fun coordinate => Real.sqrt spatialWeight * spatial coordinate)

theorem fusedDotProduct (spatialWeight : ℝ)
    (weightNonnegative : 0 ≤ spatialWeight) (weightBound : spatialWeight ≤ 1)
    (dimension : ℕ) (leftSemantic rightSemantic : Fin dimension → ℝ)
    (leftSpatial rightSpatial : Fin 3 → ℝ) :
    dotProduct (fusedVector spatialWeight dimension leftSemantic leftSpatial)
      (fusedVector spatialWeight dimension rightSemantic rightSpatial) =
      fusionScore spatialWeight (dotProduct leftSemantic rightSemantic)
        (dotProduct leftSpatial rightSpatial) := by
  have semanticTerm (coordinate : Fin dimension) :
      (Real.sqrt (1 - spatialWeight) * leftSemantic coordinate) *
        (Real.sqrt (1 - spatialWeight) * rightSemantic coordinate) =
        (1 - spatialWeight) * (leftSemantic coordinate * rightSemantic coordinate) := by
    calc
      (Real.sqrt (1 - spatialWeight) * leftSemantic coordinate) *
          (Real.sqrt (1 - spatialWeight) * rightSemantic coordinate) =
          Real.sqrt (1 - spatialWeight) ^ 2 *
            (leftSemantic coordinate * rightSemantic coordinate) := by ring
      Real.sqrt (1 - spatialWeight) ^ 2 *
          (leftSemantic coordinate * rightSemantic coordinate) =
          (1 - spatialWeight) * (leftSemantic coordinate * rightSemantic coordinate) := by
        rw [Real.sq_sqrt (sub_nonneg.mpr weightBound)]
  have spatialTerm (coordinate : Fin 3) :
      (Real.sqrt spatialWeight * leftSpatial coordinate) *
        (Real.sqrt spatialWeight * rightSpatial coordinate) =
        spatialWeight * (leftSpatial coordinate * rightSpatial coordinate) := by
    calc
      (Real.sqrt spatialWeight * leftSpatial coordinate) *
          (Real.sqrt spatialWeight * rightSpatial coordinate) =
          Real.sqrt spatialWeight ^ 2 *
            (leftSpatial coordinate * rightSpatial coordinate) := by ring
      Real.sqrt spatialWeight ^ 2 *
          (leftSpatial coordinate * rightSpatial coordinate) =
          spatialWeight * (leftSpatial coordinate * rightSpatial coordinate) := by
        rw [Real.sq_sqrt weightNonnegative]
  simp only [dotProduct, fusedVector, Fintype.sum_sum_type, Sum.elim_inl,
    Sum.elim_inr, fusionScore]
  simp_rw [semanticTerm, spatialTerm]
  rw [Finset.mul_sum, Finset.mul_sum]

theorem fusedVectorUnit (spatialWeight : ℝ)
    (weightNonnegative : 0 ≤ spatialWeight) (weightBound : spatialWeight ≤ 1)
    (dimension : ℕ) (semantic : Fin dimension → ℝ) (spatial : Fin 3 → ℝ)
    (semanticUnit : dotProduct semantic semantic = 1)
    (spatialUnit : dotProduct spatial spatial = 1) :
    dotProduct (fusedVector spatialWeight dimension semantic spatial)
      (fusedVector spatialWeight dimension semantic spatial) = 1 := by
  rw [fusedDotProduct spatialWeight weightNonnegative weightBound dimension,
    semanticUnit, spatialUnit]
  simp [fusionScore]

noncomputable def ridgeObjective (sampleCount dimension : ℕ)
    (hidden : Fin sampleCount → Fin dimension → ℝ)
    (target : Fin sampleCount → Fin 3 → ℝ)
    (weights : Fin dimension → Fin 3 → ℝ) (regularization : ℝ) : ℝ :=
  (∑ sample, ∑ coordinate,
    ((∑ feature, hidden sample feature * weights feature coordinate) -
      target sample coordinate) ^ 2) / (sampleCount : ℝ) +
    regularization * ∑ feature, ∑ coordinate, weights feature coordinate ^ 2

def IsRidgeMinimizer (sampleCount dimension : ℕ)
    (hidden : Fin sampleCount → Fin dimension → ℝ)
    (target : Fin sampleCount → Fin 3 → ℝ) (regularization : ℝ)
    (weights : Fin dimension → Fin 3 → ℝ) : Prop :=
  ∀ alternativeWeights : Fin dimension → Fin 3 → ℝ,
    ridgeObjective sampleCount dimension hidden target weights regularization ≤
      ridgeObjective sampleCount dimension hidden target alternativeWeights regularization

noncomputable def geographicTripletLoss (margin positiveScore negativeScore : ℝ) : ℝ :=
  max 0 (margin - positiveScore + negativeScore)

noncomputable def distillationLoss (pairCount : ℕ)
    (studentScore teacherScore : Fin pairCount → ℝ) : ℝ :=
  (∑ pairIndex, (studentScore pairIndex - teacherScore pairIndex) ^ 2) /
    (pairCount : ℝ)

noncomputable def adaptationObjective (tripletCount pairCount : ℕ)
    (positiveScore negativeScore : Fin tripletCount → ℝ)
    (studentScore teacherScore : Fin pairCount → ℝ) (margin penalty : ℝ) : ℝ :=
  (∑ tripletIndex, geographicTripletLoss margin
    (positiveScore tripletIndex) (negativeScore tripletIndex)) / (tripletCount : ℝ) +
    penalty * distillationLoss pairCount studentScore teacherScore

noncomputable def h3CandidateRecall (reference candidatePool : Finset ℕ) : ℝ :=
  overlapAtK reference candidatePool

structure H3Interface where
  cell : Coordinates → ℕ
  gridDistance : ℕ → ℕ → Option ℕ

def h3CandidatePool (indexing : H3Interface) (origin : Coordinates)
    (locations : ℕ → Coordinates) (candidates : Finset ℕ) (ringBudget : ℕ) : Finset ℕ :=
  candidates.filter (fun candidate =>
    (indexing.gridDistance (indexing.cell origin) (indexing.cell (locations candidate))).any
      (fun hopCount => decide (hopCount ≤ ringBudget)))

theorem excessKilometersNonnegative (distance : ℕ → ℝ)
    (referenceFirst retrievedFirst : ℕ)
    (referenceMinimal : distance referenceFirst ≤ distance retrievedFirst) :
    0 ≤ excessKilometers distance referenceFirst retrievedFirst := by
  exact sub_nonneg.mpr referenceMinimal

theorem fusionSemanticEndpoint (semanticScore geographicScore : ℝ) :
    fusionScore 0 semanticScore geographicScore = semanticScore := by
  simp [fusionScore]

theorem fusionGeographicEndpoint (semanticScore geographicScore : ℝ) :
    fusionScore 1 semanticScore geographicScore = geographicScore := by
  simp [fusionScore]

end ComeCloser
