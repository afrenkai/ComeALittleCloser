import ComeCloser.Geometry

namespace ComeCloser

noncomputable def centeredProjection (dimension : ℕ)
    (basis : Fin 3 → Fin dimension → ℝ)
    (mean embedding : Fin dimension → ℝ) : Fin 3 → ℝ :=
  fun component => ∑ feature, basis component feature *
    (embedding feature - mean feature)

theorem centeredProjectionOfMean (dimension : ℕ)
    (basis : Fin 3 → Fin dimension → ℝ)
    (mean : Fin dimension → ℝ) :
    centeredProjection dimension basis mean mean = 0 := by
  funext component
  simp [centeredProjection]

noncomputable def cosineClusteringLoss (dimension samples clusters : ℕ)
    (embeddings : Fin samples → Fin dimension → ℝ)
    (centroids : Fin clusters → Fin dimension → ℝ)
    (assignment : Fin samples → Fin clusters) : ℝ :=
  ∑ sample, (1 - cosineSimilarity (embeddings sample) (centroids (assignment sample)))

def IsCosineClusterAssignment (dimension samples clusters : ℕ)
    (embeddings : Fin samples → Fin dimension → ℝ)
    (centroids : Fin clusters → Fin dimension → ℝ)
    (assignment : Fin samples → Fin clusters) : Prop :=
  ∀ sample cluster,
    cosineSimilarity (embeddings sample) (centroids cluster) ≤
      cosineSimilarity (embeddings sample) (centroids (assignment sample))

end ComeCloser
