from come_a_little_bit_closer.models.schemas import ModelSpec


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            model_id="qwen3",
            label="Qwen3-Embedding · 0.6B",
            checkpoint="Qwen/Qwen3-Embedding-0.6B",
            revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            description="Learned text embeddings of station metadata and coordinates.",
            query_prefix=(
                "Instruct: Given a radio search query and an origin, retrieve the "
                "geographically nearest matching radio station.\nQuery: "
            ),
        ),
        ModelSpec(
            model_id="bge-m3",
            label="BGE-M3 · dense",
            checkpoint="BAAI/bge-m3",
            revision="5617a9f61b028005a4858fdac845db406aefb181",
            description="Dense multilingual embeddings; explicit-coordinate input.",
        ),
        ModelSpec(
            model_id="e5",
            label="Multilingual E5 · base",
            checkpoint="intfloat/multilingual-e5-base",
            revision="d128750597153bb5987e10b1c3493a34e5a4502a",
            description="Multilingual retrieval embeddings; explicit-coordinate input.",
            query_prefix="query: ",
            document_prefix="passage: ",
        ),
        ModelSpec(
            model_id="embeddinggemma",
            label="EmbeddingGemma 2",
            checkpoint="google/embeddinggemma-2",
            revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
            description="Text retrieval with native SearchQuery/Document prompts.",
            query_prompt_name="SearchQuery",
            document_prompt_name="Document",
            text_only=True,
        ),
        ModelSpec(
            model_id="clap",
            label="CLAP · text tower",
            checkpoint="laion/clap-htsat-unfused",
            revision="8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
            description="CLAP text projection; station audio remains unencoded.",
            backend="clap-text",
        ),
    ]


def find_model(model_id: str) -> ModelSpec:
    for spec in model_specs():
        if spec.model_id == model_id:
            return spec
    raise ValueError(f"Unknown embedding model: {model_id}")
