from come_a_little_bit_closer.models.layer_schemas import LayerModelSpec


def layer_model_specs() -> list[LayerModelSpec]:
    qwen_06b_base = LayerModelSpec(
        model_id="qwen3-0.6b-base",
        family="qwen3-0.6b",
        stage="base",
        stage_order=1,
        checkpoint="Qwen/Qwen3-0.6B-Base",
        revision="da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        architecture="decoder",
        end_token="<|endoftext|>",
    )
    xlmr_base = LayerModelSpec(
        model_id="xlmr-base",
        family="xlmr-base",
        stage="base",
        stage_order=1,
        checkpoint="FacebookAI/xlm-roberta-base",
        revision="e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
        architecture="encoder",
    )
    return [
        qwen_06b_base,
        LayerModelSpec(
            model_id="qwen3-0.6b-embedding",
            family="qwen3-0.6b",
            stage="retrieval",
            stage_order=2,
            checkpoint="Qwen/Qwen3-Embedding-0.6B",
            revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            architecture="decoder",
            end_token="<|endoftext|>",
        ),
        LayerModelSpec(
            model_id="qwen3-4b-base",
            family="qwen3-4b",
            stage="base",
            stage_order=1,
            checkpoint="Qwen/Qwen3-4B-Base",
            revision="906bfd4b4dc7f14ee4320094d8b41684abff8539",
            architecture="decoder",
            end_token="<|endoftext|>",
        ),
        LayerModelSpec(
            model_id="qwen3-4b-embedding",
            family="qwen3-4b",
            stage="retrieval",
            stage_order=2,
            checkpoint="Qwen/Qwen3-Embedding-4B",
            revision="5cf2132abc99cad020ac570b19d031efec650f2b",
            architecture="decoder",
            end_token="<|endoftext|>",
        ),
        xlmr_base,
        LayerModelSpec(
            model_id="e5-base",
            family="xlmr-base",
            stage="retrieval",
            stage_order=2,
            checkpoint="intfloat/multilingual-e5-base",
            revision="d128750597153bb5987e10b1c3493a34e5a4502a",
            architecture="encoder",
        ),
        LayerModelSpec(
            model_id="xlmr-large",
            family="xlmr-large",
            stage="base",
            stage_order=1,
            checkpoint="FacebookAI/xlm-roberta-large",
            revision="c23d21b0620b635a76227c604d44e43a9f0ee389",
            architecture="encoder",
        ),
        LayerModelSpec(
            model_id="bge-m3-retromae",
            family="xlmr-large",
            stage="long-context pretraining",
            stage_order=2,
            checkpoint="BAAI/bge-m3-retromae",
            revision="95c4f81ef8f7911dcac8e384532489c32c8dea64",
            architecture="encoder",
        ),
        LayerModelSpec(
            model_id="bge-m3-unsupervised",
            family="xlmr-large",
            stage="unsupervised contrastive",
            stage_order=3,
            checkpoint="BAAI/bge-m3-unsupervised",
            revision="46f03bc86361cf88102b0b517b36c8259f2946b1",
            architecture="encoder",
        ),
        LayerModelSpec(
            model_id="bge-m3",
            family="xlmr-large",
            stage="retrieval",
            stage_order=4,
            checkpoint="BAAI/bge-m3",
            revision="5617a9f61b028005a4858fdac845db406aefb181",
            architecture="encoder",
        ),
        qwen_06b_base.model_copy(
            update={
                "model_id": "qwen3-0.6b-random",
                "stage": "random",
                "stage_order": 0,
                "random_init": True,
            }
        ),
        xlmr_base.model_copy(
            update={
                "model_id": "xlmr-base-random",
                "stage": "random",
                "stage_order": 0,
                "random_init": True,
            }
        ),
    ]


def find_layer_model(model_id: str) -> LayerModelSpec:
    for spec in layer_model_specs():
        if spec.model_id == model_id:
            return spec
    raise ValueError(f"Unknown layer model: {model_id}")
