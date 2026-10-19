from come_a_little_bit_closer.models.clap import ClapTextEncoder
from come_a_little_bit_closer.models.encoder import TextEncoder
from come_a_little_bit_closer.models.schemas import ModelEncoder, ModelSpec


def create_encoder(
    spec: ModelSpec, revision: str, device: str, max_sequence_length: int
) -> ModelEncoder:
    if spec.backend == "clap-text":
        return ClapTextEncoder(spec, revision, device, max_sequence_length)
    return TextEncoder(spec, revision, device, max_sequence_length)
