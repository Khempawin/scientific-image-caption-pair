from transformers import AutoModel

model_path = "/home/horton/datasets/meta-scir/models/specter2-special-token-base"

model = AutoModel.from_pretrained(
        model_path,
        # cache_dir=model_args.cache_dir,
        # revision=model_args.model_revision,
        # token=model_args.token,
        trust_remote_code=True,
    )