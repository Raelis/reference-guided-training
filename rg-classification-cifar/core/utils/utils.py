from typing import Type
import json
import torch

def load_model(model_class: Type, path: str, device: str, input_dim, horizon):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_params = checkpoint["model_params"]
    model_params, other_params = model_class.prepare_model_params(model_params)
    model_params["num_features"] = input_dim
    model_params["output_dim"] = horizon
    model = model_class(**model_params).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model_class.model_postprocess(model)
    return model

def save_parquet_with_config(df, path, config: dict):
    """Persist a DataFrame and its run metadata in Parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df)

    metadata = table.schema.metadata or {}
    metadata = {
        **metadata,
        b"config": json.dumps(config).encode("utf-8")
    }

    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path)

def load_parquet_with_config(path):
    """Load a DataFrame and its run metadata from Parquet."""
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    df = table.to_pandas()

    metadata = table.schema.metadata or {}
    config = {}

    if b"config" in metadata:
        config = json.loads(metadata[b"config"].decode("utf-8"))

    return df, config
