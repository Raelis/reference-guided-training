import os
import random
import numpy as np
import json
import torch

def save_model(state, path):
    checkpoint = {
        "model_class": state.model.__class__,
        "model_params": state.params,
        "state_dict": state.model.state_dict(),
        "metrics": state.metrics.as_dict()
    }
    torch.save(checkpoint, path)

def end_train_save_model(state, path):
    last_checkpoint = None
    if os.path.exists(path):
        last_checkpoint = torch.load(path, weights_only=False)
    checkpoint = {
        "model_class": state.model.__class__,
        "model_params": state.params,
        "state_dict": (
            last_checkpoint["state_dict"]
            if last_checkpoint is not None
            else state.model.state_dict()
        ),
        "end_state_dict": state.model.state_dict(),
        "metrics": state.metrics.as_dict()
    }
    torch.save(checkpoint, path)

def save_parquet_with_config(df, path, config: dict):
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
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    df = table.to_pandas()

    metadata = table.schema.metadata or {}
    config = {}

    if b"config" in metadata:
        config = json.loads(metadata[b"config"].decode("utf-8"))

    return df, config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def minimum_finite(values, metric_name):
    if values is None:
        raise FloatingPointError(f'Missing {metric_name}')
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise FloatingPointError(f'No finite {metric_name} across training epochs')
    return float(finite.min())
