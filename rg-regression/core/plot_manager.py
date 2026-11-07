from IPython.display import clear_output
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class PlotManager:
    def __init__(self, file_path: str):
        # Train/Val Loss + Train/Val CE
        self.fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Train Loss", "Validation Loss", "Train CE", "Validation CE"),
            vertical_spacing=0.12,
        )
        self.fig.update_layout(height=700)
        self.file_path = file_path
        self._traces = {}

    def init_trial(self, trial_id: int, color: str):
        def add(metric, row, col, name):
            trace = go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name=f"{name} {trial_id}",
                line=dict(color=color),
            )
            self.fig.add_trace(trace, row=row, col=col)
            self._traces[(trial_id, metric)] = len(self.fig.data) - 1

        add("train_loss_per_epoch", 1, 1, "Tr Loss")
        add("val_loss_per_epoch", 1, 2, "Val Loss")
        add("train_ce_per_epoch", 2, 1, "Tr CE")
        add("val_ce_per_epoch", 2, 2, "Val CE")

    def add_trial_params(self, params: dict):
        text = "<br>".join(
            [f"<b>{k}</b>: {v:.4f}" if isinstance(v, float) else f"<b>{k}</b>: {v}"
             for k, v in params.items()]
        )
        annotation = dict(
            text=text,
            x=0.5,
            y=-0.25,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font=dict(size=12),
            bordercolor="black",
            borderwidth=1,
        )
        self.fig.update_layout(
            annotations=[annotation],
            margin=dict(b=200)
        )

    def update_epoch(self, trial_id: int, metrics):
        for key, values in metrics.as_dict().items():
            trace_key = (trial_id, key)
            if trace_key not in self._traces:
                continue
            idx = self._traces[trace_key]
            self.fig.data[idx].x = list(range(len(values)))
            self.fig.data[idx].y = values

    def show(self):
        clear_output(wait=True)
        self.fig.show()

    def save(self, scale: int = 3):
        # Save figure to disk. Requires `pip install -U kaleido`.
        self.fig.write_image(self.file_path, scale=scale)
