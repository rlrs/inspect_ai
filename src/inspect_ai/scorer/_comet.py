from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, cast

from inspect_ai._util.error import PrerequisiteError, pip_dependency_error
from inspect_ai.solver._task_state import TaskState

from ._metric import Score
from ._metrics import mean, stderr
from ._scorer import Scorer, scorer
from ._target import Target

SourceResolver = str | Callable[[TaskState], str]
ReferenceResolver = str | Callable[[TaskState, Target], str | Sequence[str]] | None
CandidateResolver = Callable[[TaskState], str] | None


@scorer(metrics=[mean(), stderr()])
def comet(
    model: str = "Unbabel/wmt22-comet-da",
    *,
    source: SourceResolver = "input",
    reference: ReferenceResolver = None,
    candidate: CandidateResolver = None,
    model_storage_path: str | None = None,
    local_files_only: bool = False,
    device: str = "auto",
) -> Scorer:
    """Score MT output using COMET.

    This scorer ports COMET's inference path for regression checkpoints and does not
    require the `unbabel-comet` package.

    Args:
      model: COMET model id (e.g. ``Unbabel/wmt22-comet-da``) or path to a local
        checkpoint file / model directory.
      source: Source text resolver. Use ``"input"`` (default), a metadata key, or a
        callable.
      reference: Reference resolver. Defaults to the sample target(s). You may provide
        a metadata key or callable instead.
      candidate: Candidate translation resolver. Defaults to
        ``state.output.completion``.
      model_storage_path: Optional Hugging Face cache directory for downloaded models.
      local_files_only: Only load models from local cache / local filesystem.
      device: ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
    """
    runtime: _CometRuntime | None = None
    load_lock = asyncio.Lock()
    predict_lock = asyncio.Lock()

    async def get_runtime() -> _CometRuntime:
        nonlocal runtime
        if runtime is None:
            async with load_lock:
                if runtime is None:
                    runtime = await asyncio.to_thread(
                        _get_comet_runtime,
                        model,
                        model_storage_path,
                        local_files_only,
                        device,
                    )
        return runtime

    async def score(state: TaskState, target: Target) -> Score:
        candidate_text = candidate(state) if candidate else state.output.completion
        source_text = _resolve_source_text(source=source, state=state)
        references = _resolve_reference_texts(
            reference=reference,
            state=state,
            target=target,
        )
        if len(references) == 0:
            raise PrerequisiteError(
                "COMET scorer requires at least one reference (from `target` or `reference=`)."
            )

        scorer_runtime = await get_runtime()
        batch_sources = [source_text] * len(references)
        batch_candidates = [candidate_text] * len(references)

        async with predict_lock:
            reference_scores = await asyncio.to_thread(
                score_comet_texts,
                scorer_runtime,
                batch_sources,
                batch_candidates,
                references,
            )

        best_score = max(reference_scores)
        best_reference = references[reference_scores.index(best_score)]
        return Score(
            value=best_score,
            answer=candidate_text,
            metadata={
                "model": model,
                "reference": best_reference,
                "reference_scores": reference_scores,
            },
        )

    return score


def _resolve_source_text(source: SourceResolver, state: TaskState) -> str:
    if callable(source):
        source_text = source(state)
    elif source == "input":
        try:
            source_text = state.input_text
        except ValueError as ex:
            raise PrerequisiteError(
                "COMET scorer source='input' requires TaskState.input_text to be available. "
                "Provide `source=` as a metadata key or callable."
            ) from ex
    else:
        source_text = cast(str | None, state.metadata.get(source))
        if source_text is None:
            raise PrerequisiteError(
                f"COMET scorer could not find source text in metadata['{source}']."
            )

    if not isinstance(source_text, str):
        raise PrerequisiteError(
            "COMET scorer source resolver must return a string."
        )
    return source_text


def _resolve_reference_texts(
    reference: ReferenceResolver,
    state: TaskState,
    target: Target,
) -> list[str]:
    if reference is None or reference == "target":
        return [ref for ref in target.target if ref]
    elif callable(reference):
        resolved_reference = reference(state, target)
    else:
        resolved_reference = state.metadata.get(reference)
        if resolved_reference is None:
            raise PrerequisiteError(
                f"COMET scorer could not find reference text in metadata['{reference}']."
            )

    if isinstance(resolved_reference, str):
        return [resolved_reference] if resolved_reference else []
    elif isinstance(resolved_reference, Sequence):
        if not all(isinstance(value, str) for value in resolved_reference):
            raise PrerequisiteError(
                "COMET scorer reference resolver must return a string or sequence of strings."
            )
        return [value for value in resolved_reference if value]
    else:
        raise PrerequisiteError(
            "COMET scorer reference resolver must return a string or sequence of strings."
        )


def score_comet_texts(
    runtime: "_CometRuntime",
    sources: Sequence[str],
    candidates: Sequence[str],
    references: Sequence[str] | None,
) -> list[float]:
    """Score a batch with a loaded COMET runtime."""
    return runtime.score(
        sources=list(sources),
        candidates=list(candidates),
        references=list(references) if references is not None else None,
    )


@dataclass(frozen=True)
class _CometRuntimeKey:
    model: str
    model_storage_path: str | None
    local_files_only: bool
    device: str


_COMET_RUNTIME_CACHE: dict[_CometRuntimeKey, "_CometRuntime"] = {}
_COMET_RUNTIME_CACHE_LOCK = Lock()


def _get_comet_runtime(
    model: str,
    model_storage_path: str | None,
    local_files_only: bool,
    device: str,
) -> "_CometRuntime":
    key = _CometRuntimeKey(
        model=model,
        model_storage_path=model_storage_path,
        local_files_only=local_files_only,
        device=device,
    )
    with _COMET_RUNTIME_CACHE_LOCK:
        runtime = _COMET_RUNTIME_CACHE.get(key)
        if runtime is None:
            runtime = _create_comet_runtime(
                model=model,
                model_storage_path=model_storage_path,
                local_files_only=local_files_only,
                device=device,
            )
            _COMET_RUNTIME_CACHE[key] = runtime
        return runtime


def _create_comet_runtime(
    model: str,
    model_storage_path: str | None,
    local_files_only: bool,
    device: str,
) -> "_CometRuntime":
    torch, AutoModel, AutoTokenizer, snapshot_download, yaml = _import_comet_dependencies()

    checkpoint_path, hparams = _resolve_checkpoint_and_hparams(
        model=model,
        model_storage_path=model_storage_path,
        local_files_only=local_files_only,
        snapshot_download=snapshot_download,
        yaml=yaml,
    )

    class_identifier = str(hparams.get("class_identifier", ""))
    if class_identifier not in (
        "regression_metric",
        "referenceless_regression_metric",
    ):
        raise PrerequisiteError(
            "COMET scorer currently supports regression checkpoints only "
            "(class_identifier: regression_metric or referenceless_regression_metric)."
        )

    pretrained_model = str(hparams.get("pretrained_model", "")).strip()
    if not pretrained_model:
        raise PrerequisiteError(
            f"COMET hparams is missing `pretrained_model` in {checkpoint_path}."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model,
            use_fast=True,
            local_files_only=local_files_only,
        )
    except Exception:
        # Some tokenizer builds require optional protobuf support in fast-tokenizer
        # code paths; fall back to slow tokenizers when available.
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model,
            use_fast=False,
            local_files_only=local_files_only,
        )
    encoder = AutoModel.from_pretrained(
        pretrained_model,
        local_files_only=local_files_only,
    )
    encoder.eval()

    model_device = _resolve_torch_device(torch=torch, device=device)
    predictor = _CometPredictor(
        torch=torch,
        tokenizer=tokenizer,
        encoder=encoder,
        hparams=hparams,
        requires_reference=class_identifier == "regression_metric",
    )

    state_dict = _load_checkpoint_state_dict(torch=torch, checkpoint_path=checkpoint_path)
    load_result = predictor.load_state_dict(state_dict, strict=False)
    missing_critical = [
        key
        for key in load_result.missing_keys
        if _is_critical_checkpoint_key(key)
    ]
    if missing_critical:
        raise PrerequisiteError(
            "Failed to load COMET checkpoint weights. Missing critical keys: "
            + ", ".join(missing_critical[:10])
        )

    predictor.to(model_device)
    predictor.eval()

    max_positions = int(getattr(encoder.config, "max_position_embeddings", 512))
    max_length = max(8, max_positions - 4)
    return _CometRuntime(
        torch=torch,
        predictor=predictor,
        tokenizer=tokenizer,
        device=model_device,
        max_length=max_length,
    )


def _import_comet_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        import yaml
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer
    except ImportError as ex:
        raise pip_dependency_error(
            "COMET scorer",
            ["torch", "transformers", "huggingface_hub"],
        ) from ex

    return torch, AutoModel, AutoTokenizer, snapshot_download, yaml


def _resolve_checkpoint_and_hparams(
    model: str,
    model_storage_path: str | None,
    local_files_only: bool,
    snapshot_download: Callable[..., str],
    yaml: Any,
) -> tuple[Path, dict[str, Any]]:
    model_path = Path(model).expanduser()

    if model_path.is_file() and model_path.suffix == ".ckpt":
        checkpoint_path = model_path
        root = _checkpoint_root_from_checkpoint(checkpoint_path)
    elif model_path.is_dir():
        root = model_path
        checkpoint_path = _find_checkpoint_path(root)
    else:
        try:
            snapshot_path = snapshot_download(
                repo_id=model,
                cache_dir=model_storage_path,
                local_files_only=local_files_only,
            )
        except Exception as ex:
            raise PrerequisiteError(
                f"Unable to download COMET model '{model}'. "
                "Check network access or set local_files_only=True."
            ) from ex
        root = Path(snapshot_path)
        checkpoint_path = _find_checkpoint_path(root)

    hparams_path = root / "hparams.yaml"
    if not hparams_path.exists():
        raise PrerequisiteError(
            f"COMET model is missing hparams.yaml at: {hparams_path}"
        )

    with hparams_path.open("r", encoding="utf-8") as file:
        hparams = cast(dict[str, Any], yaml.safe_load(file) or {})
    return checkpoint_path, hparams


def _checkpoint_root_from_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _find_checkpoint_path(model_root: Path) -> Path:
    expected = model_root / "checkpoints" / "model.ckpt"
    if expected.exists():
        return expected

    checkpoint_candidates = sorted((model_root / "checkpoints").glob("*.ckpt"))
    if checkpoint_candidates:
        return checkpoint_candidates[0]

    raise PrerequisiteError(
        f"Unable to find COMET checkpoint under: {model_root / 'checkpoints'}"
    )


def _load_checkpoint_state_dict(torch: Any, checkpoint_path: Path) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        state_dict = cast(Mapping[str, Any], checkpoint["state_dict"])
    elif isinstance(checkpoint, Mapping):
        state_dict = cast(Mapping[str, Any], checkpoint)
    else:
        raise PrerequisiteError(
            f"Unsupported checkpoint format in {checkpoint_path}."
        )

    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = key
        for prefix in ("model.", "module."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]
        normalized[normalized_key] = value
    return normalized


def _resolve_torch_device(torch: Any, device: str) -> Any:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        return torch.device(device)
    except Exception as ex:
        raise PrerequisiteError(
            f"Invalid COMET device '{device}'. Use one of: auto, cpu, cuda, mps."
        ) from ex


@dataclass
class _CometRuntime:
    torch: Any
    predictor: "_CometPredictor"
    tokenizer: Any
    device: Any
    max_length: int

    def score(
        self,
        sources: list[str],
        candidates: list[str],
        references: list[str] | None,
    ) -> list[float]:
        if len(sources) != len(candidates):
            raise PrerequisiteError(
                "COMET scorer received mismatched source/candidate lengths."
            )
        if self.predictor.requires_reference and references is None:
            raise PrerequisiteError(
                "Selected COMET model requires references but none were provided."
            )
        if references is not None and len(references) != len(candidates):
            raise PrerequisiteError(
                "COMET scorer received mismatched reference/candidate lengths."
            )

        src_inputs = self._tokenize(sources)
        mt_inputs = self._tokenize(candidates)
        ref_inputs = self._tokenize(references) if references is not None else None

        with self.torch.inference_mode():
            scores = self.predictor.score_batch(
                src_inputs=src_inputs,
                mt_inputs=mt_inputs,
                ref_inputs=ref_inputs,
            )
            return cast(list[float], scores.detach().cpu().tolist())

    def _tokenize(self, texts: Sequence[str] | None) -> dict[str, Any] | None:
        if texts is None:
            return None
        tokenized = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {key: value.to(self.device) for key, value in tokenized.items()}


class _CometPredictor:  # pragma: no cover - exercised through scorer + mocked tests
    def __init__(
        self,
        *,
        torch: Any,
        tokenizer: Any,
        encoder: Any,
        hparams: dict[str, Any],
        requires_reference: bool,
    ) -> None:
        self.torch = torch
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.requires_reference = requires_reference

        hidden_size = int(getattr(self.encoder.config, "hidden_size", 0))
        if hidden_size <= 0:
            raise PrerequisiteError("Invalid COMET encoder hidden size.")

        self.pool = str(hparams.get("pool", "avg"))
        self.layer = _parse_layer(hparams.get("layer", "mix"))
        self.use_context = False

        layer_transformation = str(hparams.get("layer_transformation", "softmax"))
        if layer_transformation == "sparsemax_patch":
            layer_transformation = "softmax"
        layer_norm = bool(hparams.get("layer_norm", True))
        dropout = float(hparams.get("dropout", 0.1))

        num_layers = int(getattr(self.encoder.config, "num_hidden_layers", 0)) + 1
        self.layerwise_attention: _LayerwiseAttention | None = None
        if self.layer == "mix":
            self.layerwise_attention = _LayerwiseAttention(
                torch=torch,
                num_layers=num_layers,
                layer_norm=layer_norm,
                dropout=dropout,
                layer_transformation=layer_transformation,
            )

        hidden_sizes = [int(size) for size in hparams.get("hidden_sizes", [3072, 1024])]
        activation = str(hparams.get("activations", "Tanh"))
        final_activation = cast(str | None, hparams.get("final_activation", None))
        in_dim = hidden_size * (6 if requires_reference else 4)
        self.estimator = _FeedForward(
            torch=torch,
            in_dim=in_dim,
            hidden_sizes=hidden_sizes,
            activations=activation,
            final_activation=final_activation,
            dropout=dropout,
            out_dim=1,
        )

    def to(self, device: Any) -> "_CometPredictor":
        self.encoder = self.encoder.to(device)
        self.estimator = self.estimator.to(device)
        if self.layerwise_attention is not None:
            self.layerwise_attention = self.layerwise_attention.to(device)
        return self

    def eval(self) -> None:
        self.encoder.eval()
        self.estimator.eval()
        if self.layerwise_attention is not None:
            self.layerwise_attention.eval()

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for key, value in self.encoder.state_dict().items():
            state[f"encoder.{key}"] = value
        for key, value in self.estimator.state_dict().items():
            state[f"estimator.{key}"] = value
        if self.layerwise_attention is not None:
            for key, value in self.layerwise_attention.state_dict().items():
                state[f"layerwise_attention.{key}"] = value
        return state

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = False
    ) -> Any:
        encoder_state: dict[str, Any] = {}
        estimator_state: dict[str, Any] = {}
        layerwise_state: dict[str, Any] = {}
        unexpected_keys: list[str] = []

        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_key = key[len("encoder.") :]
                # COMET checkpoints store HuggingFace encoder params as encoder.model.*
                if encoder_key.startswith("model."):
                    encoder_key = encoder_key[len("model.") :]
                encoder_state[encoder_key] = value
            elif key.startswith("estimator."):
                estimator_key = key[len("estimator.") :]
                # COMET checkpoints use estimator.ff.* while our internal FFN
                # module expects keys rooted at the sequential layers.
                if estimator_key.startswith("ff."):
                    estimator_key = estimator_key[len("ff.") :]
                estimator_state[estimator_key] = value
            elif key.startswith("layerwise_attention."):
                layerwise_state[key[len("layerwise_attention.") :]] = value
            else:
                unexpected_keys.append(key)

        encoder_result = self.encoder.load_state_dict(encoder_state, strict=False)
        estimator_result = self.estimator.load_state_dict(estimator_state, strict=False)
        missing_keys = [f"encoder.{key}" for key in encoder_result.missing_keys] + [
            f"estimator.{key}" for key in estimator_result.missing_keys
        ]
        unexpected = [f"encoder.{key}" for key in encoder_result.unexpected_keys] + [
            f"estimator.{key}" for key in estimator_result.unexpected_keys
        ]

        if self.layerwise_attention is not None:
            layer_result = self.layerwise_attention.load_state_dict(
                layerwise_state, strict=False
            )
            missing_keys.extend(
                [f"layerwise_attention.{key}" for key in layer_result.missing_keys]
            )
            unexpected.extend(
                [f"layerwise_attention.{key}" for key in layer_result.unexpected_keys]
            )

        unexpected.extend(unexpected_keys)
        return _LoadStateResult(missing_keys=missing_keys, unexpected_keys=unexpected)

    def score_batch(
        self,
        *,
        src_inputs: dict[str, Any],
        mt_inputs: dict[str, Any],
        ref_inputs: dict[str, Any] | None,
    ) -> Any:
        src_emb = self._get_sentence_embedding(src_inputs)
        mt_emb = self._get_sentence_embedding(mt_inputs)

        if self.requires_reference:
            if ref_inputs is None:
                raise PrerequisiteError(
                    "Selected COMET model requires references but none were provided."
                )
            ref_emb = self._get_sentence_embedding(ref_inputs)

            diff_ref = self.torch.abs(mt_emb - ref_emb)
            diff_src = self.torch.abs(mt_emb - src_emb)
            prod_ref = mt_emb * ref_emb
            prod_src = mt_emb * src_emb
            features = self.torch.cat(
                (mt_emb, ref_emb, prod_ref, diff_ref, prod_src, diff_src),
                dim=1,
            )
        else:
            diff_src = self.torch.abs(mt_emb - src_emb)
            prod_src = mt_emb * src_emb
            features = self.torch.cat((mt_emb, src_emb, prod_src, diff_src), dim=1)

        return self.estimator(features).view(-1)

    def _get_sentence_embedding(self, model_inputs: dict[str, Any]) -> Any:
        attention_mask = model_inputs["attention_mask"]
        model_output = self.encoder(
            input_ids=model_inputs["input_ids"],
            attention_mask=attention_mask,
            token_type_ids=model_inputs.get("token_type_ids", None),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = model_output.hidden_states
        if hidden_states is None:
            raise PrerequisiteError(
                "COMET encoder did not provide hidden states."
            )

        if self.layerwise_attention is not None:
            embeddings = self.layerwise_attention(
                tensors=list(hidden_states),
                mask=attention_mask,
            )
        elif isinstance(self.layer, int) and 0 <= self.layer < len(hidden_states):
            embeddings = hidden_states[self.layer]
        else:
            raise PrerequisiteError(f"Invalid COMET layer value: {self.layer}")

        if self.pool == "default":
            pooled = getattr(model_output, "pooler_output", None)
            if pooled is None:
                pooled = embeddings[:, 0, :]
            return pooled
        elif self.pool == "max":
            return _max_pooling(self.torch, embeddings, attention_mask)
        elif self.pool == "avg":
            return _average_pooling(self.torch, embeddings, attention_mask)
        elif self.pool == "cls":
            return embeddings[:, 0, :]
        else:
            raise PrerequisiteError(f"Invalid COMET pooling mode: {self.pool}")


@dataclass
class _LoadStateResult:
    missing_keys: list[str]
    unexpected_keys: list[str]


class _FeedForward:  # pragma: no cover - behavior validated by checkpoint loading
    def __init__(
        self,
        *,
        torch: Any,
        in_dim: int,
        hidden_sizes: list[int],
        activations: str,
        final_activation: str | None,
        dropout: float,
        out_dim: int = 1,
    ) -> None:
        self.torch = torch
        modules: list[Any] = []
        modules.append(torch.nn.Linear(in_dim, hidden_sizes[0]))
        modules.append(self._build_activation(activations))
        modules.append(torch.nn.Dropout(dropout))
        for index in range(1, len(hidden_sizes)):
            modules.append(torch.nn.Linear(hidden_sizes[index - 1], hidden_sizes[index]))
            modules.append(self._build_activation(activations))
            modules.append(torch.nn.Dropout(dropout))
        modules.append(torch.nn.Linear(hidden_sizes[-1], int(out_dim)))
        if final_activation is not None:
            modules.append(self._build_activation(final_activation))

        self.ff = torch.nn.Sequential(*modules)

    def _build_activation(self, activation: str) -> Any:
        activation_name = activation.title()
        if hasattr(self.torch.nn, activation_name):
            return getattr(self.torch.nn, activation_name)()
        raise PrerequisiteError(f"Unsupported COMET activation: {activation}")

    def __call__(self, in_features: Any) -> Any:
        ff_dtypes = {parameter.dtype for parameter in self.ff.parameters()}
        if ff_dtypes == {self.torch.float16} and in_features.dtype != self.torch.float16:
            in_features = in_features.to(self.torch.float16)
        return self.ff(in_features)

    def to(self, device: Any) -> "_FeedForward":
        self.ff = self.ff.to(device)
        return self

    def eval(self) -> None:
        self.ff.eval()

    def state_dict(self) -> dict[str, Any]:
        return self.ff.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False) -> Any:
        return self.ff.load_state_dict(state_dict, strict=strict)


class _LayerwiseAttention:  # pragma: no cover - behavior validated by checkpoint loading
    def __init__(
        self,
        *,
        torch: Any,
        num_layers: int,
        layer_norm: bool = False,
        layer_weights: list[int] | None = None,
        dropout: float | None = None,
        layer_transformation: str = "softmax",
    ) -> None:
        self.torch = torch
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        self.dropout = dropout
        self.transform_fn = (
            _sparsemax if layer_transformation == "sparsemax" else self.torch.softmax
        )
        if layer_weights is None:
            layer_weights = [0.0] * num_layers
        if len(layer_weights) != num_layers:
            raise PrerequisiteError(
                f"Invalid layer_weights size ({len(layer_weights)}), expected {num_layers}."
            )

        self.scalar_parameters = self.torch.nn.ParameterList(
            [
                self.torch.nn.Parameter(
                    self.torch.FloatTensor([layer_weights[index]]),
                    requires_grad=True,
                )
                for index in range(num_layers)
            ]
        )
        self.gamma = self.torch.nn.Parameter(
            self.torch.FloatTensor([1.0]),
            requires_grad=True,
        )
        if self.dropout:
            self.dropout_mask = self.torch.zeros(len(self.scalar_parameters))
            self.dropout_fill = self.torch.empty(len(self.scalar_parameters)).fill_(-1e20)

    def to(self, device: Any) -> "_LayerwiseAttention":
        self.scalar_parameters = cast(Any, self.scalar_parameters.to(device))
        self.gamma = self.torch.nn.Parameter(
            self.gamma.to(device),
            requires_grad=True,
        )
        if self.dropout:
            self.dropout_mask = self.dropout_mask.to(device)
            self.dropout_fill = self.dropout_fill.to(device)
        return self

    def eval(self) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {"gamma": self.gamma}
        for index, parameter in enumerate(self.scalar_parameters):
            state[f"scalar_parameters.{index}"] = parameter
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False) -> Any:
        missing_keys: list[str] = []
        unexpected_keys = [
            key
            for key in state_dict.keys()
            if key != "gamma" and not key.startswith("scalar_parameters.")
        ]

        gamma = state_dict.get("gamma")
        if gamma is None:
            missing_keys.append("gamma")
        else:
            self.gamma.data.copy_(gamma)

        for index, parameter in enumerate(self.scalar_parameters):
            key = f"scalar_parameters.{index}"
            value = state_dict.get(key)
            if value is None:
                missing_keys.append(key)
            else:
                parameter.data.copy_(value)

        return _LoadStateResult(missing_keys=missing_keys, unexpected_keys=unexpected_keys)

    def __call__(self, tensors: list[Any], mask: Any | None = None) -> Any:
        if len(tensors) != self.num_layers:
            raise PrerequisiteError(
                f"LayerwiseAttention expected {self.num_layers} tensors, received {len(tensors)}."
            )
        weights = self.torch.cat([parameter for parameter in self.scalar_parameters])
        if self.dropout:
            weights = self.torch.where(
                self.dropout_mask.uniform_() > self.dropout,
                weights,
                self.dropout_fill,
            )
        normed_weights = self.transform_fn(weights, dim=0)
        normed_weights = self.torch.split(normed_weights, split_size_or_sections=1)

        if not self.layer_norm or mask is None:
            pieces = [weight * tensor for weight, tensor in zip(normed_weights, tensors)]
            return self.gamma * sum(pieces)

        mask_float = mask.float()
        broadcast_mask = mask_float.unsqueeze(-1)
        pieces = []
        for weight, tensor in zip(normed_weights, tensors):
            tensor_masked = tensor * broadcast_mask
            batch_size, _, input_dim = tensors[0].size()
            num_elements_not_masked = mask_float.sum(1) * input_dim
            mean = tensor_masked.view(batch_size, -1).sum(1)
            mean = (mean / num_elements_not_masked).view(batch_size, 1, 1)
            variance = (((tensor_masked - mean) * broadcast_mask) ** 2).view(
                batch_size, -1
            ).sum(1) / num_elements_not_masked
            normalized = (tensor - mean) / self.torch.sqrt(variance + 1e-12).view(
                batch_size, 1, 1
            )
            pieces.append(weight * normalized)
        return self.gamma * sum(pieces)


def _average_pooling(torch: Any, embeddings: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    pooled = (embeddings * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-12)
    return pooled / denom


def _max_pooling(torch: Any, embeddings: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).bool()
    masked = embeddings.masked_fill(~mask, float("-inf"))
    return masked.max(dim=1)[0]


def _parse_layer(layer: Any) -> str | int:
    if layer == "mix":
        return "mix"
    if isinstance(layer, int):
        return layer
    if isinstance(layer, str):
        try:
            return int(layer)
        except ValueError:
            return layer
    return layer


def _is_critical_checkpoint_key(key: str) -> bool:
    if key.startswith("encoder.pooler."):
        # COMET encoders are loaded without relying on pooler outputs.
        return False
    # Ignore non-trainable/buffer style keys (e.g. embeddings.position_ids,
    # layerwise_attention.dropout_mask/dropout_fill) and focus on trainable params.
    if key.endswith(".weight") or key.endswith(".bias"):
        return True
    return key in {"layerwise_attention.gamma"} or key.startswith(
        "layerwise_attention.scalar_parameters."
    )


def _sparsemax(z: Any, dim: int = -1) -> Any:
    """Sparsemax activation used by COMET layerwise attention."""
    import torch

    z_shifted = z - z.max(dim=dim, keepdim=True)[0]
    z_sorted = torch.sort(z_shifted, dim=dim, descending=True)[0]

    dim_size = z.size(dim)
    range_values = torch.arange(1, dim_size + 1, device=z.device, dtype=z.dtype)
    view_shape = [1] * z.dim()
    view_shape[dim] = dim_size
    range_values = range_values.view(view_shape)

    cumsum = torch.cumsum(z_sorted, dim=dim)
    support = (1 + range_values * z_sorted) > cumsum
    support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)

    tau = (torch.gather(cumsum, dim=dim, index=support_size - 1) - 1) / support_size
    return torch.clamp(z_shifted - tau, min=0.0)
