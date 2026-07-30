#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
OPENSEARCH_NS = "{http://a9.com/-/spec/opensearch/1.1/}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

CHATGPT_LAUNCH = pd.Timestamp("2022-11-30")
PRE_END = pd.Timestamp("2022-10-01")       # último mes usado en el ajuste
POST_START = pd.Timestamp("2023-01-01")    # inicio del periodo evaluado
MAX_PAGE_SIZE = 2000
MAX_RESULTS_PER_QUERY = 30000


# Grupos usados solo cuando se activa --include-disciplines.
# Algunos se solapan deliberadamente: "IA y ML" es una subserie de informática
# y estadística, útil para comparaciones, pero no debe sumarse a las demás.
def classify_primary_category(primary: str) -> set[str]:
    groups: set[str] = set()

    if primary.startswith("cs."):
        groups.add("Informática")
    if primary.startswith("stat."):
        groups.add("Estadística")
    if primary.startswith("math."):
        groups.add("Matemáticas")
    if primary.startswith("q-bio."):
        groups.add("Biología cuantitativa")
    if primary.startswith("q-fin."):
        groups.add("Finanzas cuantitativas")
    if primary.startswith("eess."):
        groups.add("Ingeniería y sistemas")
    if primary.startswith("econ."):
        groups.add("Economía")

    physics_prefixes = (
        "astro-ph.", "cond-mat.", "gr-qc", "hep-ex", "hep-lat", "hep-ph",
        "hep-th", "math-ph", "nlin.", "nucl-ex", "nucl-th", "physics.",
        "quant-ph",
    )
    if primary.startswith(physics_prefixes):
        groups.add("Física y astronomía")

    if primary in {"cs.AI", "cs.LG", "stat.ML"}:
        groups.add("IA y aprendizaje automático")

    return groups


@dataclass
class FittedModel:
    """Modelo de tendencia mensual y función de predicción."""

    name: str
    predict: Callable[[np.ndarray], np.ndarray]
    slope: float
    intercept: float
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None


class ArxivClient:
    """Cliente mínimo, con caché externa y rate limiting de la API de arXiv."""

    def __init__(self, delay_seconds: float = 3.0, retries: int = 5) -> None:
        self.delay_seconds = delay_seconds
        self.retries = retries
        self.last_request_at = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "NapkinNotes-arXiv-knowledge-acceleration/1.0 "
                    "(research exploratory analysis; contact: local-run-script)"
                )
            }
        )

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self.last_request_at
        remaining = self.delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def query(self, search_query: str, start: int = 0, max_results: int = 1) -> ET.Element:
        """Hace una consulta a la API y devuelve el XML Atom ya parseado."""
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            self._respect_rate_limit()
            try:
                response = self.session.get(API_URL, params=params, timeout=120)
                self.last_request_at = time.monotonic()
                response.raise_for_status()
                return ET.fromstring(response.content)
            except (requests.RequestException, ET.ParseError) as exc:
                last_error = exc
                wait = min(60, 2**attempt)
                print(
                    f"[aviso] Petición fallida (intento {attempt}/{self.retries}): {exc}. "
                    f"Reintentando en {wait} s...",
                    file=sys.stderr,
                )
                time.sleep(wait)

        raise RuntimeError(f"No se pudo completar la consulta a arXiv: {last_error}")

    @staticmethod
    def _interval_query(start: pd.Timestamp, end: pd.Timestamp) -> str:
        """Construye un rango inclusivo de fecha con precisión de minuto GMT."""
        # La documentación de arXiv usa intervalos inclusivos. Cerramos el
        # extremo final un minuto antes para no duplicar artículos en fronteras.
        inclusive_end = end - pd.Timedelta(minutes=1)
        start_text = start.strftime("%Y%m%d%H%M")
        end_text = inclusive_end.strftime("%Y%m%d%H%M")
        return f"submittedDate:[{start_text} TO {end_text}]"

    def count_interval(self, start: pd.Timestamp, end: pd.Timestamp) -> int:
        root = self.query(self._interval_query(start, end), max_results=1)
        value = root.findtext(f"{OPENSEARCH_NS}totalResults")
        if value is None:
            raise RuntimeError("La respuesta de arXiv no contiene totalResults.")
        return int(value)

    def fetch_entries_interval(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        known_total: int | None = None,
    ) -> list[dict]:
        """
        Recupera metadatos mínimos (fecha de publicación y categoría primaria).

        Si el intervalo supera el máximo de resultados soportado por la API,
        lo divide recursivamente en dos.
        """
        total = known_total if known_total is not None else self.count_interval(start, end)

        if total == 0:
            return []

        if total > MAX_RESULTS_PER_QUERY:
            midpoint = start + (end - start) / 2
            midpoint = pd.Timestamp(midpoint).floor("min")
            if midpoint <= start or midpoint >= end:
                raise RuntimeError(
                    f"Intervalo demasiado denso para dividirse: {start} a {end} ({total} resultados)."
                )
            left = self.fetch_entries_interval(start, midpoint)
            right = self.fetch_entries_interval(midpoint, end)
            return left + right

        records: list[dict] = []
        query = self._interval_query(start, end)
        for offset in range(0, total, MAX_PAGE_SIZE):
            page_size = min(MAX_PAGE_SIZE, total - offset)
            root = self.query(query, start=offset, max_results=page_size)

            entries = root.findall(f"{ATOM_NS}entry")
            if not entries:
                raise RuntimeError(
                    f"arXiv devolvió una página vacía antes de completar el intervalo "
                    f"{start:%Y-%m} (offset={offset}, total={total})."
                )

            for entry in entries:
                primary = entry.find(f"{ARXIV_NS}primary_category")
                published = entry.findtext(f"{ATOM_NS}published")
                if primary is None or published is None:
                    continue
                records.append(
                    {
                        "published": published,
                        "primary_category": primary.attrib.get("term", "unknown"),
                    }
                )

            print(
                f"    {start:%Y-%m}: {min(offset + page_size, total):,}/{total:,} metadatos",
                flush=True,
            )

        if len(records) != total:
            print(
                f"[aviso] El intervalo {start:%Y-%m} informó {total:,} resultados, "
                f"pero se guardaron {len(records):,} entradas con metadatos completos.",
                file=sys.stderr,
            )

        return records


def parse_month(text: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(f"{text}-01")
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Fecha inválida '{text}'. Usa el formato YYYY-MM, por ejemplo 2015-01."
        ) from exc


def month_starts(start: pd.Timestamp, end_exclusive: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start=start, end=end_exclusive - pd.offsets.MonthBegin(1), freq="MS"))


def next_month(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts + pd.offsets.MonthBegin(1)).normalize()


def cache_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_global_monthly_counts(
    client: ArxivClient,
    months: Iterable[pd.Timestamp],
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    """Obtiene una serie mensual global usando totalResults de la API."""
    rows: list[dict] = []
    cache_dir.mkdir(parents=True, exist_ok=True)

    for i, start in enumerate(months, start=1):
        end = next_month(start)
        cache_file = cache_dir / f"global_count_{start:%Y-%m}.json"

        if cache_file.exists() and not refresh:
            payload = load_json(cache_file)
            count = int(payload["count"])
            origin = "cache"
        else:
            print(f"[{i}] Consultando total mensual para {start:%Y-%m}...", flush=True)
            count = client.count_interval(start, end)
            cache_json(
                cache_file,
                {
                    "month": start.strftime("%Y-%m"),
                    "count": count,
                    "query": client._interval_query(start, end),
                    "downloaded_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                },
            )
            origin = "api"

        rows.append({"date": start, "actual": count, "source": origin})

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def get_entries_for_month(
    client: ArxivClient,
    start: pd.Timestamp,
    total: int,
    cache_dir: Path,
) -> list[dict]:
    """
    Descarga y almacena solo published + primary_category.

    La caché permite retomar el análisis sin volver a pedir los meses ya
    completados. Durante una descarga incompleta se usa un archivo .part.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    finished = cache_dir / f"entries_{start:%Y-%m}.json"
    partial = cache_dir / f"entries_{start:%Y-%m}.json.part"

    if finished.exists():
        return load_json(finished)["entries"]

    if partial.exists():
        print(f"[aviso] Se elimina una descarga parcial para {start:%Y-%m} y se reinicia.")
        partial.unlink()

    print(f"  Descargando metadatos para clasificación disciplinar: {start:%Y-%m}", flush=True)
    entries = client.fetch_entries_interval(start, next_month(start), known_total=total)
    partial.write_text(
        json.dumps({"month": start.strftime("%Y-%m"), "entries": entries}, ensure_ascii=False),
        encoding="utf-8",
    )
    partial.replace(finished)
    return entries


def build_discipline_counts(
    client: ArxivClient,
    global_counts: pd.DataFrame,
    entries_cache: Path,
) -> pd.DataFrame:
    """Construye conteos mensuales por grupo a partir de categorías primarias."""
    all_rows: list[dict] = []

    for _, row in global_counts.iterrows():
        month = pd.Timestamp(row["date"])
        entries = get_entries_for_month(client, month, int(row["actual"]), entries_cache)

        counts: dict[str, int] = {}
        for entry in entries:
            for group in classify_primary_category(entry["primary_category"]):
                counts[group] = counts.get(group, 0) + 1

        for group, count in counts.items():
            all_rows.append({"date": month, "group": group, "actual": count})

    if not all_rows:
        return pd.DataFrame(columns=["date", "group", "actual"])

    return pd.DataFrame(all_rows).sort_values(["group", "date"]).reset_index(drop=True)


def fit_linear(t: np.ndarray, y: np.ndarray) -> FittedModel:
    slope, intercept = np.polyfit(t, y, deg=1)

    def predict(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, intercept + slope * x)

    return FittedModel("linear", predict, float(slope), float(intercept))


def fit_exponential(t: np.ndarray, y: np.ndarray) -> FittedModel:
    if np.any(y <= 0):
        raise ValueError("El modelo exponencial requiere conteos positivos.")
    slope, intercept = np.polyfit(t, np.log(y), deg=1)

    def predict(x: np.ndarray) -> np.ndarray:
        return np.exp(intercept + slope * x)

    return FittedModel("exponential", predict, float(slope), float(intercept))


def bootstrap_band(
    t_train: np.ndarray,
    y_train: np.ndarray,
    t_pred: np.ndarray,
    model_name: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Banda empírica de parámetros mediante bootstrap de meses históricos.

    Es una banda exploratoria de incertidumbre del ajuste. No corrige
    autocorrelación ni estacionalidad y no debe interpretarse como evidencia
    causal.
    """
    if n_bootstrap <= 0:
        nan = np.full_like(t_pred, np.nan, dtype=float)
        return nan, nan

    predictions: list[np.ndarray] = []
    n = len(t_train)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        x_boot = t_train[idx]
        y_boot = y_train[idx]

        if np.unique(x_boot).size < 2:
            continue

        try:
            model = fit_linear(x_boot, y_boot) if model_name == "linear" else fit_exponential(x_boot, y_boot)
            predictions.append(model.predict(t_pred))
        except (ValueError, np.linalg.LinAlgError):
            continue

    if len(predictions) < max(50, n_bootstrap // 5):
        nan = np.full_like(t_pred, np.nan, dtype=float)
        return nan, nan

    matrix = np.asarray(predictions)
    return np.quantile(matrix, 0.025, axis=0), np.quantile(matrix, 0.975, axis=0)


def add_model_predictions(
    df: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, FittedModel]]:
    """Ajusta ambos modelos con el periodo pre-ChatGPT y añade predicciones."""
    result = df.copy().sort_values("date").reset_index(drop=True)
    result["t"] = np.arange(len(result), dtype=float)

    train = result[result["date"] <= PRE_END].copy()
    if len(train) < 24:
        raise ValueError("Se necesitan al menos 24 meses pre-ChatGPT para ajustar la tendencia.")

    t_train = train["t"].to_numpy(dtype=float)
    y_train = train["actual"].to_numpy(dtype=float)
    t_all = result["t"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    models = {
        "linear": fit_linear(t_train, y_train),
        "exponential": fit_exponential(t_train, y_train),
    }

    for name, model in models.items():
        prediction = model.predict(t_all)
        lower, upper = bootstrap_band(
            t_train=t_train,
            y_train=y_train,
            t_pred=t_all,
            model_name=name,
            n_bootstrap=bootstrap,
            rng=rng,
        )
        model.lower = lower
        model.upper = upper
        result[f"expected_{name}"] = prediction
        result[f"lower_{name}"] = lower
        result[f"upper_{name}"] = upper

    return result, models


def impact_table(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Calcula exceso absoluto, relativo y acumulado desde enero de 2023."""
    out = df.copy()
    expected = out[f"expected_{baseline}"]
    post = out["date"] >= POST_START

    out["excess"] = np.where(post, out["actual"] - expected, np.nan)
    out["excess_pct"] = np.where(post, 100 * out["excess"] / expected, np.nan)
    out["cumulative_excess"] = out["excess"].fillna(0).cumsum()
    out["baseline"] = baseline
    return out


def monthly_growth_percent(y: np.ndarray) -> float:
    """Pendiente logarítmica convertida a crecimiento mensual porcentual."""
    if len(y) < 2 or np.any(y <= 0):
        return float("nan")
    t = np.arange(len(y), dtype=float)
    slope, _ = np.polyfit(t, np.log(y), deg=1)
    return 100 * math.expm1(float(slope))


def model_summary(
    table: pd.DataFrame,
    models: dict[str, FittedModel],
    baseline: str,
) -> tuple[pd.DataFrame, dict]:
    """Produce métricas legibles para CSV, JSON y el informe."""
    post = table[table["date"] >= POST_START].copy()
    pre = table[table["date"] <= PRE_END].copy()

    rows: list[dict] = []
    for name, model in models.items():
        expected_col = f"expected_{name}"
        total_expected = float(post[expected_col].sum())
        total_actual = float(post["actual"].sum())
        excess = total_actual - total_expected
        rows.append(
            {
                "model": name,
                "pre_monthly_growth_pct": 100 * math.expm1(model.slope) if name == "exponential" else np.nan,
                "post_actual_total": total_actual,
                "post_expected_total": total_expected,
                "cumulative_excess": excess,
                "relative_excess_pct": 100 * excess / total_expected if total_expected else np.nan,
            }
        )

    selected = next(row for row in rows if row["model"] == baseline)
    pre_growth = monthly_growth_percent(pre["actual"].to_numpy(float))
    post_growth = monthly_growth_percent(post["actual"].to_numpy(float))

    summary = {
        "analysis": "Contrafactual temporal exploratorio con arXiv como proxy parcial",
        "chatgpt_public_launch": CHATGPT_LAUNCH.strftime("%Y-%m-%d"),
        "training_period": f"{table['date'].min():%Y-%m} a {PRE_END:%Y-%m}",
        "transition_period_excluded_from_impact": "2022-11 a 2022-12",
        "evaluation_period": f"{POST_START:%Y-%m} a {post['date'].max():%Y-%m}",
        "baseline_model": baseline,
        "actual_post_total": round(float(post["actual"].sum()), 3),
        "expected_post_total": round(float(post[f"expected_{baseline}"].sum()), 3),
        "cumulative_excess": round(float(selected["cumulative_excess"]), 3),
        "relative_excess_pct": round(float(selected["relative_excess_pct"]), 3),
        "pre_observed_monthly_growth_pct": round(pre_growth, 4),
        "post_observed_monthly_growth_pct": round(post_growth, 4),
        "growth_change_percentage_points": round(post_growth - pre_growth, 4),
        "n_months_total": int(len(table)),
        "n_months_post": int(len(post)),
    }
    return pd.DataFrame(rows), summary


def _finish_plot(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)


def plot_global_counterfactual(table: pd.DataFrame, model: str, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(table["date"], table["actual"], label="Envíos reales a arXiv", linewidth=2)
    ax.plot(
        table["date"],
        table[f"expected_{model}"],
        linestyle="--",
        label=f"Contrafactual {model}",
        linewidth=2,
    )

    lower = table[f"lower_{model}"].to_numpy(float)
    upper = table[f"upper_{model}"].to_numpy(float)
    if np.isfinite(lower).any():
        ax.fill_between(table["date"], lower, upper, alpha=0.15, label="Banda bootstrap 95 %")

    ax.axvspan(pd.Timestamp("2022-11-01"), pd.Timestamp("2022-12-31"), alpha=0.12, label="Transición excluida")
    ax.axvline(CHATGPT_LAUNCH, linestyle=":", linewidth=2, label="Lanzamiento público de ChatGPT")

    _finish_plot(
        ax,
        f"arXiv: actividad observada frente al contrafactual {model}",
        "Nuevos envíos mensuales",
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outdir / f"01_contrafactual_global_{model}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_excess(impact: pd.DataFrame, model: str, outdir: Path) -> None:
    post = impact[impact["date"] >= POST_START].copy()
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.bar(post["date"], post["excess"], width=22, label="Real − esperado")
    ax.axhline(0, linewidth=1.2)
    _finish_plot(
        ax,
        f"Exceso mensual de envíos respecto al contrafactual {model}",
        "Envíos adicionales por mes",
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outdir / f"02_exceso_mensual_{model}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_excess(impact: pd.DataFrame, model: str, outdir: Path) -> None:
    post = impact[impact["date"] >= POST_START].copy()
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.plot(post["date"], post["cumulative_excess"], linewidth=2.4, label="Exceso acumulado")
    ax.axhline(0, linewidth=1.2)
    _finish_plot(
        ax,
        f"Exceso acumulado desde enero de 2023 ({model})",
        "Envíos acumulados: real − esperado",
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outdir / f"03_exceso_acumulado_{model}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_expected(impact: pd.DataFrame, model: str, outdir: Path) -> None:
    post = impact[impact["date"] >= POST_START].copy()
    expected = post[f"expected_{model}"].to_numpy(float)
    actual = post["actual"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    ax.scatter(expected, actual, label="Meses posteriores a enero de 2023")
    low = float(min(expected.min(), actual.min()))
    high = float(max(expected.max(), actual.max()))
    ax.plot([low, high], [low, high], linestyle="--", label="Sin desviación")
    ax.set_title(f"Observado frente a esperado ({model})")
    ax.set_xlabel("Envíos mensuales esperados")
    ax.set_ylabel("Envíos mensuales observados")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / f"04_observado_vs_esperado_{model}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_annual_comparison(table: pd.DataFrame, model: str, outdir: Path) -> None:
    yearly = table.copy()
    yearly["year"] = yearly["date"].dt.year
    annual = yearly.groupby("year", as_index=False).agg(
        actual=("actual", "sum"),
        expected=(f"expected_{model}", "sum"),
    )

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.plot(annual["year"], annual["actual"], marker="o", label="Real")
    ax.plot(annual["year"], annual["expected"], marker="o", linestyle="--", label=f"Esperado ({model})")
    ax.axvline(2022.92, linestyle=":", linewidth=2, label="ChatGPT")
    ax.set_title("Envíos anuales: observado frente a tendencia extrapolada")
    ax.set_xlabel("Año")
    ax.set_ylabel("Nuevos envíos")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / f"05_comparacion_anual_{model}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyse_disciplines(
    discipline_counts: pd.DataFrame,
    baseline: str,
    bootstrap: int,
    seed: int,
    outdir: Path,
) -> pd.DataFrame:
    """Ajusta y compara el exceso relativo acumulado por disciplina."""
    if discipline_counts.empty:
        return pd.DataFrame()

    summaries: list[dict] = []
    global_dates = pd.date_range(
        discipline_counts["date"].min(), discipline_counts["date"].max(), freq="MS"
    )

    for i, group in enumerate(sorted(discipline_counts["group"].unique())):
        group_df = discipline_counts[discipline_counts["group"] == group][["date", "actual"]].copy()
        group_df = (
            pd.DataFrame({"date": global_dates})
            .merge(group_df, on="date", how="left")
            .fillna({"actual": 0})
        )

        try:
            fitted, models = add_model_predictions(group_df, bootstrap=bootstrap, seed=seed + i + 1)
        except ValueError:
            continue

        impact = impact_table(fitted, baseline)
        _, summary = model_summary(impact, models, baseline)
        summary["group"] = group
        summaries.append(summary)

        post = impact[impact["date"] >= POST_START].copy()
        fig, ax = plt.subplots(figsize=(12.7, 5.7))
        ax.plot(fitted["date"], fitted["actual"], label=f"Real: {group}", linewidth=2)
        ax.plot(
            fitted["date"],
            fitted[f"expected_{baseline}"],
            linestyle="--",
            label=f"Contrafactual {baseline}",
            linewidth=2,
        )
        ax.axvspan(pd.Timestamp("2022-11-01"), pd.Timestamp("2022-12-31"), alpha=0.12)
        ax.axvline(CHATGPT_LAUNCH, linestyle=":", linewidth=2, label="ChatGPT")
        _finish_plot(ax, f"{group}: observado frente a contrafactual", "Nuevos envíos mensuales")
        fig.autofmt_xdate()
        fig.tight_layout()
        safe_name = (
            group.lower()
            .replace(" ", "_")
            .replace("í", "i")
            .replace("á", "a")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace("ñ", "n")
        )
        fig.savefig(outdir / f"disciplina_{safe_name}_{baseline}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    output = pd.DataFrame(summaries).sort_values("relative_excess_pct", ascending=False)
    if output.empty:
        return output

    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.65 * len(output))))
    ax.barh(output["group"], output["relative_excess_pct"], label="Exceso relativo acumulado")
    ax.axvline(0, linewidth=1.2)
    ax.set_title(f"Comparación del exceso relativo acumulado por grupo ({baseline})")
    ax.set_xlabel("Exceso relativo acumulado (%)")
    ax.set_ylabel("")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / f"06_comparacion_disciplinar_{baseline}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output


def write_markdown_report(
    outdir: Path,
    summary: dict,
    model_table: pd.DataFrame,
    baseline: str,
    discipline_summary: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# Estimación exploratoria: arXiv y aceleración posterior a ChatGPT",
        "",
        "## Resultado principal",
        "",
        f"- Modelo contrafactual de referencia: **{baseline}**.",
        f"- Periodo de ajuste: **{summary['training_period']}**.",
        f"- Periodo evaluado: **{summary['evaluation_period']}**.",
        f"- Envíos observados desde enero de 2023: **{summary['actual_post_total']:,.0f}**.",
        f"- Envíos esperados según el contrafactual: **{summary['expected_post_total']:,.0f}**.",
        f"- Diferencia acumulada: **{summary['cumulative_excess']:,.0f}** envíos.",
        f"- Diferencia relativa: **{summary['relative_excess_pct']:.2f} %**.",
        "",
        "## Cambio de ritmo observado",
        "",
        f"- Crecimiento mensual ajustado antes de ChatGPT: **{summary['pre_observed_monthly_growth_pct']:.3f} %**.",
        f"- Crecimiento mensual ajustado desde enero de 2023: **{summary['post_observed_monthly_growth_pct']:.3f} %**.",
        f"- Diferencia: **{summary['growth_change_percentage_points']:.3f} puntos porcentuales mensuales**.",
        "",
        "## Interpretación correcta",
        "",
        "La diferencia cuantifica una desviación respecto a una tendencia histórica extrapolada. "
        "No identifica causalidad: también pueden contribuir cambios en el tamaño de la comunidad, "
        "presiones de publicación, nuevas políticas editoriales, cambios disciplinarios, financiación, "
        "estacionalidad y otros factores.",
        "",
        "arXiv se usa como un proxy parcial de producción científica abierta. El número de preprints "
        "no equivale de forma directa a cantidad, calidad o solidez del conocimiento generado.",
        "",
        "## Comparación de modelos",
        "",
    ]

    report_models = model_table.copy()
    report_models["pre_monthly_growth_pct"] = report_models["pre_monthly_growth_pct"].map(
        lambda x: "n/a" if pd.isna(x) else f"{x:.3f} %"
    )
    report_models["post_actual_total"] = report_models["post_actual_total"].map(lambda x: f"{x:,.0f}")
    report_models["post_expected_total"] = report_models["post_expected_total"].map(lambda x: f"{x:,.0f}")
    report_models["cumulative_excess"] = report_models["cumulative_excess"].map(lambda x: f"{x:,.0f}")
    report_models["relative_excess_pct"] = report_models["relative_excess_pct"].map(lambda x: f"{x:.2f} %")
    lines.append(report_models.to_markdown(index=False))

    if discipline_summary is not None and not discipline_summary.empty:
        lines.extend(
            [
                "",
                "## Grupos disciplinares",
                "",
                "Los grupos pueden solaparse: por ejemplo, IA y aprendizaje automático es una subserie "
                "dentro de informática y estadística. No deben sumarse entre sí.",
                "",
            ]
        )
        compact = discipline_summary[
            ["group", "cumulative_excess", "relative_excess_pct", "actual_post_total", "expected_post_total"]
        ].copy()
        compact["cumulative_excess"] = compact["cumulative_excess"].map(lambda x: f"{x:,.0f}")
        compact["relative_excess_pct"] = compact["relative_excess_pct"].map(lambda x: f"{x:.2f} %")
        compact["actual_post_total"] = compact["actual_post_total"].map(lambda x: f"{x:,.0f}")
        compact["expected_post_total"] = compact["expected_post_total"].map(lambda x: f"{x:,.0f}")
        lines.append(compact.to_markdown(index=False))

    (outdir / "informe_resumen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_matplotlib() -> None:
    # Fuente genérica para que el script funcione igual en Linux, macOS y Windows.
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contrafactual de envíos a arXiv antes y después de ChatGPT."
    )
    parser.add_argument("--start", default="2015-01", type=parse_month, help="Primer mes, YYYY-MM.")
    parser.add_argument(
        "--end",
        default=None,
        type=parse_month,
        help="Mes final EXCLUSIVO, YYYY-MM. Por defecto: inicio del mes actual.",
    )
    parser.add_argument(
        "--outdir",
        default="arxiv_ia_resultados",
        type=Path,
        help="Carpeta donde se guardarán datos, caché y gráficos.",
    )
    parser.add_argument(
        "--baseline",
        choices=("linear", "exponential"),
        default="exponential",
        help="Modelo de referencia para medir el exceso posterior.",
    )
    parser.add_argument(
        "--include-disciplines",
        action="store_true",
        help="Descarga metadatos para análisis por grupos científicos; tarda mucho más.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignora la caché de conteos mensuales y vuelve a consultar la API.",
    )
    parser.add_argument(
        "--api-delay",
        type=float,
        default=3.0,
        help="Segundos de pausa entre consultas a la API de arXiv (por defecto: 3).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=700,
        help="Réplicas bootstrap para bandas exploratorias de incertidumbre.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria reproducible.")
    args = parser.parse_args()

    configure_matplotlib()

    current_month = pd.Timestamp.today().replace(day=1).normalize()
    end_exclusive = args.end if args.end is not None else current_month
    start = args.start

    if start >= end_exclusive:
        raise SystemExit("--start debe ser anterior a --end.")
    if end_exclusive <= POST_START:
        raise SystemExit("El periodo debe llegar al menos hasta enero de 2023.")
    if start > pd.Timestamp("2020-01-01"):
        print(
            "[aviso] Con un periodo histórico muy corto el contrafactual será menos estable.",
            file=sys.stderr,
        )

    months = month_starts(start, end_exclusive)
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    counts_cache = outdir / "cache" / "monthly_counts"
    entries_cache = outdir / "cache" / "entries"

    print("=" * 78)
    print("Análisis exploratorio: arXiv, tendencia previa y desviación posterior")
    print(f"Periodo solicitado: {start:%Y-%m} a {end_exclusive - pd.offsets.MonthBegin(1):%Y-%m}")
    print(f"Modelo de referencia: {args.baseline}")
    print(f"Resultados: {outdir.resolve()}")
    print("=" * 78)

    client = ArxivClient(delay_seconds=max(args.api_delay, 0.0))
    global_counts = get_global_monthly_counts(client, months, counts_cache, refresh=args.refresh)
    global_counts.to_csv(outdir / "conteos_mensuales_globales.csv", index=False)

    fitted, models = add_model_predictions(
        global_counts[["date", "actual"]],
        bootstrap=max(args.bootstrap, 0),
        seed=args.seed,
    )
    fitted.to_csv(outdir / "serie_global_con_contrafactuales.csv", index=False)

    impact = impact_table(fitted, args.baseline)
    impact.to_csv(outdir / f"impacto_mensual_{args.baseline}.csv", index=False)

    model_table, summary = model_summary(impact, models, args.baseline)
    model_table.to_csv(outdir / "comparacion_modelos.csv", index=False)
    (outdir / "resumen_metricas.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_global_counterfactual(fitted, "linear", outdir)
    plot_global_counterfactual(fitted, "exponential", outdir)
    plot_monthly_excess(impact, args.baseline, outdir)
    plot_cumulative_excess(impact, args.baseline, outdir)
    plot_actual_vs_expected(impact, args.baseline, outdir)
    plot_annual_comparison(fitted, args.baseline, outdir)

    discipline_summary: pd.DataFrame | None = None
    if args.include_disciplines:
        print("\nClasificando campos mediante categoría primaria. Este paso requiere más descargas.")
        disciplines = build_discipline_counts(client, global_counts, entries_cache)
        disciplines.to_csv(outdir / "conteos_mensuales_por_disciplina.csv", index=False)
        discipline_summary = analyse_disciplines(
            disciplines,
            baseline=args.baseline,
            bootstrap=max(args.bootstrap // 2, 100),
            seed=args.seed,
            outdir=outdir,
        )
        if not discipline_summary.empty:
            discipline_summary.to_csv(outdir / "impacto_por_disciplina.csv", index=False)

    write_markdown_report(
        outdir=outdir,
        summary=summary,
        model_table=model_table,
        baseline=args.baseline,
        discipline_summary=discipline_summary,
    )

    print("\n" + "=" * 78)
    print("Listo.")
    print(f"Exceso acumulado ({args.baseline}): {summary['cumulative_excess']:,.0f} envíos")
    print(f"Exceso relativo: {summary['relative_excess_pct']:.2f} %")
    print(f"Consulta informe_resumen.md y los PNG en: {outdir.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()
