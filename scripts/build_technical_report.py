from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
ARTIFACT_PATH = REPORT_DIR / "technical_report_artifact.json"
TITLE = "Когда DoRA помогает? Контролируемое исследование на удержанных сидах"


def source(
    identifier: str,
    label: str,
    path: str,
    description: str,
    metric_definitions: dict[str, str],
    *,
    sql: str | None = None,
    tables_used: list[str] | None = None,
) -> dict[str, object]:
    query: dict[str, object] = {
        "description": description,
        "engine": "SQLite",
        "language": "sql" if sql else "python",
        "tables_used": tables_used or [path],
        "metric_definitions": metric_definitions,
    }
    if sql:
        query["sql"] = sql
    return {
        "id": identifier,
        "label": label,
        "path": path,
        "query": query,
    }


def main() -> None:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    comparisons = pd.read_csv(ROOT / "results" / "extension_paired_comparisons.csv")
    accuracy = pd.read_csv(ROOT / "results" / "extension_accuracy_summary.csv")
    data_regime = pd.read_csv(
        ROOT / "results" / "data_sweep_mlp" / "data_sweep_accuracy_ci.csv"
    )
    data_comparisons = pd.read_csv(
        ROOT / "results" / "data_sweep_mlp" / "data_sweep_paired_comparisons.csv"
    )
    synthetic = pd.read_csv(
        ROOT
        / "results"
        / "synthetic_optimization"
        / "synthetic_optimization_verified_summary.csv"
    )

    primary = comparisons[
        (comparisons["architecture"] == "mlp")
        & (comparisons["scenario"] == "mixed")
        & (comparisons["comparator"] == "lora")
    ].iloc[0]
    replication = comparisons[
        (comparisons["architecture"] == "cnn")
        & (comparisons["scenario"] == "mixed")
        & (comparisons["comparator"] == "lora")
    ].iloc[0]
    lo_ra_plus = comparisons[
        (comparisons["architecture"] == "mlp")
        & (comparisons["scenario"] == "mixed")
        & (comparisons["comparator"] == "lora_plus")
    ].iloc[0]
    matched = comparisons[
        (comparisons["architecture"] == "mlp")
        & (comparisons["scenario"] == "mixed")
        & (comparisons["comparator"] == "lora_matched")
    ].iloc[0]
    budgeted = comparisons[
        (comparisons["architecture"] == "mlp")
        & (comparisons["scenario"] == "mixed")
        & (comparisons["comparison"] == "dora_budgeted_vs_lora")
    ].iloc[0]
    gamma08 = synthetic[synthetic["magnitude_strength"] == 0.8].iloc[0]

    data_regime_sql = """SELECT
  adaptation_examples,
  method,
  method_id,
  accuracy_mean_pct,
  accuracy_ci95_low_pct,
  accuracy_ci95_high_pct,
  accuracy_std_pct,
  trainable_parameters,
  n_seeds
FROM data_regime_accuracy
WHERE method_id IN ('dora', 'lora', 'lora_plus')
ORDER BY method_id, adaptation_examples"""
    confirmatory_sql = """SELECT
  architecture,
  scenario,
  mean_delta_pp,
  ci95_low_pp,
  ci95_high_pp,
  paired_effect_dz,
  paired_t_p,
  paired_t_p_holm,
  wins,
  ties,
  losses,
  n_pairs
FROM paired_comparisons
WHERE comparison = 'dora_vs_lora'
ORDER BY architecture, scenario"""
    with sqlite3.connect(":memory:") as connection:
        data_regime.to_sql("data_regime_accuracy", connection, index=False)
        comparisons.to_sql("paired_comparisons", connection, index=False)
        data_chart = pd.read_sql_query(data_regime_sql, connection)
        confirmatory_table = pd.read_sql_query(confirmatory_sql, connection)
    confirmatory_table["architecture"] = confirmatory_table["architecture"].str.upper()
    confirmatory_table["scenario"] = confirmatory_table["scenario"].map(
        {
            "contrast": "Контраст",
            "rotation": "Поворот",
            "mixed": "Смешанный",
        }
    )
    accuracy_display = accuracy.copy()
    accuracy_display["architecture"] = accuracy_display["architecture"].str.upper()
    accuracy_display["scenario"] = accuracy_display["scenario"].map(
        {
            "contrast": "Контраст",
            "rotation": "Поворот",
            "mixed": "Смешанный",
        }
    )
    accuracy_display["method"] = accuracy_display["method"].replace(
        {
            "Frozen": "Замороженная модель",
            "Full fine-tuning": "Полное дообучение",
            "Magnitude-only": "Только величина",
            "DoRA (LoRA-budget ceiling)": "DoRA (в пределах бюджета параметров LoRA)",
            "LoRA (DoRA-matched budget)": "LoRA (с бюджетом параметров DoRA)",
        }
    )

    confirmatory_source = source(
        "confirmatory_source",
        "Парные сравнения на удержанных сидах",
        "results/extension_paired_comparisons.csv",
        "Парные тестовые метрики DoRA минус базовый вариант, сформированные analyze_extension.py.",
        {
            "mean_delta_pp": "Средняя внутрисидовая разность тестовой точности DoRA и базового варианта, в процентных пунктах.",
            "ci95": "Двусторонний 95%-й t-интервал Стьюдента для парных разностей по сидам.",
            "paired_t_p_holm": "p-значение парного t-критерия с поправкой Холма в пределах заявленной семьи сравнений.",
        },
        sql=confirmatory_sql,
        tables_used=["paired_comparisons"],
    )
    accuracy_source = source(
            "accuracy_source",
            "Сводки точности на удержанных сидах",
            "results/extension_accuracy_summary.csv",
            "Сводки точности MLP и CNN по методам, сформированные из исходных строк удержанных сидов.",
            {
                "accuracy_mean_pct": "Средняя арифметическая точность на целевой тестовой выборке по запускам адаптации на удержанных сидах, в процентах.",
                "trainable_parameters": "Число параметров модели, обновляемых при целевой адаптации.",
            },
        )
    data_regime_source = source(
            "data_regime_source",
            "Вложенный перебор объёмов целевых данных",
            "results/data_sweep_mlp/data_sweep_accuracy_ci.csv",
            "Перебор объёмов целевых данных при смешанном сдвиге с зафиксированными конфигурациями MLP и вложенными сбалансированными подвыборками.",
            {
                "accuracy_mean_pct": "Средняя точность на целевой тестовой выборке при смешанном сдвиге по 20 парным сидам, в процентах.",
                "adaptation_examples": "Общее число сбалансированных целевых обучающих примеров по десяти классам цифр.",
            },
            sql=data_regime_sql,
            tables_used=["data_regime_accuracy"],
        )
    synthetic_source = source(
            "synthetic_source",
            "Синтетическая диагностика оптимизации",
            "results/synthetic_optimization/synthetic_optimization_verified_summary.csv",
            "Сводки оптимизации DoRA и оракулов по задачам, сформированные analyze_robustness.py.",
            {
                "relative_weight_error": "Норма Фробениуса остатка весов, делённая на целевую норму Фробениуса.",
                "convergence_rate": "Доля заявленных инициализаций адаптера с относительной ошибкой ниже 1e-3, усреднённая по задачам.",
            },
        )
    sources = [confirmatory_source, accuracy_source, data_regime_source, synthetic_source]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": "Технический отчёт проекта DoRA для Летней школы AIRI 2026.",
        "generatedAt": generated_at,
        "sources": sources,
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
            {
                "id": "technical-summary",
                "type": "markdown",
                "body": (
                    "## Техническое резюме\n\n"
                    "DoRA даёт положительные точечные оценки на удержанных сидах для намеренно сложного смешанного сдвига предметной области: "
                    f"**{primary.mean_delta_pp:+.2f} п.п.** относительно LoRA для MLP "
                    f"(95%-й ДИ [{primary.ci95_low_pp:+.2f}, {primary.ci95_high_pp:+.2f}]) и "
                    f"**{replication.mean_delta_pp:+.2f} п.п.** для CNN "
                    f"([{replication.ci95_low_pp:+.2f}, {replication.ci95_high_pp:+.2f}]). "
                    f"p-значения парного t-критерия с поправкой Холма равны {primary.paired_t_p_holm:.6f} и "
                    f"{replication.paired_t_p_holm:.6f}. Один и тот же знак наблюдается на двух фиксированных архитектурах "
                    "и при четырёх объёмах целевых данных, но это внутреннее свидетельство условно на одном чекпоинте "
                    "на архитектуру и не является статистическим подтверждением на уровне семьи сравнений или внешней "
                    "репликацией. Чистый контраст остаётся контрпримером, где адаптация только величины эффективнее."
                ),
            },
            {
                "id": "key-findings",
                "type": "markdown",
                "body": (
                    "## Оценки для смешанного сдвига положительны и против усиленных базовых вариантов, но множественность сравнений важна\n\n"
                    f"В сравнении MLP по зафиксированному протоколу DoRA превосходит LoRA+ на **{lo_ra_plus.mean_delta_pp:+.2f} п.п.** "
                    f"и почти согласованное по числу параметров распределение LoRA на **{matched.mean_delta_pp:+.2f} п.п.** "
                    f"Их p-значения парного t-критерия с поправкой Холма равны {lo_ra_plus.paired_t_p_holm:.6f} и "
                    f"{matched.paired_t_p_holm:.6f}; значение для LoRA+ выше 0.05. Фиксированное отношение LoRA+ "
                    "`B/A=16` не настраивалось отдельно для каждого сдвига. Описательная оценка DoRA в рамках бюджета "
                    f"параметров LoRA составляет **{budgeted.mean_delta_pp:+.2f} п.п.** (95%-й ДИ "
                    f"[{budgeted.ci95_low_pp:+.2f}, {budgeted.ci95_high_pp:+.2f}], исходное p={budgeted.paired_t_p:.6f}, "
                    f"p с поправкой Холма во вторичной семье сравнений={budgeted.paired_t_p_holm:.6f}) и не используется "
                    "для спасения основного утверждения."
                ),
                "sourceId": "confirmatory_source",
            },
            {
                "id": "confirmatory-table-block",
                "type": "table",
                "tableId": "confirmatory-table",
            },
            {
                "id": "data-regime-finding",
                "type": "markdown",
                "body": (
                    "## Эффект не привязан к одному размеру адаптационной выборки\n\n"
                    "После фиксации скоростей обучения и распределений ранга по валидационному этапу с 400 примерами "
                    "DoRA остаётся выше обычной LoRA при 50, 100, 200 и 400 целевых примерах. Интервалы достаточно широки, "
                    "поэтому отдельные сравнения с LoRA остаются неопределёнными, тогда как больший разрыв с LoRA+ "
                    "сохраняется после поправки Холма при нескольких объёмах данных. Вложенные сбалансированные по классам "
                    "подвыборки отделяют изменение объёма данных от замены примеров. Это вспомогательный анализ чувствительности "
                    "на том же чекпоинте, а не независимая репликация."
                ),
                "source": data_regime_source,
            },
            {"id": "data-regime-chart-block", "type": "chart", "chartId": "data-regime-chart"},
            {
                "id": "scope-definitions",
                "type": "markdown",
                "body": (
                    "## Область применимости, данные и определения метрик\n\n"
                    "В эмпирическом бенчмарке используется набор `sklearn Digits` из 1 797 изображений с одним "
                    "фиксированным стратифицированным разбиением на исходную, валидационную и тестовую выборки. "
                    "Основная MLP имеет архитектуру `64→128→64→10`; CNN — вторая проверяемая архитектура — состоит "
                    "из двух свёрточных и двух линейных слоёв. Целевые сдвиги: снижение контраста, поворот на 25° и "
                    "сочетание поворота на 18° со снижением контраста и гауссовским шумом. Основная метрика — точность "
                    "на целевой тестовой выборке. Во всех статистических сравнениях пары совпадают по целевой подвыборке, "
                    "реализации искажения, сиду оптимизатора и тестовой выборке. Это контролируемое исследование малых "
                    "моделей, а не воспроизведение в масштабе трансформеров."
                ),
            },
            {
                "id": "experimental-design",
                "type": "markdown",
                "body": (
                    "## Схема эксперимента: сначала валидация\n\n"
                    "По пилотным сидам 11/22/33/44/55 выбираются скорости обучения, а для MLP — одно из трёх почти "
                    "согласованных по числу параметров распределений ранга. Только после фиксации всех вариантов для "
                    "каждой пары «метод — сдвиг» на целевой тестовой выборке оцениваются удержанные сиды адаптации "
                    "101–120 (MLP) и 201–210 (CNN). Эти сиды измеряют изменчивость подвыборок, искажений и оптимизации "
                    "при одном фиксированном предобученном чекпоинте. Базовые варианты: замороженная модель, адаптация "
                    "только величины, LoRA, LoRA+, DoRA, полное дообучение, LoRA с бюджетом параметров DoRA и DoRA с "
                    "верхней границей бюджета параметров LoRA. Для основных сравнений приводятся средняя парная разность, "
                    "95%-й t-интервал, парный размер эффекта, парные t-критерий и критерий Уилкоксона, а также поправка "
                    "Холма. Поправки вычисляются отдельно для основной семьи сравнений MLP (`m=9`), семьи CNN (`m=6`) "
                    "и вторичной описательной семьи MLP (`m=3`)."
                ),
            },
            {
                "id": "synthetic-finding",
                "type": "markdown",
                "body": (
                    "## Выразительной способности достаточно; при экстремальном изменении величины оптимизация несовершенна\n\n"
                    f"При интенсивности изменения величины 0.8 точный SVD-оракул LoRA ранга 4 сохраняет среднюю "
                    f"относительную ошибку **{gamma08.lora_oracle_error_mean:.3f}**, тогда как допустимая конструкция "
                    f"DoRA ранга 4 остаётся вблизи численного нуля. Стандартное обучение DoRA с инициализацией, не "
                    f"меняющей исходную модель (no-op), достигает критерия `<1e-3` в **{100 * gamma08.convergence_rate_mean:.0f}%** "
                    "заявленных запусков. Неудачные инициализации сохранены в анализе: геометрия представима, но сильное "
                    "изменение величины всё же может создавать проблему оптимизации."
                ),
                "sourceId": "synthetic_source",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## Ограничения, неопределённость и устойчивость\n\n"
                    "В исследовании используются небольшой набор изображений и один фиксированный предобученный экземпляр "
                    "каждой архитектуры. Неопределённость по сидам охватывает формирование малых выборок и оптимизацию "
                    "адаптации, но не изменчивость предобучения. То же целевое разбиение использовалось на предыдущем "
                    "разведочном этапе, поэтому это внутреннее подтверждение по зафиксированному протоколу, а не полностью "
                    "независимая внешняя репликация на ранее не использованных данных. Синтетический генератор по построению "
                    "относится к семейству DoRA, поэтому его следует рассматривать только как диагностику механизма. "
                    "Отрицательные результаты существенны: согласованная по числу параметров LoRA не отличается от DoRA "
                    "при повороте для MLP, а в контрольном случае чистого контраста выигрывает адаптация только величины."
                ),
            },
            {
                "id": "next-steps",
                "type": "markdown",
                "body": (
                    "## Рекомендуемый следующий шаг\n\n"
                    "Следующий эксперимент, полезный для принятия решений, — внешняя репликация с официальным стеком PEFT "
                    "на небольшом предобученном трансформере. Следует сравнить LoRA, LoRA+, rsLoRA и DoRA при одинаковых "
                    "целевых модулях и настройке только по валидационным данным. Нужно варьировать сиды или чекпоинты "
                    "предобучения и заранее объявить одну основную задачу и метрику, чтобы отделить обобщаемость между "
                    "архитектурами от очередного разведочного перебора бенчмарков."
                ),
            },
            {
                "id": "further-questions",
                "type": "markdown",
                "body": (
                    "## Открытые вопросы\n\n"
                    "- Сохраняется ли эффект при смешанном сдвиге после замены базового чекпоинта?\n"
                    "- Чем обусловлено преимущество в реальных сетях: выразительной способностью, регуляризацией или обусловленностью оптимизации?\n"
                    "- Может ли выбранное по валидации правило маршрутизации между адаптацией только величины, LoRA и DoRA превзойти универсальный выбор одного адаптера?"
                ),
            },
        ],
        "charts": [
            {
                "id": "data-regime-chart",
                "title": "Точность при смешанном сдвиге для разных объёмов целевых данных",
                "description": "Средняя тестовая точность по 20 парным сидам; конфигурации фиксированы для всех объёмов данных.",
                "dataset": "data_regime",
                "type": "line",
                "encodings": {
                    "x": {"field": "adaptation_examples", "type": "quantitative", "title": "Целевые примеры"},
                    "y": {"field": "accuracy_mean_pct", "type": "quantitative", "title": "Тестовая точность, %"},
                    "color": {"field": "method", "type": "nominal"},
                },
                "options": {"points": "always"},
                "source": data_regime_source,
            }
        ],
        "tables": [
            {
                "id": "confirmatory-table",
                "title": "Сравнения DoRA и LoRA на удержанных сидах",
                "description": "Парные разности тестовой точности для каждой фиксированной архитектуры и каждого сдвига.",
                "dataset": "confirmatory_comparisons",
                "columns": [
                    {"field": "architecture", "label": "Архитектура", "type": "string"},
                    {"field": "scenario", "label": "Сдвиг", "type": "string"},
                    {"field": "mean_delta_pp", "label": "Средняя Δ, п.п.", "type": "number"},
                    {"field": "ci95_low_pp", "label": "Нижняя граница ДИ", "type": "number"},
                    {"field": "ci95_high_pp", "label": "Верхняя граница ДИ", "type": "number"},
                    {"field": "paired_t_p_holm", "label": "p с поправкой Холма", "type": "number"},
                    {"field": "n_pairs", "label": "Пары", "type": "number"},
                ],
                "defaultSort": {"field": "mean_delta_pp", "direction": "desc"},
                "source": confirmatory_source,
            }
        ],
    }

    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated_at,
            "datasets": {
                "data_regime": data_chart.to_dict(orient="records"),
                "confirmatory_comparisons": confirmatory_table.to_dict(orient="records"),
                "data_regime_comparisons": data_comparisons.to_dict(orient="records"),
                "accuracy_summary": accuracy_display.to_dict(orient="records"),
                "synthetic_summary": synthetic.to_dict(orient="records"),
            },
        },
        "sources": sources,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(ARTIFACT_PATH)


if __name__ == "__main__":
    main()
