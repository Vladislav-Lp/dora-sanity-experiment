# DoRA vs LoRA: controlled sanity experiment

Мини-исследование для постерной сессии «Лето с AIRI 2026». Репозиторий воспроизводит контролируемый эксперимент из постера: сравнение замороженной матрицы, LoRA rank 4 и DoRA rank 4 в задаче, где целевые веса отличаются от исходных одновременно направлением и нормами.

DoRA — Weight-Decomposed Low-Rank Adaptation — явно разделяет адаптацию направления и magnitude-компоненты веса. Эксперимент ниже изолирует именно этот механизм; он не является воспроизведением больших экспериментов статьи на LLaMA, LLaVA или VL-BART.

## Главный результат

Средние значения по seeds `123, 124, 125`:

| Метод | Test MSE | Test MAE | Обучаемые параметры | Время обучения, с* |
|---|---:|---:|---:|---:|
| Frozen base | 1.529153 ± 0.053710 | 0.875526 ± 0.036805 | 0 | 0.000 |
| LoRA rank 4 | 0.154478 ± 0.038913 | 0.274470 ± 0.044980 | 128 | 0.872 |
| DoRA rank 4 | 0.000134 ± 0.000010 | 0.009195 ± 0.000369 | 136 | 1.278 |

По среднему Test MSE DoRA уменьшила ошибку относительно LoRA примерно в `1150×` при восьми дополнительных обучаемых параметрах (`+6.25%`). Результат ожидаемо силён, поскольку задача специально построена так, чтобы требовать изменения и направления, и magnitude.

\* Время приведено только для сохранённого запуска и не является переносимым сравнением производительности между устройствами.

## Постановка эксперимента

- `W_base` имеет размер `8 × 24`.
- Целевое направление строится из `W_base` и low-rank сдвига истинного ранга 4.
- Нормы восьми строк дополнительно умножаются на коэффициенты от `0.45` до `1.95`.
- Train/test: `4096 / 1024` синтетических примеров.
- Гауссов шум: `σ = 0.01`.
- LoRA и DoRA: rank 4, одинаковые данные и настройки оптимизации.
- Adam, learning rate `0.05`, batch size `128`, `1200` шагов.
- Seeds: `123, 124, 125`.

Для LoRA обучаются матрицы `A` и `B`:

```text
W = W_base + (1 / r) BA
```

Для DoRA сначала формируется направление, затем оно нормируется по строкам и умножается на отдельный обучаемый magnitude-вектор:

```text
V = W_base + (1 / r) BA
W_DoRA = row_normalize(V) * magnitude
```

## Структура репозитория

```text
.
├── run_experiment.py
├── requirements.txt
├── notebooks/
│   └── AIRI_DoRA_experiment_clean_final.ipynb
├── results/
│   ├── AIRI_DoRA_multi_seed_results.csv
│   └── AIRI_DoRA_summary_results.csv
├── poster/
│   ├── Lapin_Vladislav_DoRA_poster_AIRI_2026.pdf
│   └── Lapin_Vladislav_DoRA_poster_AIRI_2026.pptx
└── report/
    └── AIRI_DoRA_research_proposal.pdf
```

## Быстрый запуск

Требуется Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_experiment.py
```

На Windows вместо `source .venv/bin/activate`:

```powershell
.venv\Scripts\activate
```

После запуска в каталоге `outputs/` появятся:

- результаты каждого метода для каждого seed;
- сводная таблица mean ± std;
- график Test MSE в логарифмической шкале.

Ноутбук из `notebooks/` можно открыть локально в Jupyter или загрузить в Google Colab вручную. Для самого эксперимента интернет и GPU не требуются.

## Ограничения

- Синтетическая задача сконструирована в пользу проверки механизма magnitude/direction.
- Результат не доказывает, что DoRA всегда превосходит LoRA на реальных моделях и датасетах.
- Wall-clock время зависит от оборудования и служит только иллюстрацией дополнительной стоимости обучения.
- Следующий отдельный эксперимент — LoRA/DoRA на DistilBERT/SST-2 — не используется для чисел этого постера.

## Источники

1. Hu et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), ICLR 2022.
2. Liu et al. [DoRA: Weight-Decomposed Low-Rank Adaptation](https://proceedings.mlr.press/v235/liu24bn.html), ICML 2024 Oral.
3. Официальная реализация авторов: [NVlabs/DoRA](https://github.com/NVlabs/DoRA).

Автор мини-исследования: Владислав Лапин, МФТИ ФПМИ / AI360.
