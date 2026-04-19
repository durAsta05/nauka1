# -*- coding: utf-8 -*-
"""
Спектр: интенсивность от энергии для 4 слоёв широты
Спутник: MetOp-03
Каналы: E1 (>40 keV), E2 (>130 keV), E3 (>287 keV), E4 (>612 keV)
Слои широты: 66-70°, 70-74°, 74-78°, 78-82°

Добавлено:
  — Фильтр по ошибкам измерений (max_error_ratio)
  — Взвешенное среднее: weights = 1/errors²
  — Проверка физического диапазона k (0.1 < k < 10)
  — Параметр Fo = exp(intercept) в результатах

График 1 (верх-лево): точки средних значений
График 2 (верх-право): линейная аппроксимация (E1-E3, без E4)
График 3 (низ-лево): качество аппроксимации (точки + линии в log-log, R²)
График 4 (низ-право): сводная таблица R², k, Fo и флагов качества
"""

import matplotlib

matplotlib.use("Agg")

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

print("=" * 70)
print("СПЕКТР: интенсивность от энергии")
print("=" * 70)

# =============================================================================
# ⚙️ НАСТРОЙКИ — меняй дату здесь
# =============================================================================

YEAR = 2026  # год
MONTH = 1  # месяц (1-12)
DAY =  20   # день (1-31)

RESULTS_FOLDER = rf"b:\nauka\2026-04-13"
BASE_DIR = r"b:\nauka"
METOP3_FILE = os.path.join(
    BASE_DIR, "metop3", f"poes_m03_{YEAR}{MONTH:02d}{DAY:02d}_proc.nc"
)

DPI = 300

# Слои широты (северное полушарие)
LAT_LAYERS = [
    (66, 70),
    (70, 74),
    (74, 78),
    (78, 82),
]

# Каналы энергии
CHANNELS = {
    "e1": {
        "var": "mep_ele_tel0_flux_e1",
        "err": "mep_ele_tel0_flux_e1_err",
        "label": "E1",
        "energy": 40,
    },
    "e2": {
        "var": "mep_ele_tel0_flux_e2",
        "err": "mep_ele_tel0_flux_e2_err",
        "label": "E2",
        "energy": 130,
    },
    "e3": {
        "var": "mep_ele_tel0_flux_e3",
        "err": "mep_ele_tel0_flux_e3_err",
        "label": "E3",
        "energy": 287,
    },
    "e4": {
        "var": "mep_ele_tel0_flux_e4",
        "err": "mep_ele_tel0_flux_e4_err",
        "label": "E4",
        "energy": 612,
    },
}

# Цвета для слоёв
LAYER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# =============================================================================
# 🔧 ФИЛЬТРЫ И ПРОВЕРКИ (высокий приоритет)
# =============================================================================

# Максимальное отношение ошибки к потоку (30%)
MAX_ERROR_RATIO = 0.30

# Физический диапазон спектрального индекса k
K_MIN = 0.1
K_MAX = 10.0

# Минимальный R² для "хороших данных"
MIN_R_SQUARED = 0.80

# =============================================================================
# 📖 ЧТЕНИЕ ДАННЫХ
# =============================================================================

date_label = f"{DAY:02d}.{MONTH:02d}.{YEAR}"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

print(f"\n📊 Чтение данных за {date_label}...")
print(f"Файл: {os.path.basename(METOP3_FILE)}")

if not os.path.exists(METOP3_FILE):
    print(f"  ✗ Файл не найден!")
    exit()

ds = nc.Dataset(METOP3_FILE, "r", mask=False)

lat = ds.variables["lat"][:]
lon = ds.variables["lon"][:]
tv = ds.variables["time"]
if hasattr(tv, "data"):
    time_raw = np.array(tv.data[:], dtype=np.float64)
else:
    time_raw = np.array(tv[:], dtype=np.float64)

# Чтение потоков и ошибок
channels_data = {}
errors_data = {}
for ch_id, ch_info in CHANNELS.items():
    flux = ds.variables[ch_info["var"]][:]
    channels_data[ch_id] = flux
    # Чтение ошибок (если есть в файле)
    if ch_info["err"] in ds.variables:
        errors_data[ch_id] = ds.variables[ch_info["err"]][:]
    else:
        errors_data[ch_id] = None

ds.close()

# Фильтрация валидных данных
valid_base = (time_raw > 1e12) & (~np.isnan(lat)) & (~np.isnan(lon))

print(f"  ✓ Всего точек: {np.sum(valid_base):,}")

# =============================================================================
# 📊 РАСЧЁТ СРЕДНИХ ПО СЛОЯМ (с фильтрацией по ошибкам + взвешивание)
# =============================================================================

print(f"\n📈 Расчёт средних значений по слоям широты...")
print(f"  Фильтр ошибок: max_error_ratio = {MAX_ERROR_RATIO:.0%}")
print(f"  Взвешенное среднее: weights = 1/errors²")

layer_means = []

for i, (lat_min, lat_max) in enumerate(LAT_LAYERS):
    print(f"\n  Слой {i+1}: {lat_min}°–{lat_max}°")
    layer_results = {}

    layer_mask = valid_base & (lat >= lat_min) & (lat < lat_max)

    for ch_id, ch_info in CHANNELS.items():
        flux = channels_data[ch_id]
        err = errors_data.get(ch_id)

        valid_flux = flux[layer_mask]
        valid_err = err[layer_mask] if err is not None else None

        # Базовая фильтрация
        base_mask = (~np.isnan(valid_flux)) & (valid_flux > 0)

        # Фильтр по ошибкам: |err/flux| < MAX_ERROR_RATIO
        if valid_err is not None:
            err_mask = (~np.isnan(valid_err)) & (valid_err > 0)
            combined_mask = base_mask & err_mask

            flux_ok = valid_flux[combined_mask]
            err_ok = valid_err[combined_mask]

            with np.errstate(divide="ignore", invalid="ignore"):
                error_ratio = np.abs(err_ok / flux_ok)
                ratio_mask = error_ratio < MAX_ERROR_RATIO

            flux_ok = flux_ok[ratio_mask]
            err_ok = err_ok[ratio_mask]
        else:
            flux_ok = valid_flux[base_mask]
            err_ok = None

        if len(flux_ok) > 0:
            n_total = len(flux_ok)
            n_filtered = np.sum(base_mask) if valid_err is not None else n_total

            # Взвешенное среднее (если есть ошибки)
            if err_ok is not None and len(err_ok) == len(flux_ok):
                weights = 1.0 / (err_ok**2 + 1e-10)
                mean_val = np.average(flux_ok, weights=weights)
            else:
                mean_val = np.mean(flux_ok)

            std_val = np.std(flux_ok)

            layer_results[ch_id] = {
                "mean": mean_val,
                "std": std_val,
                "count": n_total,
                "count_before_filter": n_filtered,
            }

            filter_info = ""
            if valid_err is not None and n_filtered != n_total:
                filter_info = f", отфильтровано {n_filtered}→{n_total}"

            print(
                f"    {ch_info['label']}: {mean_val:.2e} ± {std_val:.2e} ({n_total:,} точек{filter_info})"
            )
        else:
            layer_results[ch_id] = None
            print(f"    {ch_info['label']}: нет данных")

    layer_means.append(layer_results)

# =============================================================================
# 🎨 ОТРИСОВКА
# =============================================================================

print(f"\n🎨 Построение графиков...")

fig, axes = plt.subplots(2, 2, figsize=(22, 18))
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

# =========================================================================
# ГРАФИК 1: Точки средних значений
# =========================================================================

ax1.set_title(
    f"Средние значения потока (взвешенные)\n{date_label}, MetOp-03, северное полушарие",
    fontsize=13,
    fontweight="bold",
    pad=10,
)

for i, (lat_min, lat_max) in enumerate(LAT_LAYERS):
    energies = []
    means = []
    stds = []

    for ch_id in CHANNELS:
        if layer_means[i][ch_id] is not None:
            energies.append(CHANNELS[ch_id]["energy"])
            means.append(layer_means[i][ch_id]["mean"])
            stds.append(layer_means[i][ch_id]["std"])

    if energies:
        energies = np.array(energies)
        means = np.array(means)
        stds = np.array(stds)

        ax1.scatter(
            energies,
            means,
            c=LAYER_COLORS[i],
            s=80,
            zorder=5,
            label=f"{lat_min}°–{lat_max}°",
        )

        ax1.errorbar(
            energies,
            means,
            yerr=stds,
            fmt="o",
            c=LAYER_COLORS[i],
            alpha=0.5,
            capsize=3,
            zorder=4,
        )

        for j, (e, m) in enumerate(zip(energies, means)):
            ax1.annotate(
                f"{m:.1e}",
                (e, m),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=7,
                color=LAYER_COLORS[i],
                fontweight="bold",
            )

ax1.set_xlabel("Энергия [keV]", fontsize=11)
ax1.set_ylabel("Поток [см⁻² с⁻¹ ср⁻¹]", fontsize=11)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(True, alpha=0.3, which="both")

# =========================================================================
# ГРАФИК 2: Линейная аппроксимация (E1-E3, без E4)
# =========================================================================

ax2.set_title(
    f"Линейная аппроксимация (E1–E3, без E4)\n{date_label}, MetOp-03, северное полушарие",
    fontsize=13,
    fontweight="bold",
    pad=10,
)

fit_results = []

for i, (lat_min, lat_max) in enumerate(LAT_LAYERS):
    energies = []
    means = []

    for ch_id in ["e1", "e2", "e3"]:
        if layer_means[i][ch_id] is not None:
            energies.append(CHANNELS[ch_id]["energy"])
            means.append(layer_means[i][ch_id]["mean"])

    if len(energies) >= 2:
        energies = np.array(energies)
        means = np.array(means)
        log_e = np.log(energies)
        log_m = np.log(means)

        coeffs = np.polyfit(log_e, log_m, 1)
        poly_fn = np.poly1d(coeffs)

        # R²
        m_pred = poly_fn(log_e)
        ss_res = np.sum((log_m - m_pred) ** 2)
        ss_tot = np.sum((log_m - np.mean(log_m)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Параметры аппроксимации
        slope = coeffs[0]  # наклон в log-log (отрицательный для убывающего спектра)
        intercept = coeffs[1]  # a
        k = -slope  # спектральный индекс (положительный)
        Fo = np.exp(intercept)  # нормировка: F = Fo * E^(-k)

        # Проверка физического диапазона k
        k_valid = K_MIN <= k <= K_MAX
        r2_good = r_squared >= MIN_R_SQUARED

        # Флаг качества
        if k_valid and r2_good:
            quality = "ХОРОШИЕ ДАННЫЕ"
        elif not k_valid and r2_good:
            quality = "СОМНИТЕЛЬНЫЕ: нереальный k"
        elif k_valid and not r2_good:
            quality = "СОМНИТЕЛЬНЫЕ: низкое R²"
        else:
            quality = "СОМНИТЕЛЬНЫЕ: k и R²"

        fit_results.append(
            {
                "layer_idx": i,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "energies": energies,
                "means": means,
                "coeffs": coeffs,
                "poly_fn": poly_fn,
                "r_squared": r_squared,
                "slope": slope,
                "k": k,
                "intercept": intercept,
                "Fo": Fo,
                "k_valid": k_valid,
                "quality": quality,
            }
        )

        e_fit = np.linspace(40, 300, 100)
        m_fit = np.exp(poly_fn(np.log(e_fit)))

        ax2.scatter(
            energies,
            means,
            c=LAYER_COLORS[i],
            s=80,
            zorder=5,
            label=f"{lat_min}°–{lat_max}°",
        )

        # Стиль линии зависит от качества
        if quality == "ХОРОШИЕ ДАННЫЕ":
            ax2.plot(e_fit, m_fit, "--", c=LAYER_COLORS[i], linewidth=2.5, zorder=4)
        else:
            ax2.plot(e_fit, m_fit, ":", c="red", linewidth=1.5, zorder=4, alpha=0.6)

        ax2.annotate(
            f"k={k:.2f}",
            (energies[-1], means[-1]),
            textcoords="offset points",
            xytext=(10, -12),
            fontsize=8,
            color=LAYER_COLORS[i],
            fontweight="bold",
        )

ax2.set_xlabel("Энергия [keV]", fontsize=11)
ax2.set_ylabel("Поток [см⁻² с⁻¹ ср⁻¹]", fontsize=11)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(True, alpha=0.3, which="both")

# =========================================================================
# ГРАФИК 3: Качество аппроксимации — точки + линии, R² подписи
# =========================================================================

ax3.set_title(
    f"Качество линейной аппроксимации (log-log)\nR² — коэффициент детерминации",
    fontsize=13,
    fontweight="bold",
    pad=10,
)

for fit in fit_results:
    i = fit["layer_idx"]
    color = LAYER_COLORS[i]

    ax3.scatter(
        fit["energies"],
        fit["means"],
        c=color,
        s=80,
        zorder=5,
        label=f"{fit['lat_min']}°–{fit['lat_max']}°",
    )

    e_fit = np.linspace(40, 300, 100)
    m_fit = np.exp(fit["poly_fn"](np.log(e_fit)))

    # Стиль линии зависит от качества
    if fit["quality"] == "ХОРОШИЕ ДАННЫЕ":
        ax3.plot(e_fit, m_fit, "--", c=color, linewidth=2, zorder=4)
    else:
        ax3.plot(e_fit, m_fit, ":", c="red", linewidth=1.5, zorder=4, alpha=0.6)

    ax3.annotate(
        f"R²={fit['r_squared']:.4f}",
        (fit["energies"][-1], fit["means"][-1]),
        textcoords="offset points",
        xytext=(10, -15),
        fontsize=8,
        color=color,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

ax3.set_xlabel("Энергия [keV]", fontsize=11)
ax3.set_ylabel("Поток [см⁻² с⁻¹ ср⁻¹]", fontsize=11)
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.legend(fontsize=8, loc="upper right")
ax3.grid(True, alpha=0.3, which="both")

# =========================================================================
# ГРАФИК 4: Сводная таблица R², k, Fo и флагов качества
# =========================================================================

ax4.axis("off")
ax4.set_title(
    f"Параметры аппроксимации\n{date_label}, MetOp-03",
    fontsize=13,
    fontweight="bold",
    pad=10,
)

table_header = "Слой          |  k (наклон) |  Fo       |  R²      |  Качество\n"
table_sep = "—" * 72 + "\n"
table_rows = ""

for fit in fit_results:
    layer_label = f"{fit['lat_min']}°–{fit['lat_max']}°"
    k = fit["k"]
    Fo = fit["Fo"]
    r2 = fit["r_squared"]
    quality = fit["quality"]

    # Форматирование Fo
    if Fo > 1000 or Fo < 0.001:
        Fo_str = f"{Fo:.2e}"
    else:
        Fo_str = f"{Fo:.4f}"

    # Звёздочная система + флаг
    if quality == "ХОРОШИЕ ДАННЫЕ":
        stars = "★★★★★"
    elif r2 >= 0.95:
        stars = "★★★★☆"
    elif r2 >= 0.90:
        stars = "★★★☆☆"
    elif r2 >= 0.80:
        stars = "★★☆☆☆"
    else:
        stars = "★☆☆☆☆"

    quality_short = f"{stars} {quality}"

    # Маркер если k вне диапазона
    k_marker = "" if fit["k_valid"] else " [!]"

    table_rows += f"{layer_label:<14}| {k:>11.4f}{k_marker} | {Fo_str:>9} | {r2:>8.5f} | {quality_short}\n"

table_text = table_header + table_sep + table_rows

ax4.text(
    0.05,
    0.88,
    table_text,
    fontsize=9,
    fontfamily="monospace",
    va="top",
    ha="left",
    transform=ax4.transAxes,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
)

ax4.text(
    0.05,
    0.18,
    "Аппроксимация: ln(F) = a + k·ln(E)  →  F = Fo·E^(-k)\n"
    "Диапазон: 40–300 keV (E1–E3), E4 не используется\n\n"
    f"Фильтр ошибок: |err/flux| < {MAX_ERROR_RATIO:.0%}\n"
    f"Физический k: {K_MIN}–{K_MAX}\n"
    f"Минимальный R²: {MIN_R_SQUARED}\n\n"
    "[!] — k вне физического диапазона\n"
    "Стиль линии: сплошная=хорошо, красная пунктир=сомнительно",
    fontsize=8,
    va="top",
    ha="left",
    transform=ax4.transAxes,
    style="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5),
)

# =========================================================================
# ОБЩИЙ ЗАГОЛОВОК И СОХРАНЕНИЕ
# =========================================================================

fig.suptitle(
    f"MetOp-03  |  Спектр электронов  |  {date_label}  |  Слои: 66°–82° N",
    fontsize=15,
    fontweight="bold",
    y=0.995,
)

fig.tight_layout(rect=[0, 0, 1, 0.97])

out_file = f"Spectrum_MetOp03_E1-4_{YEAR}{MONTH:02d}{DAY:02d}_lat66-82N.png"
out_path = os.path.join(RESULTS_FOLDER, out_file)
fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
plt.close(fig)

print(f"\n✅ Сохранено: {out_path}")

print("\n📊 Параметры аппроксимации:")
for fit in fit_results:
    k_marker = "" if fit["k_valid"] else " [!]"
    print(
        f"  {fit['lat_min']}°–{fit['lat_max']}°: k={fit['k']:.4f}{k_marker}, "
        f"Fo={fit['Fo']:.2e}, R²={fit['r_squared']:.5f} [{fit['quality']}]"
    )

print("\n" + "=" * 70)
print("ГОТОВО!")
print("=" * 70)
