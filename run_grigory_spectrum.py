# -*- coding: utf-8 -*-
"""
Запуск кода Григория (Spectra.py) для каждого дня 16–20 января 2026
МетOp-03, широта 66-82N, долгота 0-360E
Результаты сохраняются в b:\nauka\2026-04-13
"""

import sys
import os
import importlib.util

# Пути
RESULTS_DIR = r"b:\nauka\2026-04-13"
BASE_DIR = r"b:\nauka"
SPECTRA_PATH = r"b:\nauka\grigory\Spectra.py"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Дни для обработки
DAYS = [16, 17, 18, 19, 20]
MONTH = 1
YEAR = 2026

# Широтные полосы (северное полушарие, 60-82N с шагом 4)
LAT_BANDS = [(66, 70), (70, 74), (74, 78), (78, 82)]
LON_RANGE = (0, 360)

# Загрузка модуля Spectra.py
print("Загрузка Spectra.py...")
spec = importlib.util.spec_from_file_location("Spectra", SPECTRA_PATH)
spectra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spectra)

# Патчим SAVE_DIR
spectra.SAVE_DIR = RESULTS_DIR

# Патчим FILE_PATTERNS для наших путей
spectra.FILE_PATTERNS = {
    "m01": os.path.join(BASE_DIR, "metop1", "poes_m01_{date}_proc.nc"),
    "m03": os.path.join(BASE_DIR, "metop3", "poes_m03_{date}_proc.nc"),
}


# Отключаем E4 из аппроксимации (патчим функцию)
def never_include_e4(energies, fluxes, filter_settings):
    """Всегда исключаем E4 из аппроксимации"""
    return False


spectra.should_include_e4_in_fit = never_include_e4

# Настройки: только MetOp-03
spectra.MULTI_FILE_PROCESSING = False
spectra.SELECTED_SATELLITE_INDEX = 0
spectra.SATELLITES = ["m03"]

# Настройки фильтрации (стандартные)
spectra.FILTER_SETTINGS = {
    "enable_error_filter": True,
    "max_error_ratio": 0.30,
    "enable_flux_range_filter": True,
    "min_flux": 1e-6,
    "max_flux": 1e8,
    "enable_std_filter": False,
    "max_std_ratio": 2.0,
    "enable_count_filter": True,
    "min_measurements": 5,
    "enable_e4_in_fit": False,
    "min_r_squared": 0.7,
    "min_fit_points": 2,
    "exclude_dubious_fits": False,
}

CURRENT_FILTERS = spectra.FILTER_SETTINGS

# Режим фильтрации: только по широте/долготе (без времени)
spectra.FILTER_MODES = {
    "by_lat_lon_only": True,
    "by_time_only": False,
    "by_both": False,
}

TIME_RANGE = None  # без временной фильтрации

# Обработка каждого дня
results_summary = {}

for day in DAYS:
    date_str = f"{YEAR}{MONTH:02d}{day:02d}"
    print("\n" + "=" * 70)
    print(f"📅 ОБРАБОТКА: {date_str} ({day:02d}.{MONTH:02d}.{YEAR})")
    print("=" * 70)

    spectra.TARGET_DATE = date_str

    try:
        df, fit_results, fig, lat_band_data = spectra.process_poes_combined_analysis(
            date_str, LAT_BANDS, LON_RANGE, CURRENT_FILTERS, TIME_RANGE
        )

        if fit_results and fig:
            # Сохраняем (без E4)
            out_file = f"Grigory_Spectrum_MetOp03_E1-3_noE4_{date_str}_lat66-82N.png"
            out_path = os.path.join(RESULTS_DIR, out_file)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")

            results_summary[date_str] = {
                "fit_results": fit_results,
                "image": out_file,
            }

            print(f"\n✅ Сохранено: {out_file}")
            for band, params in fit_results.items():
                print(
                    f"  {band}: k={params['k']:.3f}, R²={params['r_squared']:.4f}, Fo={params['Fo']:.2e}"
                )
        else:
            results_summary[date_str] = {"error": "Нет данных или не удалось построить"}
            print(f"  ✗ Нет данных")

    except Exception as e:
        results_summary[date_str] = {"error": str(e)}
        print(f"  ✗ Ошибка: {e}")
        import traceback

        traceback.print_exc()

# Итоговая сводка
print("\n" + "=" * 70)
print("ИТОГОВАЯ СВОДКА")
print("=" * 70)

for date_str, res in results_summary.items():
    if "error" in res:
        print(f"  {date_str}: ✗ {res['error']}")
    else:
        print(f"  {date_str}: ✓ {res['image']}")
        for band, params in res["fit_results"].items():
            print(f"    {band}: k={params['k']:.3f}, R²={params['r_squared']:.4f}")

print("\n" + "=" * 70)
print("ГОТОВО!")
print("=" * 70)
