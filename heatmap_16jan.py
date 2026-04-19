# -*- coding: utf-8 -*-
"""
Тепловая карта широта-долгота
Спутники: MetOp-01 и MetOp-03
Бин: 10°×10°
Общая цветовая шкала для сравнения
"""

import matplotlib

matplotlib.use("Agg")

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from datetime import datetime
from glob import glob

print("=" * 70)
print("ТЕПЛОВАЯ КАРТА: MetOp-01 vs MetOp-03")
print("=" * 70)

# =============================================================================
# ⚙️ НАСТРОЙКИ — меняй дату здесь
# =============================================================================

YEAR = 2026  # год
MONTH = 1  # месяц (1-12)
DAY = 20  # день (1-31)

LAT_BIN = 10.0  # шаг бина по широте
LON_BIN = 10.0  # шаг бина по долготе

DPI = 300
COLOR_MAP = "jet"

# =============================================================================
# 📁 АВТОПОИСК ФАЙЛОВ ПО ДАТЕ
# =============================================================================

date_str = f"{YEAR}{MONTH:02d}{DAY:02d}"  # например: 20260116
date_label = f"{DAY:02d}.{MONTH:02d}.{YEAR}"  # например: 16.01.2026

RESULTS_FOLDER = rf"b:\nauka\2026-04-13"
BASE_DIR = r"b:\nauka"

METOP1_FILE = os.path.join(BASE_DIR, "metop1", f"poes_m01_{date_str}_proc.nc")
METOP3_FILE = os.path.join(BASE_DIR, "metop3", f"poes_m03_{date_str}_proc.nc")

CHANNEL_VAR = "mep_ele_tel0_flux_e1"
CHANNEL_LABEL = "E1: >40 keV"

# =============================================================================
# 📖 ЧТЕНИЕ ДАННЫХ
# =============================================================================

os.makedirs(RESULTS_FOLDER, exist_ok=True)


def read_nc_file(file_path, channel_var):
    """Чтение NetCDF файла и возврат lat, lon, flux"""
    print(f"\n📂 Чтение: {os.path.basename(file_path)}")

    if not os.path.exists(file_path):
        print(f"  ✗ Файл не найден!")
        return None

    ds = nc.Dataset(file_path, "r", mask=False)

    lat = ds.variables["lat"][:]
    lon = ds.variables["lon"][:]
    flux = ds.variables[channel_var][:]

    tv = ds.variables["time"]
    if hasattr(tv, "data"):
        time_raw = np.array(tv.data[:], dtype=np.float64)
    else:
        time_raw = np.array(tv[:], dtype=np.float64)

    ds.close()

    # Фильтрация валидных данных
    valid = (
        (~np.isnan(flux))
        & (flux > 0)
        & (time_raw > 1e12)
        & (~np.isnan(lat))
        & (~np.isnan(lon))
    )

    idx = np.where(valid)[0]

    all_lat, all_lon, all_flux, all_time = [], [], [], []
    for i in idx:
        try:
            t = datetime.fromtimestamp(time_raw[i] / 1000.0)
            all_time.append(t)
            all_lat.append(lat[i])
            all_lon.append(lon[i])
            all_flux.append(flux[i])
        except:
            pass

    if not all_flux:
        print(f"  ✗ Нет валидных данных")
        return None

    print(f"  ✓ {len(all_flux):,} точек")

    return {
        "lat": np.array(all_lat),
        "lon": np.array(all_lon),
        "flux": np.array(all_flux),
        "time": all_time,
    }


print(f"\n📊 Чтение данных за {date_label}...")
print(f"Канал: {CHANNEL_LABEL}")
print(f"MetOp-01: {os.path.basename(METOP1_FILE)}")
print(f"MetOp-03: {os.path.basename(METOP3_FILE)}")

data_m01 = read_nc_file(METOP1_FILE, CHANNEL_VAR)
data_m03 = read_nc_file(METOP3_FILE, CHANNEL_VAR)

if data_m01 is None or data_m03 is None:
    print("\n✗ Нет данных для построения карты!")
    exit()

print(f"\n  MetOp-01: {len(data_m01['flux']):,} точек")
print(f"  MetOp-03: {len(data_m03['flux']):,} точек")

# Определение общей цветовой шкалы
all_fluxes = np.concatenate([data_m01["flux"], data_m03["flux"]])
vmin = 1e1
vmax = np.max(all_fluxes)
print(f"  Общая шкала: {vmin:.0e} — {vmax:.2e}")

# =============================================================================
# 📊 БИНИРОВАНИЕ 10°×10°
# =============================================================================


def bin_data_spatial(data, lat_bin, lon_bin):
    """Биннинг данных в сетку широта-долгота"""
    lat_vals = np.array(data["lat"], dtype=np.float64)
    lon_vals = np.array(data["lon"], dtype=np.float64)
    flux_vals = np.array(data["flux"], dtype=np.float64)

    lat_bins_arr = np.arange(-90, 90 + lat_bin, lat_bin)
    lon_bins_arr = np.arange(0, 360 + lon_bin, lon_bin)

    # Сумма потока в каждой ячейке
    H, _, _ = np.histogram2d(
        lon_vals, lat_vals, bins=[lon_bins_arr, lat_bins_arr], weights=flux_vals
    )
    # Количество точек в каждой ячейке
    N, _, _ = np.histogram2d(lon_vals, lat_vals, bins=[lon_bins_arr, lat_bins_arr])

    # Среднее значение потока
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(N.T > 0, H.T / N.T, np.nan)

    total_filled = int(np.sum(N > 0))
    return Z, total_filled, lat_bins_arr, lon_bins_arr


print(f"\n📈 Биннинг {LAT_BIN:.0f}°×{LON_BIN:.0f}°...")

Z_m01, filled_m01, lat_bins, lon_bins = bin_data_spatial(data_m01, LAT_BIN, LON_BIN)
print(f"  MetOp-01: заполнено {filled_m01} ячеек из {Z_m01.shape[0] * Z_m01.shape[1]}")

Z_m03, filled_m03, lat_bins, lon_bins = bin_data_spatial(data_m03, LAT_BIN, LON_BIN)
print(f"  MetOp-03: заполнено {filled_m03} ячеек из {Z_m03.shape[0] * Z_m03.shape[1]}")

# =============================================================================
# 🎨 ОТРИСОВКА
# =============================================================================


def draw_heatmap(ax, Z, title, total_filled, lat_bins_arr, lon_bins_arr, vmin, vmax):
    """Отрисовка тепловой карты"""
    Z_masked = np.ma.masked_where(np.isnan(Z) | (Z < vmin), Z)

    im = ax.pcolormesh(
        lon_bins_arr,
        lat_bins_arr,
        Z_masked,
        cmap=COLOR_MAP,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        shading="auto",
        alpha=0.9,
    )

    ax.set_xlabel("Longitude [°]", fontsize=11)
    ax.set_ylabel("Latitude [°]", fontsize=11)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.tick_params(labelsize=9)
    ax.grid(True, color="gray", linewidth=0.3, alpha=0.2)

    ax.set_title(
        f"{title}\nЗаполнено: {total_filled} ячеек",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    return im


fig, axes = plt.subplots(1, 2, figsize=(20, 9))

im1 = draw_heatmap(
    axes[0], Z_m01, "MetOp-01", filled_m01, lat_bins, lon_bins, vmin, vmax
)
im2 = draw_heatmap(
    axes[1], Z_m03, "MetOp-03", filled_m03, lat_bins, lon_bins, vmin, vmax
)

fig.suptitle(
    f"MetOp-01 vs MetOp-03  |  {CHANNEL_LABEL}  |  {date_label}  |  Бин: {LAT_BIN:.0f}°×{LON_BIN:.0f}°",
    fontsize=14,
    fontweight="bold",
    y=0.97,
)

# Общая цветовая шкала
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im1, cax=cbar_ax, label="Средний поток [см⁻² с⁻¹ ср⁻¹]")
cbar.ax.tick_params(labelsize=10)

fig.tight_layout(rect=[0, 0, 0.91, 0.94])

out_file = f"LatLon_MetOp01_vs_MetOp03_E1_{date_str}_bin{LAT_BIN:.0f}x{LON_BIN:.0f}_spatial.png"
out_path = os.path.join(RESULTS_FOLDER, out_file)
fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
plt.close(fig)

print(f"\n✅ Сохранено: {out_path}")
print("\n" + "=" * 70)
print("ГОТОВО!")
print("=" * 70)
