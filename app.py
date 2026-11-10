import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from PIL import Image
from io import BytesIO

# =====================================================
# 1. KONFIGURASI APLIKASI
# =====================================================
st.set_page_config(
    page_title="Smart Plant Monitoring",
    page_icon="🌿",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputsApp"
MASK_DIR = OUTPUT_DIR / "masks"
CSV_FILE = OUTPUT_DIR / "hasil_monitoring.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)


# Pembungkus agar kompatibel dengan versi Streamlit lama maupun baru
def tampil_gambar(img, caption):
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:
        st.image(img, caption=caption, use_container_width=True)


def tampil_tabel(df):
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


# =====================================================
# 2. FUNGSI SEGMENTASI CITRA
# =====================================================
def segmentasi_hijau(image_rgb, lower_hsv, upper_hsv, kernel_size=5):
    """
    Segmentasi warna hijau menggunakan HSV.
    Return: hsv, mask_clean, segmented, total_pixels, green_pixels, percentage
    """
    # RGB -> BGR -> HSV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Mask biner berdasarkan rentang HSV
    mask = cv2.inRange(
        hsv,
        np.array(lower_hsv, dtype=np.uint8),
        np.array(upper_hsv, dtype=np.uint8)
    )

    # Kernel morfologi
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Opening: hilangkan noise putih kecil
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Closing: tutup lubang kecil pada area putih
    mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    green_pixels = cv2.countNonZero(mask_clean)
    total_pixels = mask_clean.size
    percentage = (green_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_clean)

    return hsv, mask_clean, segmented, total_pixels, green_pixels, percentage


# =====================================================
# 3. FUNGSI MENYIMPAN HASIL KE CSV
# =====================================================
def simpan_hasil(data_baru):
    df_baru = pd.DataFrame([data_baru])

    if CSV_FILE.exists():
        df_lama = pd.read_csv(CSV_FILE)
        df_gabungan = pd.concat([df_lama, df_baru], ignore_index=True)
    else:
        df_gabungan = df_baru

    df_gabungan.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    return df_gabungan


# =====================================================
# 4. JUDUL APLIKASI
# =====================================================
st.title("🌿 Smart Plant Monitoring")
st.write(
    """
    Aplikasi monitoring tanaman menggunakan
    **Computer Vision, OpenCV, dan segmentasi warna HSV**.
    Aplikasi menghitung jumlah dan persentase piksel
    yang terdeteksi berwarna hijau pada gambar.
    """
)
st.divider()

# =====================================================
# 5. SIDEBAR: PARAMETER SEGMENTASI
# =====================================================
st.sidebar.header("⚙️ Pengaturan Segmentasi")
st.sidebar.write("Atur rentang warna HSV untuk mendeteksi warna hijau.")

h_min = st.sidebar.slider("Hue minimum", 0, 179, 25)
h_max = st.sidebar.slider("Hue maksimum", 0, 179, 95)
s_min = st.sidebar.slider("Saturation minimum", 0, 255, 35)
s_max = st.sidebar.slider("Saturation maksimum", 0, 255, 255)
v_min = st.sidebar.slider("Value minimum", 0, 255, 30)
v_max = st.sidebar.slider("Value maksimum", 0, 255, 255)

kernel_size = st.sidebar.select_slider(
    "Ukuran kernel morfologi",
    options=[3, 5, 7, 9],
    value=5
)

if h_min > h_max:
    st.sidebar.error("Hue minimum tidak boleh lebih besar daripada Hue maksimum.")
    st.stop()

if s_min > s_max or v_min > v_max:
    st.sidebar.error("Batas minimum tidak boleh lebih besar daripada batas maksimum.")
    st.stop()

lower_hsv = [h_min, s_min, v_min]
upper_hsv = [h_max, s_max, v_max]

# =====================================================
# 6. FORM INPUT DATA TANAMAN
# =====================================================
st.subheader("1. Input Data Tanaman")

col_input1, col_input2 = st.columns(2)
with col_input1:
    plant_id = st.text_input(
        "ID Tanaman", value="P001",
        help="Contoh: P001, P002, atau P003."
    )
with col_input2:
    lokasi = st.text_input("Lokasi Pengamatan", value="Kebun Percobaan")

uploaded_file = st.file_uploader(
    "Unggah foto tanaman atau daun pepaya",
    type=["jpg", "jpeg", "png", "bmp"]
)

# =====================================================
# 7. PEMROSESAN GAMBAR
# =====================================================
if uploaded_file is not None:
    image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
    image_rgb = np.array(image)

    st.subheader("2. Hasil Pengolahan Citra")

    (hsv, mask, segmented,
     total_pixels, green_pixels, percentage) = segmentasi_hijau(
        image_rgb, lower_hsv, upper_hsv, kernel_size
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        tampil_gambar(image_rgb, "Gambar Asli")
    with col2:
        tampil_gambar(mask, "Mask Segmentasi Hijau")
    with col3:
        tampil_gambar(segmented, "Hasil Segmentasi")

    st.subheader("3. Hasil Perhitungan")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Total Piksel", f"{total_pixels:,}")
    metric2.metric("Piksel Hijau", f"{green_pixels:,}")
    metric3.metric("Persentase Hijau", f"{percentage:.2f}%")

    st.progress(min(max(percentage / 100, 0.0), 1.0))

    st.info(
        "Persentase ini menunjukkan proporsi piksel yang lolos segmentasi HSV, "
        "bukan otomatis persentase luas daun atau tingkat kesehatan tanaman."
    )

    # Nama aman untuk berkas
    safe_plant_id = "".join(
        c for c in plant_id.strip() if c.isalnum() or c in ("-", "_")
    ) or "tanaman"
    safe_filename = "".join(
        c for c in Path(uploaded_file.name).stem if c.isalnum() or c in ("-", "_")
    ) or "foto"

    # Kunci unik: foto + parameter. Jika berubah, hasil baru boleh disimpan lagi.
    kunci_hasil = (
        f"{plant_id.strip()}|{uploaded_file.name}|{uploaded_file.size}|"
        f"{lower_hsv}|{upper_hsv}|{kernel_size}"
    )

    # Tombol simpan
    if st.button("💾 Simpan Hasil Monitoring", type="primary"):
        if st.session_state.get("kunci_tersimpan") == kunci_hasil:
            st.warning(
                "Hasil ini (foto + parameter yang sama) sudah disimpan. "
                "Ganti foto atau ubah parameter untuk pengukuran berikutnya."
            )
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            mask_filename = f"{safe_plant_id}_{safe_filename}_{timestamp}_mask.png"
            mask_path = MASK_DIR / mask_filename
            cv2.imwrite(str(mask_path), mask)

            data_monitoring = {
                "waktu_pemrosesan": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "id_tanaman": plant_id.strip(),
                "lokasi": lokasi.strip(),
                "nama_file": uploaded_file.name,
                "total_piksel": total_pixels,
                "piksel_hijau": green_pixels,
                "persentase_hijau": round(percentage, 2),
                "hsv_bawah": str(lower_hsv),
                "hsv_atas": str(upper_hsv),
                "ukuran_kernel": kernel_size,
                "file_mask": str(mask_path.relative_to(BASE_DIR))
            }
            simpan_hasil(data_monitoring)
            st.session_state["kunci_tersimpan"] = kunci_hasil
            st.success(f"Hasil monitoring tanaman {plant_id} berhasil disimpan.")

    # Tombol unduh mask
    mask_buffer = BytesIO()
    Image.fromarray(mask).save(mask_buffer, format="PNG")
    st.download_button(
        label="⬇️ Unduh Mask Segmentasi",
        data=mask_buffer.getvalue(),
        file_name=f"{safe_plant_id}_mask.png",
        mime="image/png"
    )

    # Tombol unduh hasil segmentasi
    segmented_buffer = BytesIO()
    Image.fromarray(segmented).save(segmented_buffer, format="PNG")
    st.download_button(
        label="⬇️ Unduh Hasil Segmentasi",
        data=segmented_buffer.getvalue(),
        file_name=f"{safe_plant_id}_segmentasi.png",
        mime="image/png"
    )

# =====================================================
# 8. RIWAYAT MONITORING
# =====================================================
st.divider()
st.subheader("4. Riwayat Monitoring")

if CSV_FILE.exists():
    df_history = pd.read_csv(CSV_FILE)

    if not df_history.empty:
        # Perbaikan: filter berdasarkan ID tanaman
        daftar_id = sorted(df_history["id_tanaman"].astype(str).unique())
        id_dipilih = st.multiselect(
            "Filter ID Tanaman", options=daftar_id, default=daftar_id
        )
        df_view = df_history[df_history["id_tanaman"].astype(str).isin(id_dipilih)]

        tampil_tabel(df_view)

        st.download_button(
            label="⬇️ Unduh Riwayat CSV",
            data=df_view.to_csv(index=False).encode("utf-8-sig"),
            file_name="hasil_monitoring.csv",
            mime="text/csv"
        )

        # Perbaikan: ringkasan statistik
        df_plot = df_view.copy()
        df_plot["persentase_hijau"] = pd.to_numeric(
            df_plot["persentase_hijau"], errors="coerce"
        )
        df_plot = df_plot.dropna(subset=["persentase_hijau"])

        if not df_plot.empty:
            st.subheader("Ringkasan Statistik")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Jumlah Data", f"{len(df_plot)}")
            s2.metric("Rata-rata", f"{df_plot['persentase_hijau'].mean():.2f}%")
            s3.metric("Minimum", f"{df_plot['persentase_hijau'].min():.2f}%")
            s4.metric("Maksimum", f"{df_plot['persentase_hijau'].max():.2f}%")

            st.subheader("5. Grafik Monitoring")

            df_plot = df_plot.reset_index(drop=True)
            df_plot["label_grafik"] = (
                (df_plot.index + 1).astype(str) + ". "
                + df_plot["id_tanaman"].astype(str) + " - "
                + df_plot["nama_file"].astype(str)
                + " (k=" + df_plot["ukuran_kernel"].astype(str) + ")"
            )

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(df_plot["label_grafik"], df_plot["persentase_hijau"])
            ax.set_title("Persentase Piksel Tersegmentasi Hijau")
            ax.set_xlabel("Tanaman dan Nama File")
            ax.set_ylabel("Persentase (%)")
            ax.tick_params(axis="x", labelrotation=45)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            fig.tight_layout()

            st.pyplot(fig)

            grafik_path = OUTPUT_DIR / "grafik_persentase_hijau.png"
            fig.savefig(grafik_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            st.info("Tidak ada data untuk filter yang dipilih.")
    else:
        st.info("Belum ada data monitoring.")
else:
    st.info("Belum ada riwayat monitoring. Unggah gambar dan simpan hasil pertama Anda.")

# =====================================================
# 9. INFORMASI APLIKASI
# =====================================================
st.divider()
st.caption("Smart Plant Monitoring | Python, OpenCV, HSV, Matplotlib, dan Streamlit")