## Link YT = https://youtu.be/j7FrvU4tAEk?si=nJFCSdMdq5d3yNtb

import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import librosa
import scipy.signal as signal
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
# 1. FUNGSI-FUNGSI PENGOLAHAN SINYAL DIGITAL (DSP)

def add_noise(y, noise_type='white', amount=0.005):
    """Menambahkan derau ke sinyal audio."""
    if noise_type == 'white':
        noise = np.random.randn(len(y))
    y_noisy = y + noise * amount
    return y_noisy


def apply_lowpass_filter(y, sr, cutoff_hz=1000):
    """Menerapkan filter low-pass Butterworth."""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_hz / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    return signal.lfilter(b, a, y)


def apply_highpass_filter(y, sr, cutoff_hz=500):
    """Menerapkan filter high-pass Butterworth."""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_hz / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
    return signal.lfilter(b, a, y)


def apply_wiener_filter(y):
    """Menerapkan filter Wiener untuk reduksi derau."""
    return signal.wiener(y, mysize=15)


# 2. KELAS UTAMA APLIKASI GUI TKINTER

class AudioFilterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aplikasi Filter Audio Digital")
        self.geometry("1000x800")

        # Inisialisasi variabel data
        self.original_audio = None
        self.noisy_audio = None
        self.filtered_audio = None
        self.sample_rate = None

        # Mengatur style
        style = ttk.Style(self)
        style.theme_use('clam')

        # Membuat frame utama
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Membuat frame kontrol (kiri) dan frame plot (kanan)
        control_frame = ttk.Frame(main_frame, width=250, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        plot_frame = ttk.Frame(main_frame, padding="10")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Menambahkan widget ke frame kontrol
        self._create_control_widgets(control_frame)

        # Menambahkan kanvas plot ke frame plot
        self._create_plot_widgets(plot_frame)

    def _create_control_widgets(self, parent):
        """Membuat semua tombol, slider, dan dropdown."""

        # --- Bagian 1: Unggah File ---
        ttk.Label(parent, text="1. Unggah File Audio", font=("Helvetica", 12, "bold")).pack(pady=5, anchor="w")
        self.btn_upload = ttk.Button(parent, text="Pilih File (.wav, .mp3)", command=self.load_audio)
        self.btn_upload.pack(fill=tk.X, pady=5)
        self.lbl_filename = ttk.Label(parent, text="Belum ada file dipilih", wraplength=200)
        self.lbl_filename.pack(pady=5, anchor="w")

        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=10)

        # --- Bagian 2: Tambah Derau ---
        ttk.Label(parent, text="2. Tambahkan Derau", font=("Helvetica", 12, "bold")).pack(pady=5, anchor="w")
        ttk.Label(parent, text="Intensitas Derau:").pack(anchor="w")
        self.noise_slider = ttk.Scale(parent, from_=0.0, to=0.1, orient=tk.HORIZONTAL)
        self.noise_slider.set(0.005)
        self.noise_slider.pack(fill=tk.X, pady=5)
        self.btn_add_noise = ttk.Button(parent, text="Tambahkan Derau Putih", command=self.add_noise_callback,
                                        state=tk.DISABLED)
        self.btn_add_noise.pack(fill=tk.X, pady=5)

        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=10)

        # --- Bagian 3: Terapkan Filter ---
        ttk.Label(parent, text="3. Terapkan Filter Reduksi", font=("Helvetica", 12, "bold")).pack(pady=5, anchor="w")
        self.filter_var = tk.StringVar()
        self.filter_menu = ttk.Combobox(parent, textvariable=self.filter_var, state='readonly')
        self.filter_menu['values'] = ('Low-Pass Filter', 'High-Pass Filter', 'Wiener Filter')
        self.filter_menu.current(0)
        self.filter_menu.pack(fill=tk.X, pady=5)
        self.btn_apply_filter = ttk.Button(parent, text="Terapkan Filter", command=self.apply_filter_callback,
                                           state=tk.DISABLED)
        self.btn_apply_filter.pack(fill=tk.X, pady=5)

        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=10)

        # --- Bagian 4: Kontrol Pemutaran ---
        ttk.Label(parent, text="4. Putar Audio", font=("Helvetica", 12, "bold")).pack(pady=5, anchor="w")
        self.btn_play_orig = ttk.Button(parent, text="Putar Asli", command=lambda: self.play_audio('original'),
                                        state=tk.DISABLED)
        self.btn_play_orig.pack(fill=tk.X, pady=5)
        self.btn_play_noisy = ttk.Button(parent, text="Putar Bising", command=lambda: self.play_audio('noisy'),
                                         state=tk.DISABLED)
        self.btn_play_noisy.pack(fill=tk.X, pady=5)
        self.btn_play_filtered = ttk.Button(parent, text="Putar Hasil Filter",
                                            command=lambda: self.play_audio('filtered'), state=tk.DISABLED)
        self.btn_play_filtered.pack(fill=tk.X, pady=5)

        # --- Label Status ---
        self.status_label = ttk.Label(parent, text="Status: Menunggu file audio...", foreground="blue")
        self.status_label.pack(side=tk.BOTTOM, pady=10)

    def _create_plot_widgets(self, parent):
        """Membuat kanvas untuk menampilkan plot Matplotlib."""
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)

        self.ax1.set_title("Sinyal Asli")
        self.ax2.set_title("Sinyal Bising")
        self.ax3.set_title("Sinyal Hasil Filter")

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def plot_waveform(self, ax, data, sr, title):
        """Fungsi helper untuk menggambar waveform."""
        ax.clear()
        if data is not None and sr is not None:
            librosa.display.waveshow(data, sr=sr, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Amplitudo")
        self.canvas.draw()

    def load_audio(self):
        """Membuka dialog file dan memuat data audio."""
        filepath = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if not filepath:
            return

        try:
            self.original_audio, self.sample_rate = librosa.load(filepath, sr=None)
            self.lbl_filename.config(text=filepath.split('/')[-1])
            self.status_label.config(text="Status: File berhasil dimuat.")

            # Reset plot dan data lain
            self.noisy_audio = None
            self.filtered_audio = None
            self.plot_waveform(self.ax1, self.original_audio, self.sample_rate, "Sinyal Asli")
            self.plot_waveform(self.ax2, None, None, "Sinyal Bising")
            self.plot_waveform(self.ax3, None, None, "Sinyal Hasil Filter")

            # Aktifkan tombol yang relevan
            self.btn_add_noise.config(state=tk.NORMAL)
            self.btn_play_orig.config(state=tk.NORMAL)
            self.btn_apply_filter.config(state=tk.DISABLED)
            self.btn_play_noisy.config(state=tk.DISABLED)
            self.btn_play_filtered.config(state=tk.DISABLED)

        except Exception as e:
            self.status_label.config(text=f"Error: {e}")

    def add_noise_callback(self):
        """Callback untuk tombol 'Tambahkan Derau'."""
        if self.original_audio is None:
            return
        noise_amount = self.noise_slider.get()
        self.noisy_audio = add_noise(self.original_audio, amount=noise_amount)
        self.plot_waveform(self.ax2, self.noisy_audio, self.sample_rate, "Sinyal Bising")

        self.btn_apply_filter.config(state=tk.NORMAL)
        self.btn_play_noisy.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Derau berhasil ditambahkan.")

    def apply_filter_callback(self):
        """Callback untuk tombol 'Terapkan Filter'."""
        if self.noisy_audio is None:
            return

        filter_type = self.filter_var.get()

        if filter_type == 'Low-Pass Filter':
            self.filtered_audio = apply_lowpass_filter(self.noisy_audio, self.sample_rate)
        elif filter_type == 'High-Pass Filter':
            self.filtered_audio = apply_highpass_filter(self.noisy_audio, self.sample_rate)
        elif filter_type == 'Wiener Filter':
            self.filtered_audio = apply_wiener_filter(self.noisy_audio)

        self.plot_waveform(self.ax3, self.filtered_audio, self.sample_rate, f"Hasil {filter_type}")
        self.btn_play_filtered.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Filter berhasil diterapkan.")

    def play_audio(self, audio_type):
        """Memutar audio menggunakan sounddevice."""
        data = None
        if audio_type == 'original':
            data = self.original_audio
        elif audio_type == 'noisy':
            data = self.noisy_audio
        elif audio_type == 'filtered':
            data = self.filtered_audio

        if data is not None:
            # Menghentikan audio yang sedang berjalan
            sd.stop()
            # Memutar audio di thread terpisah agar GUI tidak macet
            threading.Thread(target=sd.play, args=(data, self.sample_rate), daemon=True).start()
            self.status_label.config(text=f"Status: Memutar audio {audio_type}...")


# 3. TITIK MASUK APLIKASI
if __name__ == "__main__":
    app = AudioFilterApp()
    app.mainloop()