import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ImageAnalyzerApp:
    def __init__(self, root):
        """Inisialisasi aplikasi utama."""
        self.root = root
        self.root.title("Image Fusion and Zoom Analyzer")
        self.root.geometry("1400x900")  # Ukuran jendela bisa disesuaikan

        # Inisialisasi variabel untuk menyimpan path dan data gambar
        self.image1_path = None
        self.image2_path = None
        self.cv_image1 = None
        self.cv_image2 = None
        self.display_size = (300, 300)  # Ukuran tampilan thumbnail

        # Panggil method untuk membangun UI
        self.setup_ui()

    def setup_ui(self):
        """Membangun semua elemen antarmuka grafis (GUI)."""
        # === Frame Kontrol ===
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(control_frame, text="Load Image 1", command=self.load_image_1).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Load Image 2", command=self.load_image_2).pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Blending (Alpha):").pack(side=tk.LEFT, padx=(20, 5))
        self.alpha_slider = tk.Scale(control_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, length=200,
                                     command=self.process_and_display)
        self.alpha_slider.set(0.5)
        self.alpha_slider.pack(side=tk.LEFT)

        tk.Label(control_frame, text="Faktor Skala:").pack(side=tk.LEFT, padx=(20, 5))
        self.scale_var = tk.StringVar(value="1.5")
        self.scale_entry = tk.Entry(control_frame, textvariable=self.scale_var, width=5)
        self.scale_entry.pack(side=tk.LEFT)
        self.scale_entry.bind("<Return>", self.process_and_display)  # Proses saat menekan Enter

        # === Frame Tampilan Gambar ===
        images_frame = tk.Frame(self.root, padx=10, pady=10)
        images_frame.pack(expand=True, fill=tk.BOTH)

        # Konfigurasi grid agar responsif
        images_frame.grid_columnconfigure(0, weight=1)
        images_frame.grid_columnconfigure(1, weight=1)
        images_frame.grid_rowconfigure(1, weight=1)
        images_frame.grid_rowconfigure(3, weight=1)

        # --- Kolom 1 ---
        tk.Label(images_frame, text="Gambar Asli 1", font=("Helvetica", 12, "bold")).grid(row=0, column=0, pady=5)
        self.img1_label = tk.Label(images_frame, bg="lightgrey", width=self.display_size[0],
                                   height=self.display_size[1])
        self.img1_label.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.fig1, self.ax1 = plt.subplots(figsize=(4, 2))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=images_frame)
        self.canvas1.get_tk_widget().grid(row=2, column=0, padx=5, pady=5)

        tk.Label(images_frame, text="Hasil Penjumlahan (Blended)", font=("Helvetica", 12, "bold")).grid(row=3, column=0,
                                                                                                        pady=5)
        self.blended_label = tk.Label(images_frame, bg="lightgrey", width=self.display_size[0],
                                      height=self.display_size[1])
        self.blended_label.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")

        self.fig3, self.ax3 = plt.subplots(figsize=(4, 2))
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=images_frame)
        self.canvas3.get_tk_widget().grid(row=5, column=0, padx=5, pady=5)

        # --- Kolom 2 ---
        tk.Label(images_frame, text="Gambar Asli 2", font=("Helvetica", 12, "bold")).grid(row=0, column=1, pady=5)
        self.img2_label = tk.Label(images_frame, bg="lightgrey", width=self.display_size[0],
                                   height=self.display_size[1])
        self.img2_label.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        self.fig2, self.ax2 = plt.subplots(figsize=(4, 2))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=images_frame)
        self.canvas2.get_tk_widget().grid(row=2, column=1, padx=5, pady=5)

        tk.Label(images_frame, text="Hasil Perbesaran (Scaled)", font=("Helvetica", 12, "bold")).grid(row=3, column=1,
                                                                                                      pady=5)
        self.scaled_label = tk.Label(images_frame, bg="lightgrey", width=self.display_size[0],
                                     height=self.display_size[1])
        self.scaled_label.grid(row=4, column=1, padx=5, pady=5, sticky="nsew")

        self.fig4, self.ax4 = plt.subplots(figsize=(4, 2))
        self.canvas4 = FigureCanvasTkAgg(self.fig4, master=images_frame)
        self.canvas4.get_tk_widget().grid(row=5, column=1, padx=5, pady=5)

    def load_image_1(self):
        """Membuka dialog file dan memuat gambar pertama."""
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if path:
            self.image1_path = path
            self.cv_image1 = cv2.imread(self.image1_path)
            # Resize jika gambar kedua sudah ada, agar ukurannya sama
            if self.cv_image2 is not None:
                h, w, _ = self.cv_image2.shape
                self.cv_image1 = cv2.resize(self.cv_image1, (w, h))

            self.update_image_display(self.cv_image1, self.img1_label)
            self.update_histogram(self.cv_image1, self.ax1, self.canvas1, "Histogram Asli 1")
            self.process_and_display()  # Langsung proses setelah memuat

    def load_image_2(self):
        """Membuka dialog file dan memuat gambar kedua."""
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if path:
            self.image2_path = path
            self.cv_image2 = cv2.imread(self.image2_path)
            # Resize jika gambar pertama sudah ada, agar ukurannya sama
            if self.cv_image1 is not None:
                h, w, _ = self.cv_image1.shape
                self.cv_image2 = cv2.resize(self.cv_image2, (w, h))

            self.update_image_display(self.cv_image2, self.img2_label)
            self.update_histogram(self.cv_image2, self.ax2, self.canvas2, "Histogram Asli 2")
            self.process_and_display()  # Langsung proses setelah memuat

    def process_and_display(self, event=None):
        """Melakukan operasi blending dan scaling, lalu menampilkan hasilnya."""
        if self.cv_image1 is None or self.cv_image2 is None:
            return

        alpha = self.alpha_slider.get()
        beta = 1.0 - alpha

        # 1. Operasi Penjumlahan (Blending)
        blended_image = cv2.addWeighted(self.cv_image1, alpha, self.cv_image2, beta, 0)
        self.update_image_display(blended_image, self.blended_label)
        self.update_histogram(blended_image, self.ax3, self.canvas3, "Histogram Blended")

        # 2. Operasi Perbesaran (Scaling)
        try:
            scale_factor = float(self.scale_var.get())
            if scale_factor <= 0:
                # Jangan proses jika faktor skala tidak valid
                self.update_image_display(None, self.scaled_label)  # Kosongkan display
                return
        except ValueError:
            return  # Jangan proses jika input bukan angka

        width = int(blended_image.shape[1] * scale_factor)
        height = int(blended_image.shape[0] * scale_factor)
        dim = (width, height)

        # Gunakan INTER_LINEAR untuk hasil yang lebih halus
        scaled_image = cv2.resize(blended_image, dim, interpolation=cv2.INTER_LINEAR)
        self.update_image_display(scaled_image, self.scaled_label)
        self.update_histogram(scaled_image, self.ax4, self.canvas4, "Histogram Scaled")

    def update_image_display(self, cv_image, label_widget):
        """Mengkonversi gambar OpenCV dan menampilkannya di label Tkinter."""
        if cv_image is None:
            # Jika tidak ada gambar, tampilkan area abu-abu
            img_tk = ImageTk.PhotoImage(Image.new('RGB', self.display_size, 'lightgrey'))
            label_widget.config(image=img_tk)
            label_widget.image = img_tk
            return

        # Konversi BGR (OpenCV) ke RGB
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Buat objek Image dari array NumPy
        img_pil = Image.fromarray(img_rgb)
        # Resize untuk thumbnail
        img_pil.thumbnail(self.display_size)
        # Konversi ke format Tkinter
        img_tk = ImageTk.PhotoImage(image=img_pil)

        label_widget.config(image=img_tk)
        label_widget.image = img_tk  # Simpan referensi agar tidak hilang oleh garbage collector

    def update_histogram(self, cv_image, ax, canvas, title):
        """Menghitung dan menggambar ulang histogram pada kanvas Matplotlib."""
        ax.clear()  # Hapus plot lama
        if cv_image is not None:
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                hist = cv2.calcHist([cv_image], [i], None, [256], [0, 256])
                ax.plot(hist, color=col)
            ax.set_xlim([0, 256])

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Intensitas Piksel", fontsize=8)
        ax.set_ylabel("Jumlah Piksel", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnalyzerApp(root)
    root.mainloop()