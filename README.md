# Face Mask Detection on a Single-Board Computer

Real-time application that watches a webcam stream, checks whether each person wears a face mask, estimates the distance between people, and plays an audio warning ("please wear a mask" / "please keep your distance"). It was built as a compact setup for a small single-board computer.

This is the code for my **D3 Computer Engineering final project** at Telkom University:

> *Aplikasi Pendeteksi Otomatis Pemakaian Masker Menggunakan Webcam dengan Metode Viola-Jones Berbasis Small Single Board Computers*

- Book in the Telkom University Open Library: [openlibrary.telkomuniversity.ac.id](https://openlibrary.telkomuniversity.ac.id/home/catalog/id/185907/slug/aplikasi-pendeteksi-otomatis-pemakaian-masker-menggunakan-webcam-dengan-metode-viola-jones-berbasis-small-single-board-computers.html)
- Portfolio page: [ramahrinaldi.id/publications/buku-proyek-akhir-d3-deteksi-masker-viola-jones](https://ramahrinaldi.id/publications/buku-proyek-akhir-d3-deteksi-masker-viola-jones/)

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](#requirements)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)](#requirements)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](#requirements)

## What the code does

1. **Face detection**: an SSD face detector (OpenCV `res10_300x300`) finds faces in each frame.
2. **Mask classification**: a MobileNetV2 classifier (`mask_detector.model`) labels each face as *with mask* or *without mask* and draws a green or red box.
3. **Person detection and distance**: a MobileNet-SSD person detector estimates each person's distance from the box height. People closer than 150 cm are marked high risk, and 150–200 cm medium risk.
4. **Audio alerts**: a voice message plays through `mpg321` when someone has no mask or people are too close.
5. An on-screen counter shows high-risk, medium-risk and detected people. Press `q` to quit.

## Repository layout

```
app.py                       main application
mask_detector.model          MobileNetV2 mask classifier (Keras)
face_detector/               SSD face detector (OpenCV res10)
SSD_MobileNet*.{txt,caffemodel}  MobileNet-SSD person detector (Caffe)
audio/                       alert sounds (mp3)
cek-kamera.sh                lists connected cameras (v4l2)
dataset/                     placeholder, see dataset/README.md
```

## Requirements

- A Linux single-board computer or PC with a USB webcam (developed on a Raspberry Pi, ARM64)
- Python 3.9, `tensorflow==2.7.0`, `opencv-python`, `imutils`, `numpy` (see `requirements.txt`)
- `mpg321` for the audio alerts: `sudo apt install mpg321`

## Run

```bash
pip install -r requirements.txt
sudo apt install mpg321
python app.py
```

Run it from the repository root because the model paths are relative. To use another camera, change `camera = 0` at the top of `app.py`. Use `bash cek-kamera.sh` to list camera devices.

## Credits

This project builds on the open-source **[Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection)** by chandrikadeb7 (MIT License). The face detector, the mask classifier and the training dataset are based on that project. The person-detection and distance logic uses a MobileNet-SSD (Caffe) model. What this version adds: the person and distance risk logic, Indonesian on-screen labels, audio alerts, and the setup for a small single-board computer.

The training images are not redistributed here, see [dataset/README.md](dataset/README.md).

## License

MIT. See [LICENSE](LICENSE). It keeps the copyright notice of the original project.

## Author

**Ramah Rinaldi Ruslan**, Computer Engineering, Telkom University
Portfolio: https://ramahrinaldi.id · GitHub: [@rama907](https://github.com/rama907)

---

### Bahasa Indonesia

Aplikasi yang membaca webcam secara langsung, memeriksa apakah setiap orang memakai masker, memperkirakan jarak antarorang, lalu memutar peringatan suara. Ini adalah kode Proyek Akhir D3 Teknik Komputer Telkom University. Dibangun di atas proyek open source Face-Mask-Detection (MIT), dengan tambahan logika jarak, peringatan suara, dan penyesuaian untuk komputer papan tunggal. Jalankan dengan `python app.py` dari folder root repositori.
