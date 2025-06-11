# v2
import os
import cv2
import numpy as np
import time
import pickle
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["DATASET_FOLDER"] = "dataset"
app.config["MODEL_FOLDER"] = "static/models"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

# Ensure directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["DATASET_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)

# Initialize face detector and recognizer
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
le = None


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def count_total_images(dataset_path):
    total = 0
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            total += len(
                [
                    f
                    for f in os.listdir(folder_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
    return total


def load_dataset(dataset_path):
    faces = []
    labels = []
    label_names = []
    total_images = count_total_images(dataset_path)

    if total_images == 0:
        return faces, labels, label_names

    processed_count = 0
    start_time = time.time()

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            label = folder_name.replace("pins_", "")
            label_names.append(label)
            for image_name in os.listdir(folder_path):
                if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                processed_count += 1
                image_path = os.path.join(folder_path, image_name)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces_rect = face_detector.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5
                    )
                    for x, y, w, h in faces_rect:
                        face_roi = gray[y : y + h, x : x + w]
                        faces.append(face_roi)
                        labels.append(label)
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    continue

    return faces, labels, list(set(label_names))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    if "dataset" not in request.files:
        return jsonify({"error": "No dataset folder selected"}), 400

    dataset_files = request.files.getlist("dataset")
    person_name = request.form.get("person_name", "unknown")

    if not person_name or person_name.lower() == "unknown":
        return jsonify({"error": "Please provide a valid person name"}), 400

    person_folder = os.path.join(app.config["DATASET_FOLDER"], f"pins_{person_name}")
    os.makedirs(person_folder, exist_ok=True)

    saved_count = 0
    for file in dataset_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(person_folder, filename))
            saved_count += 1

    return jsonify(
        {
            "message": f"Successfully uploaded {saved_count} images for {person_name}",
            "person_name": person_name,
            "image_count": saved_count,
        }
    )


@app.route("/train_model", methods=["POST"])
def train_model():
    global face_recognizer, le

    try:
        is_update = request.form.get("is_update", "false").lower() == "true"

        if is_update and (
            not os.path.exists(
                os.path.join(app.config["MODEL_FOLDER"], "face_recognizer_model.yml")
            )
            or not os.path.exists(
                os.path.join(app.config["MODEL_FOLDER"], "label_encoder.pkl")
            )
        ):
            return (
                jsonify({"error": "No existing model found. Train a new model first."}),
                400,
            )

        faces, labels, label_names = load_dataset(app.config["DATASET_FOLDER"])
        if len(faces) == 0:
            return jsonify({"error": "No faces found in dataset to train."}), 400

        start_time = time.time()

        if is_update:
            # Load existing model first
            face_recognizer.read(
                os.path.join(app.config["MODEL_FOLDER"], "face_recognizer_model.yml")
            )
            with open(
                os.path.join(app.config["MODEL_FOLDER"], "label_encoder.pkl"), "rb"
            ) as f:
                le = pickle.load(f)

            labels_encoded = le.transform(labels)
            face_recognizer.update(faces, np.array(labels_encoded))
            msg = f"Successfully added {len(faces)} new faces."
        else:
            # New training
            le = LabelEncoder()
            labels_encoded = le.fit_transform(labels)
            face_recognizer.train(faces, np.array(labels_encoded))
            msg = f"Trained on {len(faces)} faces of {len(label_names)} people."

        # Save model and encoder
        face_recognizer.save(
            os.path.join(app.config["MODEL_FOLDER"], "face_recognizer_model.yml")
        )
        with open(
            os.path.join(app.config["MODEL_FOLDER"], "label_encoder.pkl"), "wb"
        ) as f:
            pickle.dump(le, f)

        training_time = time.time() - start_time

        return jsonify(
            {
                "success": True,
                "message": msg,
                "training_time": f"{training_time:.2f} seconds",
                "faces_count": len(faces),
                "people_count": len(label_names),
            }
        )

    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500


@app.route("/recognize", methods=["POST"])
def recognize():
    global le

    # Pastikan model sudah dimuat terlebih dahulu
    if face_recognizer is None or le is None:
        try:
            # Coba muat model secara otomatis jika belum ada
            model_path = os.path.join(
                app.config["MODEL_FOLDER"], "face_recognizer_model.yml"
            )
            encoder_path = os.path.join(app.config["MODEL_FOLDER"], "label_encoder.pkl")
            if not os.path.exists(model_path) or not os.path.exists(encoder_path):
                return jsonify({"error": "Please train a model first."}), 400
            face_recognizer.read(model_path)
            with open(encoder_path, "rb") as f:
                le = pickle.load(f)
        except Exception as e:
            return (
                jsonify(
                    {"error": f"Model not loaded and failed to autoload: {str(e)}"}
                ),
                400,
            )

    if "image" not in request.files:
        return jsonify({"error": "No image selected"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        original_filepath = os.path.join(app.config["UPLOAD_FOLDER"], original_filename)
        file.save(original_filepath)

        try:
            start_time = time.time()
            image = cv2.imread(original_filepath)
            if image is None:
                return jsonify({"error": "Failed to load image"}), 400

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces_rect = face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5
            )

            results = []
            annotated_image = image.copy()

            if len(faces_rect) == 0:
                processing_time = time.time() - start_time
                return jsonify(
                    {
                        "message": "No faces detected in the image.",
                        "processing_time": f"{processing_time:.2f} seconds",
                        "faces_count": 0,
                        "original_image": original_filename,
                        "annotated_image": None,
                    }
                )

            for x, y, w, h in faces_rect:
                face_roi = gray[y : y + h, x : x + w]

                label, confidence = face_recognizer.predict(face_roi)

                # LBPH confidence: 0 is a perfect match.
                # Kita bisa set threshold, misalnya di bawah 100 dianggap dikenali.
                if confidence < 100:
                    predicted_name = le.inverse_transform([label])[0]
                    # Konversi 'distance' confidence menjadi persentase kemiripan
                    confidence_percent = max(0, 100 - confidence)
                    confidence_text = f"{confidence_percent:.2f}%"

                    result = {
                        "person": predicted_name,
                        "confidence": confidence_text,
                        "is_recognized": True,
                    }

                    cv2.rectangle(
                        annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2
                    )

                    # Menampilkan teks nama dan persentase di atas kotak
                    text_to_display = f"{predicted_name} ({confidence_text})"
                    cv2.putText(
                        annotated_image,
                        text_to_display,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,  # Ukuran font sedikit dikecilkan agar tidak tumpang tindih
                        (0, 255, 0),
                        2,
                    )
                else:
                    result = {
                        "person": "Unknown",
                        "confidence": "0%",
                        "is_recognized": False,
                    }
                    cv2.rectangle(
                        annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 2
                    )
                    cv2.putText(
                        annotated_image,
                        "Unknown",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                    )
                results.append(result)

            # Simpan gambar yang sudah ditandai
            annotated_filename = f"annotated_{original_filename}"
            annotated_path = os.path.join(
                app.config["UPLOAD_FOLDER"], annotated_filename
            )
            cv2.imwrite(annotated_path, annotated_image)

            processing_time = time.time() - start_time

            return jsonify(
                {
                    "success": True,
                    "results": results,
                    "original_image": original_filename,
                    "annotated_image": annotated_filename,
                    "processing_time": f"{processing_time:.2f} seconds",
                    "faces_count": len(faces_rect),
                }
            )

        except Exception as e:
            return jsonify({"error": f"Recognition failed: {str(e)}"}), 500

    return jsonify({"error": "Invalid file type"}), 400


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/load_model", methods=["POST"])
def load_model():
    global face_recognizer, le

    try:
        model_path = os.path.join(
            app.config["MODEL_FOLDER"], "face_recognizer_model.yml"
        )
        encoder_path = os.path.join(app.config["MODEL_FOLDER"], "label_encoder.pkl")

        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            return jsonify({"error": "Model files not found"}), 404

        face_recognizer.read(model_path)
        with open(encoder_path, "rb") as f:
            le = pickle.load(f)

        return jsonify({"success": True, "message": "Model loaded successfully"})

    except Exception as e:
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
