import os
import re
import cv2
import numpy as np
import pickle
import pandas as pd
import logging
import tensorflow as tf
from datetime import datetime
from email import policy
from email.parser import BytesParser
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
from pyzbar.pyzbar import decode
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from contextlib import contextmanager
from typing import Dict, Optional, List, Union
from sklearn.ensemble import RandomForestClassifier

from Phishing_URL_Models.feature_extraction import extract_features
from Phishing_Image_Models.data_loader import preprocess_image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})

# Cấu hình
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./Uploads")
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_PDF_EXTENSION = {"pdf"}
ALLOWED_EMAIL_EXTENSION = {"eml"}
MAX_FILE_SIZE = 10 * 1024 * 1024 

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("server.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

class UnicodeSafeFormatter(logging.Formatter):
    def format(self, record):
        record.msg = record.msg.encode('ascii', errors='replace').decode('ascii')
        return super().format(record)

for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        handler.setFormatter(UnicodeSafeFormatter("%(asctime)s - %(levelname)s - %(message)s"))

logger = logging.getLogger(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Danh sách đặc trưng email
EMAIL_FEATURES = [
    "hops", "missing_subject", "missing_to", "missing_content-type",
    "missing_mime-version", "missing_x-mailer", "missing_delivered-to",
    "missing_list-unsubscribe", "missing_received-spf", "missing_reply-to",
    "str_from_chevron", "str_to_chevron", "str_message-ID_dollar",
    "str_return-path_bounce", "str_content-type_texthtml",
    "domain_match_from_return-path", "domain_match_to_from",
    "domain_match_to_message-id", "domain_match_from_reply-to",
    "domain_match_message-id_from", "length_from", "num_recipients_to",
    "num_recipients_cc", "time_zone", "day_of_week", "span_time",
    "date_comp_date_received", "content-encoding-val", "received_str_forged",
    "number_replies", "label"
]

# Danh sách đặc trưng PDF
EXPECTED_FEATURES = [
    "PdfSize", "MetadataSize", "Pages", "XrefLength", "TitleCharacters",
    "isEncrypted", "EmbeddedFiles", "Images", "Text", "Header", "Obj",
    "Endobj", "Stream", "Endstream", "Xref", "Trailer", "StartXref",
    "PageNo", "Encrypt", "ObjStm", "JS", "Javascript", "AA", "OpenAction",
    "Acroform", "JBIG2Decode", "RichMedia", "Launch", "EmbeddedFile",
    "XFA", "Colors"
]

# Quản lý mô hình
class ModelRegistry:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models = {}
    
    def load_model(self, model_name: str, model_type: str):
        key = f"{model_name}_{model_type}"
        if key not in self.models:
            try:
                model_path = os.path.join(self.model_dir, f"{model_name}.{'pkl' if model_type != 'keras' else 'keras'}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file {model_path} does not exist")
                if model_type == "pickle":
                    model = pickle.load(open(model_path, "rb"))
                    if not isinstance(model, RandomForestClassifier):
                        raise TypeError(f"Loaded object from {model_path} is not a RandomForestClassifier")
                elif model_type == "keras":
                    model = tf.keras.models.load_model(model_path)
                self.models[key] = model
                logger.info(f"Loaded model: {key}")
            except Exception as e:
                logger.error(f"Failed to load model {key}: {e}")
                raise
        return self.models[key]

model_registry = ModelRegistry(MODEL_DIR)

@contextmanager
def temp_file(file, filename: str):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    try:
        file.save(file_path)
        yield file_path
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Cleaned up file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to clean up file {file_path}: {e}")

def allowed_file(filename: str, allowed_extensions: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

def extract_email_features(headers: Union[str, Dict]) -> pd.DataFrame:
    if isinstance(headers, str):
        try:
            with open(headers, 'rb') as f:
                msg = BytesParser(policy=policy.default).parse(f)
            headers_dict = dict(msg.items())
            full_msg = msg
        except Exception as e:
            logger.error(f"Failed to parse .eml file {headers}: {e}")
            raise ValueError(f"Invalid .eml file: {e}")
    else:
        headers_dict = headers
        full_msg = None

    sanitized_headers = {k: v.encode('ascii', errors='replace').decode('ascii') for k, v in headers_dict.items()}
    logger.info(f"Extracted headers: {sanitized_headers}")

    received_headers = full_msg.get_all("Received", []) if full_msg else []
    hops = len(received_headers)
    most_recent_received = received_headers[0] if received_headers else ""

    is_missing = lambda h: int(h not in headers_dict or not headers_dict[h].strip())
    contains = lambda h, pattern: int(bool(re.search(pattern, headers_dict.get(h, ''), re.IGNORECASE)))
    domain_match = lambda h1, h2_val: int(
        bool(re.search(r'@([\w\.-]+)', headers_dict.get(h1, ''))) and
        bool(re.search(r'@([\w\.-]+)', h2_val)) and
        re.search(r'@([\w\.-]+)', headers_dict.get(h1, '')).group(1) ==
        re.search(r'@([\w\.-]+)', h2_val).group(1)
    )

    span_time = 0
    if "Date" in headers_dict and most_recent_received:
        try:
            dt = datetime.strptime(headers_dict["Date"][:31], "%a, %d %b %Y %H:%M:%S %z")
            rec = most_recent_received.split(";")[-1].strip()
            if rec:
                rec_dt = datetime.strptime(rec[:31], "%a, %d %b %Y %H:%M:%S %z")
                span_time = abs((dt - rec_dt).total_seconds())
        except (ValueError, IndexError):
            logger.debug("Failed to parse Date or Received for span_time")

    time_zone = int(bool(re.search(r'([+-]\d{4})', headers_dict.get("Date", ""))))
    day_of_week = 0
    if "Date" in headers_dict:
        try:
            dt = datetime.strptime(headers_dict["Date"][:31], "%a, %d %b %Y %H:%M:%S %z")
            day_of_week = dt.weekday()
        except ValueError:
            logger.debug("Failed to parse Date for day_of_week")

    date_comp_date_received = int("Date" in headers_dict and "Received" in headers_dict)
    content_encoding_val = headers_dict.get("Content-Transfer-Encoding", "").lower()
    if not content_encoding_val:
        content_encoding_val = 0
    elif "quoted-printable" in content_encoding_val:
        content_encoding_val = 1
    elif "base64" in content_encoding_val:
        content_encoding_val = 2
    elif "7bit" in content_encoding_val or "8bit" in content_encoding_val:
        content_encoding_val = 3
    else:
        content_encoding_val = 4

    received_str_forged = 0
    for rec in received_headers:
        if "forged" in rec.lower() or not re.search(r'from [\w\.-]+', rec):
            received_str_forged = 1
            break

    str_from_chevron = int(bool(re.search(r'<[\w\.-]+@[\w\.-]+>', headers_dict.get("From", ""))))
    str_to_chevron = int(bool(re.search(r'<[\w\.-]+@[\w\.-]+>', headers_dict.get("To", ""))))
    length_from = len(headers_dict.get("From", ""))
    num_recipients_to = len([x for x in headers_dict.get("To", "").split(",") if x.strip()])
    num_recipients_cc = len([x for x in headers_dict.get("Cc", "").split(",") if x.strip()])
    missing_x_mailer = is_missing("X-Mailer")
    missing_reply_to = is_missing("Reply-To")
    str_message_id_dollar = int(bool(re.search(r'\$', headers_dict.get("Message-ID", ""))))
    str_return_path_bounce = int(bool(re.search(r'bounce', headers_dict.get("Return-Path", ""), re.IGNORECASE)))
    str_content_type_texthtml = int(bool(re.search(r'text/html', headers_dict.get("Content-Type", ""), re.IGNORECASE)))
    domain_match_to_from = domain_match("To", headers_dict.get("From", ""))
    domain_match_to_message_id = domain_match("To", headers_dict.get("Message-ID", ""))
    domain_match_from_reply_to = domain_match("From", headers_dict.get("Reply-To", ""))
    domain_match_message_id_from = domain_match("Message-ID", headers_dict.get("From", ""))
    number_replies = len([h for h in headers_dict.get("References", "").split() if h.strip()]) if "References" in headers_dict else 0

    feats = {
        "hops": hops,
        "missing_subject": is_missing("Subject"),
        "missing_to": is_missing("To"),
        "missing_content-type": is_missing("Content-Type"),
        "missing_mime-version": is_missing("MIME-Version"),
        "missing_x-mailer": missing_x_mailer,
        "missing_delivered-to": is_missing("Delivered-To"),
        "missing_list-unsubscribe": is_missing("List-Unsubscribe"),
        "missing_received-spf": is_missing("Received-SPF"),
        "missing_reply-to": missing_reply_to,
        "str_from_chevron": str_from_chevron,
        "str_to_chevron": str_to_chevron,
        "str_message-ID_dollar": str_message_id_dollar,
        "str_return-path_bounce": str_return_path_bounce,
        "str_content-type_texthtml": str_content_type_texthtml,
        "domain_match_from_return-path": domain_match("From", headers_dict.get("Return-Path", "")),
        "domain_match_to_from": domain_match_to_from,
        "domain_match_to_message-id": domain_match_to_message_id,
        "domain_match_from_reply-to": domain_match_from_reply_to,
        "domain_match_message-id_from": domain_match_message_id_from,
        "length_from": length_from,
        "num_recipients_to": num_recipients_to,
        "num_recipients_cc": num_recipients_cc,
        "time_zone": time_zone,
        "day_of_week": day_of_week,
        "span_time": span_time,
        "date_comp_date_received": date_comp_date_received,
        "content-encoding-val": content_encoding_val,
        "received_str_forged": received_str_forged,
        "number_replies": number_replies,
        "label": 0
    }

    df = pd.DataFrame([[feats[f] for f in EMAIL_FEATURES if f != "label"]], 
                     columns=[f for f in EMAIL_FEATURES if f != "label"])
    logger.info(f"Extracted email features: {df.to_dict(orient='records')[0]}")
    return df

def extract_pdf_features(pdf_path: str) -> Dict:
    try:
        pdf = PdfReader(pdf_path)
        features = {feat: 0 for feat in EXPECTED_FEATURES}
        features["PdfSize"]       = os.path.getsize(pdf_path) / 1024.0
        features["MetadataSize"]  = len(str(pdf.metadata)) if pdf.metadata else 0
        features["Pages"]         = len(pdf.pages)
        features["PageNo"]        = len(pdf.pages)
        features["TitleCharacters"] = len(pdf.metadata.get("/Title", "")) if pdf.metadata else 0
        features["isEncrypted"]   = 1 if pdf.is_encrypted else 0

        with open(pdf_path, "rb") as f:
            content = f.read()

        features["Header"] = content.count(b"%PDF")
        xref_sections = re.findall(b"xref(.*?)trailer", content, re.DOTALL)
        features["XrefLength"] = sum(len(sec) for sec in xref_sections)
        features["Obj"]        = content.count(b" obj")
        features["Endobj"]     = content.count(b"endobj")
        features["Stream"]     = content.count(b"stream")
        features["Endstream"]  = content.count(b"endstream")
        features["Xref"]       = content.count(b"xref")
        features["Trailer"]    = content.count(b"trailer")
        features["StartXref"]  = content.count(b"startxref")
        features["Encrypt"]    = content.count(b"/Encrypt")
        features["ObjStm"]     = content.count(b"/ObjStm")
        features["AA"]         = content.count(b"/AA")
        features["OpenAction"] = content.count(b"/OpenAction")
        features["Acroform"]   = content.count(b"/AcroForm")
        features["JBIG2Decode"] = content.count(b"JBIG2Decode")
        features["RichMedia"]  = content.count(b"/RichMedia")
        features["Launch"]     = content.count(b"/Launch")
        features["EmbeddedFile"] = content.count(b"/EmbeddedFile")
        features["EmbeddedFiles"] = content.count(b"/EmbeddedFiles")
        features["XFA"]        = content.count(b"/XFA")
        features["Colors"]     = content.count(b"/Color")

        try:
            text = extract_text(pdf_path, laparams=LAParams())
            features["Text"]       = len(text) if text else 0
            features["JS"]         = 1 if "javascript" in text.lower() else 0
            features["Javascript"] = features["JS"]
        except Exception as e:
            logger.warning(f"Failed to extract text from PDF {pdf_path}: {e}")
            features["Text"] = 0
            features["JS"] = features["Javascript"] = 0

        features["Images"] = 1 if features["Pages"] > 0 and (features["PdfSize"] / features["Pages"]) > 50 else 0
        return features
    except Exception as e:
        logger.error(f"Error extracting PDF features from {pdf_path}: {e}")
        raise ValueError(f"Failed to process PDF: {e}")

def compute_ensemble_score(rf_prob: float, cnn_prob: Optional[float] = None) -> float:
    if cnn_prob is not None:
        return (rf_prob + cnn_prob) / 2
    return rf_prob

def preprocess_image_for_cnn(image: np.ndarray, img_size: tuple = (128, 128)) -> np.ndarray:
    try:
        image = cv2.resize(image, img_size)
        image = image.astype("float32")
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    threshold = 0.5
    try:
        if "file" in request.files and request.files["file"].filename:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No selected file"}), 400

            filename = secure_filename(file.filename)
            with temp_file(file, filename) as file_path:
                if allowed_file(filename, ALLOWED_EMAIL_EXTENSION):
                    try:
                        df = extract_email_features(file_path)
                        rf_model = model_registry.load_model("random_forest_email", "pickle")
                        rf_p = rf_model.predict_proba(df)[:, 1][0]
                        ensemble = compute_ensemble_score(rf_p)
                        result = "Phishing" if ensemble > threshold else "Legitimate"
                        logger.info(f"Prediction for {filename}: rf_confidence={rf_p:.4f}, result={result}")
                        return jsonify({
                            "rf_confidence": round(float(rf_p), 4),
                            "result": result
                        }), 200
                    except Exception as e:
                        logger.error(f"Error processing .eml file {filename}: {e}")
                        return jsonify({"error": f"Failed to process .eml file: {str(e)}"}), 400

                elif allowed_file(filename, ALLOWED_IMAGE_EXTENSIONS):
                    # Load ảnh
                    image = cv2.imread(file_path)
                    if image is None:
                        logger.error(f"Failed to load image: {file_path}")
                        return jsonify({"error": "Invalid or corrupted image file"}), 400

                    # Kiểm tra QR code
                    qr_codes = decode(image)
                    if qr_codes:
                        qr_results = []
                        for qr in qr_codes:
                            url = qr.data.decode("utf-8")
                            logger.info(f"Detected QR code: {url}")
                            rf_features = pd.DataFrame([extract_features(url)])
                            rf_pred = model_registry.load_model("random_forest_URL", "pickle").predict_proba(rf_features)[:, 1][0]
                            ensemble = compute_ensemble_score(rf_pred)
                            res = "Phishing" if ensemble > threshold else "Legitimate"
                            qr_results.append({
                                "qr_url": url,
                                "rf_confidence": round(float(rf_pred), 4),
                                "result": res
                            })
                        return jsonify({"qr_results": qr_results}), 200

                    # Dự đoán bằng CNN
                    try:
                        cnn_input = preprocess_image_for_cnn(image)
                        cnn_model = model_registry.load_model("cnn_phishing_image", "keras")
                        cnn_pred = float(cnn_model.predict(cnn_input, verbose=0)[0][0])
                        result = "Phishing" if cnn_pred > threshold else "Legitimate"
                        logger.info(f"Image prediction for {filename}: cnn_confidence={cnn_pred:.4f}, result={result}, threshold={threshold}")
                        return jsonify({
                            "cnn_confidence": round(float(cnn_pred), 4),
                            "result": result,
                            "filename": filename,
                            "threshold_used": threshold
                        }), 200
                    except Exception as e:
                        logger.error(f"CNN prediction failed for {filename}: {e}")
                        return jsonify({"error": f"CNN prediction failed: {str(e)}"}), 400

                elif allowed_file(filename, ALLOWED_PDF_EXTENSION):
                    features = extract_pdf_features(file_path)
                    feature_df = pd.DataFrame([features], columns=EXPECTED_FEATURES)
                    rf_prob = model_registry.load_model("random_forest_file", "pickle").predict_proba(feature_df)[0][1]
                    ensemble = compute_ensemble_score(rf_prob)
                    res = "Phishing" if ensemble > threshold else "Legitimate"
                    return jsonify({
                        "rf_confidence": round(float(rf_prob), 4),
                        "result": res
                    }), 200

                else:
                    return jsonify({"error": "Invalid file type"}), 400

        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No valid input data"}), 400

        if "url" in data:
            url = data["url"].strip()
            if not url:
                return jsonify({"error": "URL is required"}), 400
            rf_features = pd.DataFrame([extract_features(url)])
            rf_pred = model_registry.load_model("random_forest_URL", "pickle").predict_proba(rf_features)[:, 1][0]
            ensemble = compute_ensemble_score(rf_pred)
            res = "Phishing" if ensemble > threshold else "Legitimate"
            return jsonify({
                "rf_confidence": round(float(rf_pred), 4),
                "url": url,
                "result": res
            }), 200

        if "text" in data:
            text = data["text"].strip()
            if not text:
                return jsonify({"error": "Text is required"}), 400
            rf_features = pd.DataFrame([extract_features(text)])
            rf_pred = model_registry.load_model("random_forest_URL", "pickle").predict_proba(rf_features)[:, 1][0]
            ensemble = compute_ensemble_score(rf_pred)
            res = "Phishing" if ensemble > threshold else "Legitimate"
            return jsonify({
                "rf_confidence": round(float(rf_pred), 4),
                "result": res
            }), 200

        return jsonify({"error": "Invalid request format"}), 400

    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)