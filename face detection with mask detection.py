import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from deepface import DeepFace
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class ImageQualityCheckerApp:
    def __init__(self, master):
        self.master = master
        master.title("プロフィール画像画質＆表情＆マスク判定")

        self.image_path = None

        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise IOError("Haar Cascade XMLファイルをロードできませんでした。パスを確認してください。")
        except Exception as e:
            messagebox.showerror("エラー", f"顔検出モデルのロードに失敗しました: {e}\n\n"
                                           "haarcascade_frontalface_default.xml が存在するか確認してください。")
            self.master.destroy()

        self.mask_model_path = 'densenet121_detection_model.h5'
        try:
            self.mask_model = tf.keras.models.load_model(self.mask_model_path)
        except Exception as e:
            messagebox.showerror("エラー", f"マスク検出モデルのロードに失敗しました: {e}\n\n"
                                           f"'{self.mask_model_path}' が存在し、破損していないか確認してください。"
                                           "\nKaggleノートブックを実行してモデルを保存・ダウンロードしてください。")
            self.master.destroy()

        self.SMILE_THRESHOLD = 50.0

        self.select_button = tk.Button(master, text="画像を選択", command=self.select_image)
        self.select_button.pack(pady=10)

        self.canvas = tk.Canvas(master, width=400, height=300, bg="lightgray")
        self.canvas.pack(pady=10)

        self.check_button = tk.Button(master, text="画質・表情・マスク判定を実行", command=self.run_quality_check)
        self.check_button.pack(pady=10)

        self.result_label = tk.Label(master, text="ここに判定結果が表示されます", font=("Helvetica", 12))
        self.result_label.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.result_label.config(text="画像が選択されました。判定ボタンを押してください。")

    def display_image(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((400, 300))
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(200, 150, image=self.photo, anchor=tk.CENTER)
        except Exception as e:
            messagebox.showerror("エラー", f"画像の表示に失敗しました: {e}")

    def run_quality_check(self):
        if not self.image_path:
            messagebox.showwarning("警告", "画像を先に選択してください。")
            return

        self.result_label.config(text="判定中...しばらくお待ちください。")
        self.master.update_idletasks()

        result_message = self.perform_quality_check(self.image_path)
        self.result_label.config(text=f"判定結果: {result_message}")

    def perform_quality_check(self, image_path):
        try:
            nparr = np.fromfile(image_path, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return "エラー: 画像を読み込めませんでした。ファイルが破損しているか、対応していない形式です。"

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # --- 1. 顔検出 (マスク顔も検出できるようにパラメータを調整) ---
            # minNeighborsを小さくすると誤検出が増える可能性があるが、マスク顔も検出されやすくなる
            # minSizeも小さめに設定し、小さな顔も検出対象にする
            faces_haar_initial_detection = self.face_cascade.detectMultiScale(
                gray_img,
                scaleFactor=1.1,
                minNeighbors=3, # 検出を少し緩やかにする
                minSize=(20, 20) # 最小サイズも小さくする
            )

            # --- 複数顔検出の一次チェックと、最も大きい顔の選定 ---
            # マスク検出のために、最も大きい顔を一つ選ぶ
            if len(faces_haar_initial_detection) == 0:
                # そもそも顔が全く検出されない場合は、ここでNG
                return "NG: 顔が検出されませんでした。顔がはっきりと写っているか確認してください。"
            
            # 最も大きい顔を選ぶ（複数検出されても一旦処理を進めるため）
            # 後で最終的な「複数人の顔」判定を行う
            x, y, w, h = sorted(faces_haar_initial_detection, key=lambda f: f[2] * f[3], reverse=True)[0]
            face_img_roi = img[y:y+h, x:x+w]

            # --- 2. マスク検出の実行（最優先でチェック） ---
            MASK_MODEL_INPUT_SIZE = (224, 224)

            face_for_mask_detection = cv2.resize(face_img_roi, MASK_MODEL_INPUT_SIZE)
            face_for_mask_detection = cv2.cvtColor(face_for_mask_detection, cv2.COLOR_BGR2RGB)
            face_for_mask_detection = preprocess_input(face_for_mask_detection)
            face_for_mask_detection = np.expand_dims(face_for_mask_detection, axis=0)

            mask_prediction_raw = self.mask_model.predict(face_for_mask_detection)[0]
            
            # ノートブックのクラス順序が 'with_mask' (0), 'without_mask' (1) の場合
            mask_confidence = mask_prediction_raw[1]
            #no_mask_confidence = mask_prediction_raw[1] # 現時点では使用しないが、参考として

            MASK_CONFIDENCE_THRESHOLD = 0.9 
            is_wearing_mask = mask_confidence > MASK_CONFIDENCE_THRESHOLD

            # マスクを着用している場合は、ここでNGを確定して終了
            if is_wearing_mask:
                return f"NG: マスクを着用しています。プロフィール画像は顔全体が見えるものにしてください (マスク信頼度: {mask_confidence:.2f})。"

            # --- ここから、マスクをしていない（と判断された）顔のチェック ---

            # --- 3. 複数人の顔検出の最終チェック ---
            # マスクがないことが確認された上で、改めて顔の数を厳しくチェック
            if len(faces_haar_initial_detection) > 1:
                return "NG: 複数人の顔が検出されました。プロフィール画像は、ご自身お一人で写っているものにしてください。"


            # --- 4. deepface による表情判定と顔パーツ認識度チェック ---
            # ここからは、すでに単一のマスクなし顔が検出されていることを前提とする
            face_parts_successfully_recognized = False 
            
            try:
                demographies = DeepFace.analyze(
                    img_path=face_img_roi, # マスクがなければここでdeepfaceも検出できるはず
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='ssd' 
                )

                if not demographies:
                    return "NG: 顔の解析に失敗しました。顔がはっきり写っていないか、極端な角度の可能性があります。"
                
                face_parts_successfully_recognized = True

                analysis = demographies[0]
                emotions = analysis['emotion']
                dominant_emotion = analysis['dominant_emotion']

                smile_percentage = emotions.get('happy', 0.0)
                is_smiling = smile_percentage >= self.SMILE_THRESHOLD

            except ValueError as ve:
                return f"NG: 顔の表情やパーツの解析に失敗しました。画像が不鮮明か、顔の向きが不適切です。詳細: {ve}"
            except Exception as de_err:
                return f"NG: 表情・パーツ解析中に予期せぬエラーが発生しました。詳細: {de_err}"


            # --- 5. シャープネス（鮮明さ）の評価 (顔領域) ---
            face_roi_gray = cv2.cvtColor(face_img_roi, cv2.COLOR_BGR2GRAY)
            sharpness_score = cv2.Laplacian(face_roi_gray, cv2.CV_64F).var()

            # --- 6. 解像度の評価 ---
            height, width = img.shape[:2]
            min_resolution = 200

            # --- 7. 総合的な判定ロジック ---
            feedback = []
            SHARPNESS_THRESHOLD = 60

            # マスク判定はすでに上で処理済みなので、ここでは含めない

            if sharpness_score < SHARPNESS_THRESHOLD:
                feedback.append(f"画像がぼやけているようです (鮮明度スコア: {sharpness_score:.2f})。")

            if width < min_resolution or height < min_resolution:
                feedback.append(f"画像の解像度が低すぎます ({width}x{height})。最低{min_resolution}x{min_resolution}が必要です。")
            
            image_area = width * height
            face_area = w * h
            if face_area / image_area < 0.05:
                 feedback.append(f"顔が画像全体に対して小さすぎます。もっと顔を大きく写してください。")

            if is_smiling:
                feedback.append(f"表情は良好です！笑顔度: {smile_percentage:.2f}% (優勢感情: {dominant_emotion})")
            else:
                feedback.append(f"笑顔が不足しているかもしれません。笑顔のほうがマッチングしやすい傾向があります。笑顔度: {smile_percentage:.2f}% (優勢感情: {dominant_emotion})")

            if not face_parts_successfully_recognized:
                feedback.append("NG: 顔のパーツが十分に認識できませんでした。顔が隠れていないか、照明が適切か確認してください。")
            else:
                feedback.append("顔の主要なパーツは正常に認識できました。")
            
            # 最終的な判定
            if any("NG:" in f for f in feedback) or not is_smiling:
                final_status = "NG"
            else:
                final_status = "OK"

            if final_status == "OK":
                return f"OK: プロフィール画像として良好です！\n" + "\n".join(f for f in feedback if not f.startswith("NG:"))
            else:
                ng_messages = [f for f in feedback if f.startswith("NG:")]
                ok_messages = [f for f in feedback if not f.startswith("NG:")]
                
                if not is_smiling and not any("笑顔が不足している" in msg for msg in ng_messages):
                    ng_messages.append(f"笑顔が不足しているかもしれません。笑顔のほうがマッチングしやすい傾向があります。笑顔度: {smile_percentage:.2f}%")

                return f"NG: プロフィール画像に以下の問題があります。\n" + "\n".join(ng_messages) + \
                       ("\n" + "\n".join(ok_messages) if ok_messages else "")

        except Exception as e:
            return f"判定中に予期せぬエラーが発生しました: {e}"

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageQualityCheckerApp(root)
    root.mainloop()