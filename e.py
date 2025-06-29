import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from deepface import DeepFace

class ImageQualityCheckerApp:
    def __init__(self, master):
        self.master = master
        master.title("プロフィール画像画質＆表情判定")

        self.image_path = None

        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise IOError("Haar Cascade XMLファイルをロードできませんでした。パスを確認してください。")
        except Exception as e:
            messagebox.showerror("エラー", f"顔検出モデルのロードに失敗しました: {e}\n\n"
                                           "haarcascade_frontalface_default.xml が存在するか確認してください。")
            self.master.destroy()

        self.SMILE_THRESHOLD = 50.0

        self.select_button = tk.Button(master, text="画像を選択", command=self.select_image)
        self.select_button.pack(pady=10)

        self.canvas = tk.Canvas(master, width=400, height=300, bg="lightgray")
        self.canvas.pack(pady=10)

        self.check_button = tk.Button(master, text="画質・表情判定を実行", command=self.run_quality_check)
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

            # --- 1. 顔検出 (OpenCV Haar Cascade) ---
            faces_haar = self.face_cascade.detectMultiScale(
                gray_img,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces_haar) == 0:
                return "NG: 顔が検出されませんでした。顔がはっきりと写っているか確認してください。"
            elif len(faces_haar) > 1:
                return "NG: 複数人の顔が検出されました。プロフィール画像は、ご自身お一人で写っているものにしてください。"

            x, y, w, h = faces_haar[0]
            face_img_roi = img[y:y+h, x:x+w]

            # --- 2. deepface による表情判定と顔パーツ認識度チェック ---
            # 'landmarks' を削除
            # 顔パーツ認識度は、deepfaceが正常に解析できたか（demographiesが空でないか）で判断
            face_parts_successfully_recognized = False # 顔パーツ認識度を初期化
            
            try:
                demographies = DeepFace.analyze(
                    img_path=face_img_roi,
                    actions=['emotion'], # 'landmarks' を削除
                    enforce_detection=False,
                    detector_backend='ssd' # または 'retinaface' 'mtcnn'などを試す
                )

                if not demographies:
                    # deepfaceが何も検出できなかった場合、顔パーツ認識失敗と見なす
                    return "NG: 顔の解析に失敗しました。顔がはっきり写っていないか、極端な角度の可能性があります。"
                
                # ここまで到達すれば、deepfaceは顔を認識し、感情分析できたと判断
                face_parts_successfully_recognized = True

                analysis = demographies[0]
                emotions = analysis['emotion']
                dominant_emotion = analysis['dominant_emotion']

                smile_percentage = emotions.get('happy', 0.0)
                is_smiling = smile_percentage >= self.SMILE_THRESHOLD

            except ValueError as ve:
                # DeepFaceが顔を検出できなかった場合など、顔パーツ認識失敗と見なす
                return f"NG: 顔の表情やパーツの解析に失敗しました。画像が不鮮明か、顔の向きが不適切です。詳細: {ve}"
            except Exception as de_err:
                return f"NG: 表情・パーツ解析中に予期せぬエラーが発生しました。詳細: {de_err}"

            # --- 3. シャープネス（鮮明さ）の評価 (顔領域) ---
            face_roi_gray = gray_img[y:y+h, x:x+w]
            sharpness_score = cv2.Laplacian(face_roi_gray, cv2.CV_64F).var()

            # --- 4. 解像度の評価 ---
            height, width = img.shape[:2]
            min_resolution = 200

            # --- 5. 総合的な判定ロジック ---
            feedback = []
            SHARPNESS_THRESHOLD = 60

            if sharpness_score < SHARPNESS_THRESHOLD:
                feedback.append(f"画像がぼやけているようです (鮮明度スコア: {sharpness_score:.2f})。")

            if width < min_resolution or height < min_resolution:
                feedback.append(f"画像の解像度が低すぎます ({width}x{height})。最低{min_resolution}x{min_resolution}が必要です。")
            
            image_area = width * height
            face_area = w * h
            if face_area / image_area < 0.05:
                 feedback.append(f"顔が画像全体に対して小さすぎます。もっと顔を大きく写してください。")

            # 表情判定のフィードバック
            if is_smiling:
                feedback.append(f"表情は良好です！笑顔度: {smile_percentage:.2f}% (優勢感情: {dominant_emotion})")
            else:
                feedback.append(f"笑顔が不足しているかもしれません。笑顔のほうがマッチングしやすい傾向があります。笑顔度: {smile_percentage:.2f}% (優勢感情: {dominant_emotion})")

            # 顔パーツ認識度のフィードバック (deepfaceが正常に解析できたかで判断)
            if not face_parts_successfully_recognized:
                # このブロックには基本的に到達しないはず（既にNGリターンされているため）
                # しかし、もし何らかの理由で到達した場合に備えて残しておく
                feedback.append("NG: 顔のパーツが十分に認識できませんでした。顔が隠れていないか、照明が適切か確認してください。")
            else:
                feedback.append("顔の主要なパーツは正常に認識できました。") # 成功した旨を伝える

            # 最終的な判定
            # NGフィードバックが含まれるか、笑顔でないか、顔パーツ認識に問題がある場合
            # 顔パーツ認識は`face_parts_successfully_recognized`でカバーされる
            if any("NG:" in f for f in feedback) or not is_smiling: # "NG:" を含むメッセージで判断
                final_status = "NG"
            else:
                final_status = "OK"

            if final_status == "OK":
                return f"OK: プロフィール画像として良好です！\n" + "\n".join(f for f in feedback if not f.startswith("NG:"))
            else:
                # NGメッセージは先頭に"NG:"を付けていることを利用
                ng_messages = [f for f in feedback if f.startswith("NG:")]
                ok_messages = [f for f in feedback if not f.startswith("NG:")]
                
                # 笑顔に関するフィードバックもNGと見なす場合はここに追加
                if not is_smiling:
                    ng_messages.append(f"笑顔が不足しているかもしれません。笑顔のほうがマッチングしやすい傾向があります。笑顔度: {smile_percentage:.2f}%")

                # 顔パーツ認識の成功メッセージはOKの場合のみ表示する
                # それ以外は、エラーメッセージでカバーされるため
                
                return f"NG: プロフィール画像に以下の問題があります。\n" + "\n".join(ng_messages) + \
                       ("\n" + "\n".join(ok_messages) if ok_messages else "")

        except Exception as e:
            return f"判定中に予期せぬエラーが発生しました: {e}"

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageQualityCheckerApp(root)
    root.mainloop()