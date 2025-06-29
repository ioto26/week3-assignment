import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2 # OpenCVをインポート
import numpy as np # 画像操作でNumPyが必要になる場合があるため

class ImageQualityCheckerApp:
    def __init__(self, master):
        self.master = master
        master.title("プロフィール画像画質判定")

        self.image_path = None # 選択された画像のパスを保持

        # Haar Cascade 分類器のロード
        # OpenCVのデータパスからXMLファイルを読み込みます
        # お使いのOpenCVのインストール状況によってパスが異なる場合があります
        # 一般的には以下のパスでアクセスできます
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise IOError("Haar Cascade XMLファイルをロードできませんでした。パスを確認してください。")
        except Exception as e:
            messagebox.showerror("エラー", f"顔検出モデルのロードに失敗しました: {e}\n\n"
                                           "haarcascade_frontalface_default.xml が存在するか確認してください。")
            self.master.destroy() # エラー時はアプリケーションを終了

        # --- ウィジェットの作成 ---
        # ファイル選択ボタン
        self.select_button = tk.Button(master, text="画像を選択", command=self.select_image)
        self.select_button.pack(pady=10)

        # 画像表示用のキャンバス
        self.canvas = tk.Canvas(master, width=400, height=300, bg="lightgray")
        self.canvas.pack(pady=10)

        # 判定実行ボタン
        self.check_button = tk.Button(master, text="画質判定を実行", command=self.run_quality_check)
        self.check_button.pack(pady=10)

        # 結果表示ラベル
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
            img.thumbnail((400, 300)) # キャンバスサイズに合わせてリサイズ
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(200, 150, image=self.photo, anchor=tk.CENTER)
        except Exception as e:
            messagebox.showerror("エラー", f"画像の表示に失敗しました: {e}")

    def run_quality_check(self):
        if not self.image_path:
            messagebox.showwarning("警告", "画像を先に選択してください。")
            return

        # ここで実際の画質判定ロジックを呼び出す
        result_message = self.perform_quality_check(self.image_path)
        self.result_label.config(text=f"判定結果: {result_message}")

    def perform_quality_check(self, image_path):
        """
        プロフィール画像の画質（顔の判別しやすさ）を判定するロジック。
        """
        try:
            # 画像の読み込み
            # OpenCVは日本語パスに弱いため、numpyで読み込むのが安全
            nparr = np.fromfile(image_path, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return "エラー: 画像を読み込めませんでした。ファイルが破損しているか、対応していない形式です。"

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # --- 1. 顔検出 ---
            faces = self.face_cascade.detectMultiScale(
                gray_img,
                scaleFactor=1.1, # 縮小スケールファクター
                minNeighbors=5,  # 候補矩形が保持すべき最低限の隣接数
                minSize=(30, 30) # 検出されるオブジェクトの最小サイズ
            )
            
            if len(faces) == 0:
                return "NG: 顔が検出されませんでした。顔がはっきりと写っているか確認してください。"
            elif len(faces) > 1:
                return "NG: 複数人の顔が検出されました。プロフィール画像は、ご自身お一人で写っているものにしてください。"

            # --- 2. シャープネス（鮮明さ）の評価 ---
            # 顔が検出された領域でシャープネスを評価する
            # 複数の顔が検出された場合は、最も大きい顔の領域を使用する
            
            x, y, w, h = faces[0]

            # 顔の領域を切り抜く
            face_roi = gray_img[y:y+h, x:x+w]

            # ラプラシアン変換の分散でシャープネスを評価
            # 分散が大きいほどシャープ
            sharpness_score = cv2.Laplacian(face_roi, cv2.CV_64F).var()

            # --- 3. 解像度の評価 ---
            height, width = img.shape[:2]
            # 例として、最低解像度を200x200とする
            min_resolution = 200

            # --- 4. 総合的な判定ロジック ---
            feedback = []

            # シャープネスの閾値 (調整が必要な可能性あり)
            # この値は画像や環境によって調整してください
            # 一般的に、この値より低いとぼやけている可能性が高い
            SHARPNESS_THRESHOLD = 50 

            if sharpness_score < SHARPNESS_THRESHOLD:
                feedback.append(f"画像がぼやけているようです (鮮明度スコア: {sharpness_score:.2f})。")

            if width < min_resolution or height < min_resolution:
                feedback.append(f"画像の解像度が低すぎます ({width}x{height})。最低{min_resolution}x{min_resolution}が必要です。")
            
            # 顔のサイズが小さすぎる場合の判定 (例: 画像全体の面積の10%未満)
            image_area = width * height
            face_area = w * h
            if face_area / image_area < 0.05: # 10%未満
                 feedback.append(f"顔が画像全体に対して小さすぎます。もっと顔を大きく写してください。")


            if not feedback:
                # すべての基準を満たした場合
                return f"OK: プロフィール画像として良好です！ (顔検出: 1人, 鮮明度スコア: {sharpness_score:.2f})"
            else:
                # 何らかの問題がある場合
                return "NG: プロフィール画像に以下の問題があります。\n" + "\n".join(feedback)

        except Exception as e:
            return f"判定中にエラーが発生しました: {e}"

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageQualityCheckerApp(root)
    root.mainloop()