import cv2
import time

def main():
    # Kamerayı başlat
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        print("Kameraya erişilemiyor. Lütfen kamera izinlerini kontrol edin.")
        return
    
    # Haarcascade modellerini yükle
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Pencereyi isimlendir ve özelleştir
    cv2.namedWindow('Gelişmiş Kamera Akışı', cv2.WINDOW_GUI_NORMAL)

    # FPS hesaplaması için zaman değişkenleri
    prev_time = time.time()

    # Ekran görüntüsü sayacı
    screenshot_count = 0

    try:
        while True:
            # Kameradan görüntü al
            ret, frame = vid.read()
            if not ret:
                print("Kameradan görüntü alınamadı. Uygulama kapatılıyor.")
                break

            # Görüntüyü gri tonlamaya çevir ve histogram eşitlemesi uygula
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.equalizeHist(gray_frame)

            # Yüz algılama
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=7, minSize=(50, 50))

            for (x, y, w, h) in faces:
                # Yüz bölgesini işaretle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # Göz algılama (yalnızca yüz içinde)
                roi_gray = gray_frame[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            # FPS hesaplama
            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time))
            prev_time = curr_time

            # FPS ve yüz sayısını ekrana yazdır
            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Algilanan Yuz Sayisi: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Görüntüyü göster
            cv2.imshow('Gelişmiş Kamera Akışı', frame)

            # Tuşları kontrol et
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Çıkış
                break
            elif key == ord('s'):  # Ekran görüntüsü kaydet
                screenshot_name = f"screenshot_{screenshot_count}.png"
                cv2.imwrite(screenshot_name, frame)
                print(f"Ekran görüntüsü kaydedildi: {screenshot_name}")
                screenshot_count += 1

    except Exception as e:
        print(f"Bir hata oluştu: {e}")
    finally:
        # Kamera ve pencere kaynaklarını serbest bırak
        vid.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
