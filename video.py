import cv2
import time

def main():
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        print("Kameraya erişilemiyor. Lütfen kamera izinlerini kontrol edin.")
        return
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    cv2.namedWindow('Gelişmiş Kamera Akışı', cv2.WINDOW_GUI_NORMAL)

    prev_time = time.time()

    screenshot_count = 0

    try:
        while True:
            ret, frame = vid.read()
            if not ret:
                print("Kameradan görüntü alınamadı. Uygulama kapatılıyor.")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                roi_gray = gray_frame[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time))
            prev_time = curr_time

            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Algilanan Yuz Sayisi: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.imshow('Gelişmiş Kamera Akışı', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_name = f"screenshot_{screenshot_count}.png"
                cv2.imwrite(screenshot_name, frame)
                print(f"Ekran görüntüsü kaydedildi: {screenshot_name}")
                screenshot_count += 1

    except Exception as e:
        print(f"Bir hata oluştu: {e}")
    finally:
        vid.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
