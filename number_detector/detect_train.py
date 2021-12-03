import pytesseract
import cv2


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


class Detector:
    def __init__(self, config, image_width=640, image_heigh=640):
        self.config = config
        self.image_width = image_width
        self.image_heigh = image_heigh

    # распознавание номера вагона
    def get_number(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.image_width, self.image_heigh))
        number = pytesseract.image_to_string(img, config=self.config)

        return number

    # распознавание номера на картинке
    def save_image(self, input_path, output_path):
        img = cv2.imread(input_path)
        img = cv2.resize(img, (self.image_width, self.image_heigh))

        boxes = pytesseract.image_to_data(img, config=self.config)

        for x, b in enumerate(boxes.splitlines()):
            if x!=0:
                b = b.split()
                if len(b) == 12:
                    x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                    cv2.rectangle(img, (x, y), (w + x, h + y), (0,0,255), 3)
                    cv2.putText(img, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 2)


        cv2.imwrite(output_path, img)


    # распознавание номера на видео
    def save_video(self, video_path):
        cap = cv2.VideoCapture(video_path)  

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w, h = 300, 200

        print("Total frames: {0}, Frames per second: {1}".format(frame_count, fps))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('number_detector/res/' + self.image_path.split('/')[1], fourcc, fps, (w, h))

        count = 0
        while True:
            success, img = cap.read()     # считываем кадры
            if success:
                img = cv2.resize(img, (300, 200))
                boxes = pytesseract.image_to_data(img, config=self.config)

                for x, b in enumerate(boxes.splitlines()):
                    if x!=0:
                        b = b.split()
                        if len(b) == 12:
                            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                            cv2.rectangle(img, (x, y), (w + x, h + y), (0,0,255), 3)
                            cv2.putText(img, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 2)

                writer.write(img)          # сохраняем результат
                count += 1
                print(count)
            else:
                break

        cap.release()
        writer.release() 


if __name__ == '__main__':
    detector = Detector(r'--oem 3 --psm 11 outputbase digits')
    res = detector.get_number('number_detector/numbers/num1.jpg')
    print(res)

    detector.save_image(input_path='number_detector/numbers/num2.jpg',
                         output_path='number_detector/res/res.jpg')
