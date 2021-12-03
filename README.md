# Accenture_hackathon

## Инструкция для запуска(Windows + python3.9)
1. git clone https://github.com/jetminded/accenture_hackathon_DS.git
2. Создание и активация виртуального окружения:
    * cd путь к папке Accenture_hackathon
    * python -m venv venv
    * cd venv/Scripts
    * activate.bat
3. Скачайте tesseract exe по ссылке: https://github.com/UB-Mannheim/tesseract/wiki
   Установите файл exe в следующую папку: C:\Program Files\Tesseract- OCR
4. Установка зависимостей:
    * cd путь к папке Accenture_hackathon
    * pip install -r requirements.txt
5. Теперь Вы можете запустить один из 3 основных файлов:
    * number_detector/detect_train.py - распознавание номера вагона
    * rubbish_classifier/inference.py - классификация брака в вагоне 
    * rubbish_вуеусещк/detect_rubbish.py - детекция брака в вагоне + процент брака


## Дополнительная информация
* Обучение классфикатора проходило в google collab и все результаты и визуализации находятся в rubbish_classifier/trainer.ipynb
* В файлах detect_train.py, inference.py и detect_rubbish.py в соответстующие объекты классов нужно передавать пути к интересующему изображенияю.
* Рекомендуется сначала запускать классификатор на обнаружение наличия или отсутствия брака(так как он обучен с высокими метриками accuracy и f1score). В случае если брак на изображении есть, то тогда  запускать детектор, чтобы определить где брак и какой процент он составляет от общей площади.
