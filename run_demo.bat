@echo off
setlocal

python predict.py --model artifacts\face_recognizer.joblib --image data\demo\olivetti_faces\person_07\person_07_070.png

