from ....core.logging import logger
from ..load_models import cv2, face_recognition, os, shutil, datetime, numpy as np, math

CWD = os.getcwd()

# Module specific business logic (will be use for endpoints)
class RecogService:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []

    def process(self, image):
        # Get time now for filename
        timeNow = self.getTimeNow()

        count = 1
        filename = f"{CWD}/data/output/{timeNow}/{count}/data/input.jpg"

        tmpcount = 1
        while os.path.exists(filename):
            filename = f"{CWD}/data/output/{timeNow}/{tmpcount}/data/input.jpg"
            count = tmpcount
            tmpcount += 1

        if not os.path.exists(f"{CWD}/data/output/{timeNow}/"):
            os.mkdir(f"{CWD}/data/output/{timeNow}/")
        if not os.path.exists(f"{CWD}/data/output/{timeNow}/{count}/"):
            os.mkdir(f"{CWD}/data/output/{timeNow}/{count}/")
        if not os.path.exists(f"{CWD}/data/output/{timeNow}/{count}/data/"):
            os.mkdir(f"{CWD}/data/output/{timeNow}/{count}/data/")

        # Save the image that is sent from the request and reject if filename is not valid
        with open(filename, "wb") as f:
            if image.filename.split(".")[-1].lower() not in ["jpg", "png", "jpeg", "heif"]:
                logger.warning("Filename not supported")
                return {"path_frame": None, "path_result": None, "result": None, "error_message": "No face detected", "status": 0}
            else:
                shutil.copyfileobj(image.file, f)

        frame = cv2.imread(filename)

        filenameDatas = {"timeNow": timeNow, "id": filename.split(f"{timeNow}/")[1].split("/data")[0]}

        boxes, confidences, filenames = self.getFaceCoordinates(frame, filenameDatas)

        if len(boxes) == 0:
            os.remove(filename)
            return {"path_frame": None, "path_result": None, "result": None, "error_message": "No face detected", "status": 0}
        
        resultRaw = []
        for currentFilename in filenames:
            try:
                resultRaw.append(self.recog(f"{CWD}/data/output/{currentFilename}"))
            except:
                resultRaw.append(["Unknown", "0%"])

        logger.info(resultRaw)

        frameNames = (i.split("/")[-1].split(".")[0] for i in filenames)
        
        result = {}
        for i, frameName in enumerate(frameNames):
            userDetected = resultRaw[i][0]
            confidence = resultRaw[i][1]
            if userDetected == "Unknown":
                confidence = confidences[i]
            result.update({frameName: f"{userDetected} :: {confidence}"})
        
        JSONFilename = f"{CWD}/data/output/{timeNow}/{count}/data/face.json"

        with open(JSONFilename, "w") as f:
            f.write(str(result))

        return {"path_frame": filenames, "path_result": JSONFilename.split("output/")[1], "result": result, "status": 1}

    def getTimeNow(self):
        # before: %d-%b-%y.%H-%M-%S
        return datetime.datetime.now().strftime("%Y%m%d")
    
    def getFaceCoordinates(self, frame, filenameDatas):
        timeNow = filenameDatas["timeNow"]
        id = filenameDatas["id"]
        detector = cv2.FaceDetectorYN.create(f"{CWD}/ml-models/face_detection_yunet/face_detection_yunet_2022mar.onnx", "", (320, 320))

        height, width, channels = frame.shape

        # Set input size
        detector.setInputSize((width, height))
        # Getting detections
        channel, faces = detector.detect(frame)
        faces = faces if faces is not None else []

        boxes = []
        confidences = []
        filenames = []
        count = 1
        
        for face in faces:
            box = list(map(int, face[:4]))
            boxes.append(box)
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            faceCropped = frame[y:y + h, x:x + w]

            ### SEMENTARA MASIH TANPA FILTERING MINIMUM PIXEL SHAPE 50
            # if w >= 50 and h >= 50 and x >= 0 and y >= 0:
            filename = f"{CWD}/data/output/{timeNow}/{id}/frame/frame{str(count).zfill(3)}.jpg"
            if not os.path.exists(f"{CWD}/data/output/{timeNow}/{id}/frame/"):
                os.mkdir(f"{CWD}/data/output/{timeNow}/{id}/frame/")
            filenames.append(filename.split("output/")[1])
            cv2.imwrite(filename, faceCropped)
            cv2.imwrite(filename, self.resize(filename, 360))
            count += 1
                
            confidence = face[-1]
            confidence = "{:.2f}%".format(confidence*100)

            confidences.append(confidence)

        return (boxes, confidences, filenames)
    
    def resize(self, filename: str, resolution: int):
        frame = cv2.imread(filename)
        if frame.shape[0] != resolution or frame.shape[1] != resolution:
            return cv2.resize(frame, (0, 0), fx=1-(frame.shape[1]-resolution)/frame.shape[1], fy=1-(frame.shape[1]-resolution)/frame.shape[1])
        else:
            return frame

    def recog(self, filename: str):
        # Read image as cv2
        frame = cv2.imread(filename)

        frame = self.resize(filename, 480)
        frame = self.convertBGRtoRGB(frame)

        faceNames = list(self.getFaceNames(frame))

        tmpFaceNames = []
        for i in faceNames:
            IDdetected = i.split("-")[0]
            if IDdetected == "Unknown (0%)":
                IDdetected = "Unknown"
                confidence = 0
            else:
                confidence = i.split("jpg (")[1].split("%")[0]
            # NOTE: CHANGE THIS LATER
            # Adds a minumum confidence of 10% for the API to return
            if float(confidence) > 10 or IDdetected == "Unknown":
                tmpFaceNames.append([IDdetected, f"{confidence}%"])
        faceNames = tmpFaceNames

        print(faceNames, flush=True)

        return [faceNames[0][0], faceNames[0][1]]

    def getFaceNames(self, frame):
        # Find all the faces and face encodings in the image
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = '0%'

            # Calculate the shortest distance to face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = self.faceConfidence(face_distances[best_match_index])
            
            face_names.append(f'{name} ({confidence})')

        return face_names
    
    def encodeFaces(self):
        # Encoding faces (Re-training for face detection algorithm)
        logger.info("Encoding Faces... (This may take a while)")
        for image in os.listdir(f'{CWD}/data/dataset'):
            face_image = face_recognition.load_image_file(f'{CWD}/data/dataset/{image}')
            try:
                face_encoding = face_recognition.face_encodings(face_image)[0]
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(image)
            except IndexError:
                pass
        
        logger.info("Encoding Done!")

    def convertBGRtoRGB(self, frame):
        return frame[:, :, ::-1]
    
    def faceConfidence(self, face_distance, face_match_threshold=0.6):
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)

        if face_distance > face_match_threshold:
            return str(round(linear_val * 100, 2)) + '%'
        else:
            value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
            return str(round(value, 2)) + '%'

recogService = RecogService()
recogService.encodeFaces()