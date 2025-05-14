import cv2
import numpy as np
import os

class FaceDetector:
    """Classe para detecção de faces em imagens"""
    
    def __init__(self, use_haar=False, detection_scale=1.1, detection_neighbors=5):
        """
        Inicializa o detector facial
        
        Args:
            use_haar: Se True, usa o detector Haar Cascade (mais rápido, menos preciso)
                     Se False, usa o detector DNN (mais preciso, mais lento)
            detection_scale: Fator de escala para detecção (apenas para Haar)
            detection_neighbors: Número mínimo de vizinhos (apenas para Haar)
        """
        self.use_haar = use_haar
        self.detection_scale = detection_scale
        self.detection_neighbors = detection_neighbors
        
        if use_haar:
            # Carregar o detector de faces Haar Cascade
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        else:
            # Carregar o detector DNN
            model_file = "models/res10_300x300_ssd_iter_140000.caffemodel"
            config_file = "models/deploy.prototxt"
            
            # Verificar se os arquivos existem
            if not os.path.exists(model_file) or not os.path.exists(config_file):
                print("Arquivos de modelo DNN não encontrados. Usando Haar Cascade como fallback.")
                self.use_haar = True
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            else:
                try:
                    self.net = cv2.dnn.readNetFromCaffe(config_file, model_file)
                except Exception as e:
                    print(f"Erro ao carregar modelo DNN: {e}")
                    print("Usando Haar Cascade como fallback.")
                    self.use_haar = True
                    self.face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_faces(self, frame):
        """
        Detecta faces em um frame
        
        Args:
            frame: Imagem/frame para detecção
        
        Returns:
            Lista de retângulos de faces (x, y, width, height)
        """
        if self.use_haar:
            # Converter para escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=self.detection_scale, 
                minNeighbors=self.detection_neighbors,
                minSize=(30, 30)
            )
            
            return faces
        else:
            # Usar o detector DNN
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0)
            )
            
            self.net.setInput(blob)
            detections = self.net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Limiar de confiança
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Converter para formato (x, y, w, h)
                    x = max(0, startX)
                    y = max(0, startY)
                    w = min(width - x, endX - startX)
                    h = min(height - y, endY - startY)
                    
                    # Verificar se a face está dentro dos limites da imagem
                    if w > 0 and h > 0:
                        faces.append((x, y, w, h))
            
            return faces
    
    def align_face(self, frame, face_rect):
        """
        Extrai e alinha a face para reconhecimento
        
        Args:
            frame: Imagem/frame
            face_rect: Retângulo da face (x, y, w, h)
        
        Returns:
            Face extraída e redimensionada
        """
        x, y, w, h = face_rect
        
        # Extrair a região da face
        face = frame[y:y+h, x:x+w]
        
        # Redimensionar para um tamanho padrão
        try:
            return cv2.resize(face, (160, 160))
        except Exception as e:
            print(f"Erro ao redimensionar face: {e}")
            return None
    
    def set_detection_parameters(self, scale=None, neighbors=None):
        """
        Atualiza os parâmetros de detecção
        
        Args:
            scale: Fator de escala para detecção
            neighbors: Número mínimo de vizinhos
        """
        if scale is not None:
            self.detection_scale = max(1.01, scale)
        if neighbors is not None:
            self.detection_neighbors = max(1, neighbors)
