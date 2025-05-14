import cv2
import time
from threading import Thread, Lock

class VideoStream:
    """Classe para gerenciar streams de vídeo de diferentes fontes"""
    
    def __init__(self, src=0, name="VideoStream", width=640, height=480, flip_horizontal=False):
        """
        Inicializa o stream de vídeo
        
        Args:
            src: Fonte do vídeo (0 para webcam padrão, string para IP/RTSP)
            name: Nome identificador do stream
            width: Largura desejada para os frames
            height: Altura desejada para os frames
            flip_horizontal: Se True, inverte a imagem horizontalmente
        """
        self.src = src
        self.stream = None
        self.name = name
        self.width = width
        self.height = height
        self.flip_horizontal = flip_horizontal
        self.stopped = False
        self.frame = None
        self.enhanced_frame = None
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        self.lock = Lock()
        self.brightness = 50  # 0-100
        self.contrast = 50    # 0-100
        
    def start(self):
        """Inicia a thread para leitura de frames"""
        # Inicializar a câmera
        self._initialize_camera()
        if self.stream is None:
            return False
            
        # Iniciar thread de captura
        Thread(target=self.update, args=()).start()
        return True
    
    def _initialize_camera(self, max_attempts=5):
        """Inicializa a câmera com múltiplas tentativas"""
        for attempt in range(max_attempts):
            print(f"Tentativa {attempt+1} de inicializar a câmera {self.src}...")
            self.stream = cv2.VideoCapture(self.src)
            
            if self.stream.isOpened():
                # Configurar propriedades
                self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Aumentar buffer
                
                # Verificar se consegue ler um frame
                ret, frame = self.stream.read()
                if ret:
                    print(f"Câmera {self.src} inicializada com sucesso!")
                    print(f"Resolução: {frame.shape[1]}x{frame.shape[0]}")
                    return True
                else:
                    print(f"Câmera {self.src} aberta, mas não conseguiu ler frames.")
                    self.stream.release()
                    self.stream = None
            else:
                print(f"Não foi possível abrir a câmera {self.src}.")
            
            # Esperar antes de tentar novamente
            time.sleep(1)
        
        print(f"Falha ao inicializar a câmera após {max_attempts} tentativas.")
        return False
        
    def update(self):
        """Loop principal para captura contínua de frames"""
        while not self.stopped:
            # Verificar se a câmera está aberta
            if self.stream is None or not self.stream.isOpened():
                print("Câmera fechada. Tentando reinicializar...")
                if not self._initialize_camera():
                    time.sleep(1)
                    continue
            
            # Ler frame
            ret, frame = self.stream.read()
            if not ret:
                print("Erro ao ler frame. Tentando recuperar...")
                self.stream.release()
                self.stream = None
                self._initialize_camera()
                time.sleep(0.5)
                continue
            
            # Aplicar flip horizontal se configurado
            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)
            
            # Melhorar imagem para condições de baixa luz
            enhanced = self._enhance_image(frame)
            
            # Atualizar frame com lock para thread safety
            with self.lock:
                self.frame = frame
                self.enhanced_frame = enhanced
                
                # Cálculo de FPS
                self.frame_count += 1
                elapsed_time = time.time() - self.last_time
                if elapsed_time >= 1.0:
                    self.fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.last_time = time.time()
            
            # Pequena pausa para reduzir uso de CPU
            time.sleep(0.01)
    
    def _enhance_image(self, frame):
        """Melhora a imagem para condições de baixa luz"""
        # Converter para LAB para separar luminosidade
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Mesclar canais de volta
        limg = cv2.merge((cl, a, b))
        
        # Converter de volta para BGR
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Ajustar brilho e contraste
        brightness = self.brightness / 50.0 - 1.0  # -1.0 a 1.0
        contrast = self.contrast / 50.0  # 0.0 a 2.0
        
        enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=brightness * 127)
        
        return enhanced
    
    def read(self):
        """Retorna o frame mais recente"""
        with self.lock:
            if self.frame is None:
                return None, None
            return self.frame.copy(), self.enhanced_frame.copy()
    
    def get_fps(self):
        """Retorna o FPS atual"""
        with self.lock:
            return self.fps
    
    def set_brightness(self, value):
        """Define o brilho (0-100)"""
        with self.lock:
            self.brightness = max(0, min(100, value))
    
    def set_contrast(self, value):
        """Define o contraste (0-100)"""
        with self.lock:
            self.contrast = max(0, min(100, value))
    
    def set_flip_horizontal(self, value):
        """Define se a imagem deve ser invertida horizontalmente"""
        with self.lock:
            self.flip_horizontal = value
    
    def stop(self):
        """Para a captura de vídeo"""
        self.stopped = True
        if self.stream is not None and self.stream.isOpened():
            self.stream.release()


class CameraManager:
    """Gerencia múltiplas câmeras/streams"""
    
    def __init__(self):
        self.cameras = {}
        
    def add_camera(self, camera_id, src=0, width=640, height=480, flip_horizontal=False):
        """Adiciona uma nova câmera ao gerenciador"""
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            
        camera = VideoStream(src=src, name=f"Camera-{camera_id}", 
                             width=width, height=height,
                             flip_horizontal=flip_horizontal)
        if camera.start():
            self.cameras[camera_id] = camera
            return camera
        return None
    
    def remove_camera(self, camera_id):
        """Remove uma câmera do gerenciador"""
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            del self.cameras[camera_id]
            
    def get_camera(self, camera_id):
        """Retorna uma câmera específica"""
        return self.cameras.get(camera_id)
    
    def get_all_frames(self):
        """Retorna os frames de todas as câmeras"""
        frames = {}
        for camera_id, camera in self.cameras.items():
            frame, enhanced = camera.read()
            if frame is not None:
                frames[camera_id] = (frame, enhanced)
        return frames
    
    def stop_all(self):
        """Para todas as câmeras"""
        for camera in self.cameras.values():
            camera.stop()
        self.cameras = {}
