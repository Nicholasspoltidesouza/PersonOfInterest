class FaceRecognizer:
    """Classe para reconhecimento facial usando OpenCV com banco de dados SQL"""
    
    def __init__(self, database=None, recognition_threshold=50):
        """
        Inicializa o reconhecedor facial
        
        Args:
            database: Instância de FaceDatabase ou None para usar o padrão
            recognition_threshold: Limiar de similaridade para reconhecimento (0-100)
        """
        # Inicializar o reconhecedor facial LBPH do OpenCV
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Banco de dados de faces
        self.database = database or FaceDatabase()
        self.label_map = {}  # Mapeia person_id para label numérico
        
        # Limiar de similaridade (quanto maior, mais permissivo)
        self.recognition_threshold = recognition_threshold
        
        # Carregar faces do banco de dados e treinar o modelo
        self._load_and_train()
        
        # Verificar se o modelo já existe e carregá-lo
        if os.path.exists("models/lbph_model.yml"):
            try:
                self.recognizer.read("models/lbph_model.yml")
                print("Modelo LBPH carregado com sucesso")
            except Exception as e:
                print(f"Erro ao carregar modelo LBPH: {e}")
    
    def _load_and_train(self):
        """Carrega faces do banco de dados e treina o modelo"""
        faces_data = self.database.get_all_faces()
        
        if not faces_data:
            print("Nenhuma face encontrada no banco de dados")
            return
        
        # Criar mapeamento de IDs
        unique_ids = set(person_id for person_id, _ in faces_data)
        self.label_map = {person_id: i for i, person_id in enumerate(unique_ids)}
        
        # Preparar dados para treinamento
        faces = []
        labels = []
        
        for person_id, face_data in faces_data:
            label = self.label_map[person_id]
            faces.append(face_data)
            labels.append(label)
        
        # Treinar o reconhecedor
        if faces and labels:
            self.recognizer.train(faces, np.array(labels))
            print(f"Modelo treinado com {len(faces)} faces de {len(unique_ids)} pessoas")
            
            # Salvar o modelo
            os.makedirs("models", exist_ok=True)
            self.recognizer.write("models/lbph_model.yml")
    
    def extract_features(self, face_img):
        """
        Extrai características da face
        
        Args:
            face_img: Imagem da face alinhada
        
        Returns:
            Imagem em escala de cinza redimensionada
        """
        # Converter para escala de cinza
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Redimensionar para tamanho padrão
        return cv2.resize(gray, (100, 100))
    
    def register_face(self, person_id, face_img, metadata=None):
        """
        Registra uma face no banco de dados
        
        Args:
            person_id: Identificador único da pessoa
            face_img: Imagem da face alinhada
            metadata: Informações adicionais (nome, cargo, etc.)
        
        Returns:
            True se o registro foi bem-sucedido, False caso contrário
        """
        try:
            # Extrair características
            face_features = self.extract_features(face_img)
            
            # Adicionar pessoa ao banco de dados se não existir
            name = metadata.get('name', person_id) if metadata else person_id
            self.database.add_person(person_id, name, metadata)
            
            # Adicionar face ao banco de dados
            if self.database.add_face(person_id, face_features):
                # Recarregar e treinar o modelo
                self._load_and_train()
                
                print(f"Face registrada para '{person_id}'")
                return True
            
            return False
        
        except Exception as e:
            print(f"Erro ao registrar face: {e}")
            return False
    
    def identify_face(self, face_img, camera_id="main"):
        """
        Identifica uma face
        
        Args:
            face_img: Imagem da face alinhada
            camera_id: ID da câmera para registro de log
        
        Returns:
            (person_id, confidence, metadata) ou (None, 0, None) se não reconhecido
        """
        try:
            # Extrair características
            face_features = self.extract_features(face_img)
            
            # Tentar reconhecer com LBPH
            label, confidence = self.recognizer.predict(face_features)
            
            # Converter confiança (menor é melhor no LBPH)
            # para similaridade (maior é melhor)
            similarity = 100 - confidence
            
            if similarity >= self.recognition_threshold:
                # Converter label numérico para person_id
                person_id = None
                
                # Encontrar o person_id correspondente ao label
                for pid, lbl in self.label_map.items():
                    if lbl == label:
                        person_id = pid
                        break
                
                if person_id:
                    # Registrar o reconhecimento no banco de dados
                    self.database.log_recognition(person_id, similarity, camera_id)
                    
                    # Obter informações da pessoa
                    person_info = self.database.get_person_info(person_id)
                    
                    return person_id, similarity, person_info
        
        except Exception as e:
            print(f"Erro ao identificar face: {e}")
        
        return None, 0, None
    
    def set_recognition_threshold(self, threshold):
        """
        Define o limiar de similaridade para reconhecimento
        
        Args:
            threshold: Valor entre 0 e 100 (quanto maior, mais permissivo)
        """
        self.recognition_threshold = max(0, min(100, threshold))
