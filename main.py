import argparse
import os
import time
import cv2
import numpy as np

from src.camera.camera_manager import CameraManager
from src.detection.face_detector import FaceDetector
from src.recognition.face_recognizer import FaceRecognizer
from src.data.face_database import FaceDatabase


def main():
    # Configurar argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Sistema de Reconhecimento Facial')
    parser.add_argument('--camera', type=int, default=0, 
                        help='ID da câmera (padrão: 0)')
    parser.add_argument('--width', type=int, default=640, 
                        help='Largura do frame (padrão: 640)')
    parser.add_argument('--height', type=int, default=480, 
                        help='Altura do frame (padrão: 480)')
    parser.add_argument('--method', type=str, default='dnn', 
                        choices=['haar', 'dnn'],
                        help='Método de detecção facial (padrão: dnn)')
    parser.add_argument('--register', type=str, default=None,
                        help='ID da pessoa para registrar (modo de registro)')
    parser.add_argument('--name', type=str, default=None,
                        help='Nome da pessoa (para modo de registro)')
    parser.add_argument('--flip', action='store_true',
                        help='Inverter imagem horizontalmente')
    args = parser.parse_args()
    
    # Atualizar configurações
    config['flip_horizontal'] = args.flip
    
    # Criar diretórios necessários
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Inicializar banco de dados
    face_database = FaceDatabase()
    
    # Inicializar componentes
    camera_manager = CameraManager()
    camera = camera_manager.add_camera("main", args.camera, args.width, args.height, args.flip)
    
    if camera is None:
        print("Erro crítico: Não foi possível inicializar a câmera.")
        return
    
    face_detector = FaceDetector(use_haar=(args.method == 'haar'), 
                                detection_scale=config['detection_scale'],
                                detection_neighbors=config['detection_neighbors'])
    
    face_recognizer = FaceRecognizer(database=face_database, 
                                    recognition_threshold=config['recognition_threshold'])
    
    # Configurar controles deslizantes
    setup_controls(camera, face_detector, face_recognizer)
    
    # Modo de registro
    register_mode = args.register is not None
    if register_mode:
        print(f"MODO DE REGISTRO: Registrando pessoa com ID '{args.register}'")
        if args.name:
            print(f"Nome: {args.name}")
        print("Pressione 'c' para capturar o rosto e 'q' para sair")
        registration_count = 0
    
    # Variáveis para controle de teclas
    last_key_time = time.time()
    
    # Loop principal
    try:
        while True:
            # Capturar frame
            frame, enhanced_frame = camera.read()
            if frame is None:
                print("Erro ao capturar frame. Verificando câmera...")
                time.sleep(0.5)
                continue
            
            # Usar o frame melhorado para processamento
            display_frame = enhanced_frame.copy()
            
            # Detectar faces
            face_rects = face_detector.detect_faces(enhanced_frame)
            
            # Processar cada face
            for (x, y, w, h) in face_rects:
                # Extrair e alinhar face
                try:
                    aligned_face = face_detector.align_face(enhanced_frame, (x, y, w, h))
                    
                    if aligned_face is None:
                        continue
                    
                    if register_mode:
                        # Desenhar retângulo verde em modo de registro
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(display_frame, "Pressione 'c' para capturar", 
                                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                   (0, 255, 0), 2)
                    else:
                        # Identificar face
                        person_id, similarity, person_info = face_recognizer.identify_face(aligned_face, "main")
                        
                        if person_id:
                            # Pessoa reconhecida
                            name = person_info.get('name', person_id) if person_info else person_id
                            confidence = f"{similarity:.1f}%"
                            color = (0, 255, 0)  # Verde
                            label = f"{name} ({confidence})"
                        else:
                            # Pessoa desconhecida
                            color = (0, 0, 255)  # Vermelho
                            label = f"Desconhecido ({similarity:.1f}%)"
                        
                        # Desenhar retângulo e nome
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(display_frame, label, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                except Exception as e:
                    print(f"Erro ao processar face: {e}")
            
            # Mostrar FPS
            fps = camera.get_fps()
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Mostrar modo
            mode_text = f"MODO: REGISTRO ({args.register})" if register_mode else "MODO: RECONHECIMENTO"
            cv2.putText(display_frame, mode_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Mostrar número de faces detectadas
            cv2.putText(display_frame, f"Faces: {len(face_rects)}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Exibir frame
            cv2.imshow("Reconhecimento Facial", display_frame)
            
            # Capturar tecla
            key = cv2.waitKey(1) & 0xFF
            
            # Processar teclas apenas a cada 200ms para evitar múltiplos eventos
            current_time = time.time()
            if current_time - last_key_time > 0.2:
                # Sair se pressionar 'q'
                if key == ord('q'):
                    break
                
                # Alternar flip horizontal se pressionar 'f'
                elif key == ord('f'):
                    config['flip_horizontal'] = not config['flip_horizontal']
                    camera.set_flip_horizontal(config['flip_horizontal'])
                    print(f"Flip horizontal: {config['flip_horizontal']}")
                    last_key_time = current_time
                
                # Capturar face se pressionar 'c' no modo de registro
                elif register_mode and key == ord('c') and len(face_rects) > 0:
                    # Pegar a maior face (assumindo que é a pessoa a ser registrada)
                    largest_face = max(face_rects, key=lambda rect: rect[2] * rect[3])
                    aligned_face = face_detector.align_face(enhanced_frame, largest_face)
                    
                    if aligned_face is not None:
                        # Registrar face
                        metadata = {
                            'name': args.name or args.register,
                            'timestamp': time.time()
                        }
                        if face_recognizer.register_face(args.register, aligned_face, metadata):
                            registration_count += 1
                            print(f"Face {registration_count} registrada para '{args.register}'")
                            
                            # Mostrar face capturada
                            x, y, w, h = largest_face
                            captured_face = enhanced_frame[y:y+h, x:x+w]
                            cv2.imshow("Face Capturada", captured_face)
                            cv2.waitKey(1000)
                            cv2.destroyWindow("Face Capturada")
                    
                    last_key_time = current_time
                
                # Mostrar reconhecimentos recentes se pressionar 'r'
                elif key == ord('r'):
                    recent = face_database.get_recent_recognitions(10)
                    print("\n--- Reconhecimentos Recentes ---")
                    for rec in recent:
                        print(f"{rec['timestamp']} - {rec['name']} ({rec['confidence']:.1f}%)")
                    print("-------------------------------\n")
                    last_key_time = current_time
    
    except KeyboardInterrupt:
        print("Programa interrompido pelo usuário")
    except Exception as e:
        print(f"Erro inesperado: {e}")
    
    finally:
        # Limpar recursos
        camera_manager.stop_all()
        cv2.destroyAllWindows()
        print("Programa encerrado")
