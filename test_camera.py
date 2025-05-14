# test_camera.py
import cv2

def test_camera(camera_id=0):
    print(f"Tentando abrir câmera com ID: {camera_id}")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir a câmera com ID {camera_id}")
        return False
    
    print(f"Câmera {camera_id} aberta com sucesso!")
    print(f"Largura: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Altura: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    
    # Tentar ler um frame
    ret, frame = cap.read()
    if not ret:
        print("Erro: Não foi possível ler um frame da câmera")
        cap.release()
        return False
    
    print("Frame lido com sucesso!")
    print(f"Dimensões do frame: {frame.shape}")
    
    # Mostrar o frame
    cv2.imshow("Teste de Câmera", frame)
    print("Pressione qualquer tecla para fechar...")
    cv2.waitKey(0)
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    return True

def test_all_cameras():
    print("Testando todas as câmeras disponíveis...")
    for i in range(10):  # Testar IDs de 0 a 9
        if test_camera(i):
            print(f"Câmera {i} está funcionando!")
        else:
            print(f"Câmera {i} não está disponível ou não está funcionando.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Teste de Câmera')
    parser.add_argument('--camera', type=int, default=None, 
                        help='ID da câmera para testar (padrão: testar todas)')
    args = parser.parse_args()
    
    if args.camera is not None:
        test_camera(args.camera)
    else:
        test_all_cameras()
