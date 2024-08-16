import cv2
import os

# Obter o caminho completo para a pasta onde estão as imagens
name_folder = './images/raw_images'
image_folder = os.path.join(os.getcwd(), name_folder)

# Obter lista de todas as imagens .jpg na pasta
images = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg')]

# Carregar o classificador Haar Cascade para detecção de face frontal
face_cascade = cv2.CascadeClassifier('./classifier/haarcascade_frontalface_default.xml')

# instancializa uma lista para armazenar caminho de arquivos em que não foi identificado nenhuma face
img_nofaces = []

# Diretório onde as imagens serão salvas
output_dir = './images/cropped_images'
os.makedirs(output_dir, exist_ok=True)

# itera sob cada indice do array images para obter o caminho de cada imagem
for image_path in images:
    # Carrega a imagem
    image = cv2.imread(image_path)

    # Converter a imagem para escala de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar as faces na imagem
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.09, minNeighbors=10, minSize=(30, 30)) # o retorno é um array com as coordenadas de todas faces detectadas

    # Verifica se foi detectada alguma face
    if len(faces) == 0:
        print(f'\nNenhuma face detectada:\n{image_path}\n')
        img_nofaces.append(image_path)
        continue
    
    # Encontra a maior face detectada (em termos de área)
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    (x, y, w, h) = largest_face # Extrai as coordenadas da maior face

    # Definir o fator de expansão (por exemplo, 1.3 vezes o tamanho da face)
    expansion_factor = 1.3
    # Calcular o novo tamanho expandido
    new_w = int(w * expansion_factor)
    new_h = int(h * expansion_factor)
    # Calcular o novo ponto superior esquerdo, centralizando o retângulo expandido
    new_x = max(x - (new_w - w) // 2, 0)
    new_y = max(y - (new_h - h) // 2, 0)
    
    # farantir que o recorte não ultrapasse os limites da imagem
    new_x = min(new_x, image.shape[1] - new_w)
    new_y = min(new_y, image.shape[0] - new_h)
    
    # Recortar a imagem usando as novas coordenadas e dimensões
    face_image = image[new_y:new_y+new_h, new_x:new_x+new_w]

    # cria o caminho completo com o nome do arquivo para a pasta newImg
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    face_filename = os.path.join(output_dir, f'{base_name}.jpg')
    
    # Salvar a imagem da face recortada
    cv2.imwrite(face_filename, face_image)
    print(f'\narquivo salvo:\n{face_filename}')
    #cv2.imshow('Face', face_image)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()

print(f'\nCaminho de arquivos com faces não detectadas:\n{img_nofaces}')