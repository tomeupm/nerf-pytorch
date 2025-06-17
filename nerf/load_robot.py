import os
import re
import sys

import imageio
import numpy as np

def parse_poses_file(poses_file_path):
    """
    Parsea el archivo poses.txt y extrae las matrices de transformación.
    
    Args:
        poses_file_path: Ruta al archivo poses.txt
        
    Returns:
        Lista de matrices de transformación 4x4 como arrays de numpy
    """
    poses = []
    
    try:
        with open(poses_file_path, 'r') as f:
            content = f.read()
        
        # Buscar todas las matrices en el formato [... ; ... ; ... ; ...]
        matrix_pattern = r'\[\s*(.*?)\s*\]'
        matrices = re.findall(matrix_pattern, content, re.DOTALL)
        
        for matrix_str in matrices:
            # Dividir por punto y coma para obtener las filas
            rows = matrix_str.split(';')
            matrix_rows = []
            
            for row in rows:
                # Limpiar la fila y dividir por espacios/comas
                row_clean = row.strip().replace(',', '')
                if row_clean:  # Ignorar filas vacías
                    # Extraer números de la fila
                    numbers = re.findall(r'-?\d+\.?\d*', row_clean)
                    if len(numbers) == 4:  # Debe tener 4 elementos por fila
                        matrix_rows.append([float(num) for num in numbers])
            
            if len(matrix_rows) == 4:  # Debe tener 4 filas
                poses.append(np.array(matrix_rows))
        
        print(f"Se han parseado {len(poses)} matrices de poses desde {poses_file_path}")
        return poses
        
    except Exception as e:
        print(f"Error al parsear el archivo de poses {poses_file_path}: {e}")
        return []
    
def generate_render_poses(poses, n_poses=120, radius_scale=1.0):
    """
    Genera poses para renderizado en una trayectoria esférica alrededor de la escena.
    
    Args:
        poses: Array de poses de entrada [N, 4, 4]
        n_poses: Número de poses de renderizado a generar
        radius_scale: Factor de escala para el radio de la trayectoria
        
    Returns:
        render_poses: Array de poses de renderizado [n_poses, 4, 4]
    """
    # Calcular el centro y radio promedio de las poses
    centers = poses[:, :3, 3]  # Posiciones de las cámaras
    center = np.mean(centers, axis=0)
    
    # Calcular radio promedio desde el centro
    radii = np.linalg.norm(centers - center, axis=1)
    radius = np.mean(radii) * radius_scale
    
    # Calcular dirección promedio "hacia arriba"
    up_vecs = poses[:, :3, 1]  # Vector Y (up) de cada pose
    up = np.mean(up_vecs, axis=0)
    up = up / np.linalg.norm(up)
    
    # Crear trayectoria esférica
    render_poses = []
    
    for i in range(n_poses):
        # Ángulos para la trayectoria esférica
        theta = 2 * np.pi * i / n_poses  # Rotación horizontal
        phi = np.pi / 6  # Elevación fija (30 grados)
        
        # Posición en coordenadas esféricas
        x = radius * np.cos(phi) * np.cos(theta)
        y = radius * np.sin(phi)
        z = radius * np.cos(phi) * np.sin(theta)
        
        position = center + np.array([x, y, z])
        
        # Crear matriz de transformación
        # Z apunta hacia el centro de la escena
        z_axis = center - position
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # X es perpendicular a Z y al vector up
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y es perpendicular a X y Z
        y_axis = np.cross(z_axis, x_axis)
        
        # Crear matriz de pose 4x4
        pose = np.eye(4)
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        pose[:3, 3] = position
        
        render_poses.append(pose)
    
    return np.array(render_poses)


def load_robot_data(basedir, half_res=False, train_split=0.8):
    """
    Carga imágenes y poses para entrenamiento de NeRF, similar a load_blender_data.
    
    Args:
        basedir: Ruta base que contiene las imágenes y el archivo poses.txt
        half_res: Si reducir la resolución a la mitad
        train_split: Proporción de datos para entrenamiento (resto se usa para validación)
        
    Returns:
        imgs: Array de imágenes normalizadas [N, H, W, 3] (RGB)
        poses: Array de matrices de pose [N, 4, 4]
        render_poses: Poses para renderizado (espiral alrededor de la escena)
        hwf: [Height, Width, Focal] de las imágenes
        i_split: Índices para [train, val, test]
    """
    # Verificar que el directorio de imágenes exista
    if not os.path.isdir(basedir):
        print(f"Error: El directorio base '{basedir}' no existe.")
        sys.exit(1)

    img_path = os.path.join(basedir, "images_robot")
    poses_file_path = os.path.join(basedir, "poses.txt")

    # Verificar que el archivo poses.txt exista
    if not os.path.isfile(poses_file_path) or not poses_file_path.endswith('poses.txt'):
        print(f"Error: Debe proporcionar un archivo poses.txt válido. Archivo proporcionado: {poses_file_path}")
        sys.exit(1)
    
    # Obtener lista de archivos de imágenes y ordenarlas por el número al final del nombre
    img_files = [f for f in os.listdir(img_path) 
                 if os.path.isfile(os.path.join(img_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Verificar que haya imágenes
    if not img_files:
        print(f"Error: No se encontraron imágenes en '{img_path}'.")
        sys.exit(1)
    
    def extract_number_from_filename(filename):
        """Extrae el número al final del nombre del archivo."""
        # Buscar números al final del nombre (antes de la extensión)
        match = re.search(r'(\d+)\.', filename)
        return int(match.group(1)) if match else float('inf')
    
    # Ordenar imágenes por el número al final del nombre
    img_files.sort(key=extract_number_from_filename)
    
    print(f"Procesando archivo de poses: {poses_file_path}")
    
    # Parsear las matrices de poses
    poses = parse_poses_file(poses_file_path)
    
    if not poses:
        print("Error: No se pudieron parsear matrices de poses del archivo.")
        sys.exit(1)
    
    print(f"Se encontraron {len(img_files)} imágenes ordenadas numéricamente.")
    print(f"Se tienen {len(poses)} matrices de poses.")
    
    # Verificar que el número de poses coincida con el número de imágenes
    if len(poses) != len(img_files):
        print(f"Error: El número de poses ({len(poses)}) no coincide con el número de imágenes ({len(img_files)})")
        print("Asegúrese de que el archivo poses.txt contenga exactamente una matriz por cada imagen.")
        sys.exit(1)
    
    # Cargar imágenes
    print("Cargando imágenes...")
    imgs = []
    for i, img_file in enumerate(img_files):
        img_path_full = os.path.join(img_path, img_file)
        img = imageio.imread(img_path_full)
        
        # Convertir a RGB
        if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]  # Tomar solo los canales RGB
        elif len(img.shape) == 2:  # Escala de grises
            img = np.stack([img]*3, axis=-1)  # Convertir a RGB
            
        imgs.append(img)
        print(f"Cargada imagen {i+1}/{len(img_files)}: {img_file} - Shape: {img.shape}")
    
    # Convertir a arrays numpy y normalizar
    imgs = np.array(imgs).astype(np.float32) / 255.0
    poses = np.array(poses).astype(np.float32)
    
    print(f"Shape final de imágenes: {imgs.shape}")
    print(f"Shape final de poses: {poses.shape}")
    
    # Obtener dimensiones de imagen
    H, W = imgs[0].shape[:2]
    
    # Calcular focal length basado en las especificaciones de la cámara
    # FOV horizontal: 87°, FOV vertical: 58°
    fov_horizontal = 87.0 * np.pi / 180.0  # Convertir a radianes
    fov_vertical = 58.0 * np.pi / 180.0    # Convertir a radianes
    
    # Calcular focal length usando el FOV horizontal
    focal_x = W / (2.0 * np.tan(fov_horizontal / 2.0))
    # Calcular focal length usando el FOV vertical  
    focal_y = H / (2.0 * np.tan(fov_vertical / 2.0))
    
    # Usar el promedio de ambos focales o el horizontal (más común)
    focal = focal_x  # NeRF típicamente usa focal cuadrado
    
    print(f"FOV de la cámara - Horizontal: {87.0}°, Vertical: {58.0}°")
    print(f"Focal length calculado - fx: {focal_x:.2f}, fy: {focal_y:.2f}, usando: {focal:.2f}")
    
    # Aplicar reducción de resolución si se solicita
    if half_res:
        print("Aplicando reducción de resolución a la mitad...")
        new_h, new_w = H//2, W//2
        new_imgs = []
        
        for img in imgs:
            # Submuestreo simple (tomar cada segundo píxel)
            resized = img[::2, ::2]
            new_imgs.append(resized)
        
        imgs = np.array(new_imgs)
        H = H // 2
        W = W // 2
        focal = focal / 2.0
        
        print(f"Nueva resolución: {H}x{W}, nuevo focal: {focal:.2f}")
    
    # Crear splits para entrenamiento/validación/test
    n_imgs = len(imgs)
    n_train = int(n_imgs * train_split)
    n_val = max(1, int(n_imgs * 0.1))  # Al menos 1 para validación, 10% del total
    n_test = n_imgs - n_train - n_val
    
    # Si no hay suficientes imágenes para test, usar validación como test
    if n_test <= 0:
        n_test = n_val
        n_val = max(1, n_imgs - n_train - n_test)
    
    # Crear índices para los splits
    indices = np.arange(n_imgs)
    np.random.seed(42)  # Para reproducibilidad
    np.random.shuffle(indices)
    
    i_train = indices[:n_train]
    i_val = indices[n_train:n_train + n_val]
    if n_test == n_val:  # Usar validación como test
        i_test = i_val
    else:
        i_test = indices[n_train + n_val:n_train + n_val + n_test]
    
    i_split = [i_train, i_val, i_test]
    
    # Generar poses de renderizado para novel view synthesis
    render_poses = generate_render_poses(poses)
    
    print(f"Split de datos: {len(i_train)} entrenamiento, {len(i_val)} validación, {len(i_test)} test")
    print(f"Dimensiones de imagen: H={H}, W={W}, Focal={focal:.2f}")
    print(f"Generadas {len(render_poses)} poses para renderizado")
    
    hwf = [H, W, focal]
    
    return imgs, poses, render_poses, hwf, i_split