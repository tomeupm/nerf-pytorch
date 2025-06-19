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


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w

def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses

def load_robot_data(basedir, factor=8, recenter=True, bd_factor=0.75, spherify=False, path_zflat=False):
    """
    Carga imágenes y poses para entrenamiento de NeRF con tratamiento similar a LLFF.
    
    Args:
        basedir: Ruta base que contiene las imágenes y el archivo poses.txt
        factor: Factor de reducción de resolución
        recenter: Si recentrar las poses
        bd_factor: Factor para el reescalado de bounds
        spherify: Si esfericizar las poses
        path_zflat: Si aplanar la trayectoria en Z
        
    Returns:
        images: Array de imágenes normalizadas [N, H, W, 3] (RGB)
        poses: Array de matrices de pose [N, 3, 5] (incluye H, W, focal)
        bds: Bounds de la escena [N, 2] (near, far)
        render_poses: Poses para renderizado
        i_test: Índice de la vista de test
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
    
    # Aplicar factor de reducción
    if factor > 1:
        print(f"Aplicando factor de reducción: {factor}")
        new_h, new_w = H//factor, W//factor
        new_imgs = []
        
        for img in imgs:
            # Submuestreo simple
            resized = img[::factor, ::factor]
            new_imgs.append(resized)
        
        imgs = np.array(new_imgs)
        H = H // factor
        W = W // factor
        print(f"Nueva resolución: {H}x{W}")
    
    # Calcular focal length
    fov_horizontal = 87.0 * np.pi / 180.0
    focal = W / (2.0 * np.tan(fov_horizontal / 2.0))
    if factor > 1:
        focal = focal / factor
    
    print(f"Focal length: {focal:.2f}")
    
    # Convertir poses al formato LLFF [N, 3, 5] donde las últimas columnas son [H, W, focal]
    poses_llff = np.zeros((len(poses), 3, 5), dtype=np.float32)
    poses_llff[:, :, :4] = poses[:, :3, :].astype(np.float32)  # Solo las primeras 3 filas de la matriz 4x4
    poses_llff[:, :, 4] = np.array([H, W, focal], dtype=np.float32)  # Añadir H, W, focal
    
    # Estimar bounds de la escena
    print("Estimando bounds de la escena...")
    positions = poses[:, :3, 3].astype(np.float32)  # Posiciones de las cámaras
    center = np.mean(positions, axis=0).astype(np.float32)
    distances = np.linalg.norm(positions - center, axis=1).astype(np.float32)
    
    # Calcular near y far basado en las distancias de las cámaras
    scene_radius = np.max(distances).astype(np.float32)
    near = (scene_radius * 0.1).astype(np.float32)  # 10% del radio de la escena
    far = (scene_radius * 3.0).astype(np.float32)   # 300% del radio de la escena
    
    # Crear bounds para cada vista
    bds = np.array([[near, far] for _ in range(len(poses))], dtype=np.float32)
    
    print(f"Bounds estimados: near={near:.3f}, far={far:.3f}")
    
    # Aplicar reescalado basado en bd_factor
    if bd_factor is not None:
        sc = (1.0 / (bds.min() * bd_factor)).astype(np.float32)
        poses_llff[:, :3, 3] *= sc
        bds *= sc
        print(f"Aplicado reescalado con factor: {sc:.3f}")
    
    # Recentrar poses si se solicita
    if recenter:
        print("Recentrando poses...")
        # Convertir temporalmente a formato completo para recentrado
        poses_temp = np.zeros((len(poses_llff), 4, 4))
        poses_temp[:, :3, :] = poses_llff[:, :, :4]
        poses_temp[:, 3, 3] = 1.0
        
        poses_temp = recenter_poses(poses_temp)
        poses_llff[:, :, :4] = poses_temp[:, :3, :]
    
    # Generar poses de renderizado
    c2w = poses_avg(poses_llff)
    up = normalize(poses_llff[:, :3, 1].sum(0))
    
    # Calcular parámetros para la trayectoria espiral
    close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
    dt = 0.75
    mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
    focal_spiral = mean_dz
    
    # Obtener radios para la trayectoria espiral
    shrink_factor = 0.8
    zdelta = close_depth * 0.2
    tt = poses_llff[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_views = 120
    N_rots = 2
    
    if path_zflat:
        zloc = -close_depth * 0.1
        c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
        rads[2] = 0.0
        N_rots = 1
        N_views = N_views // 2
    
    render_poses = render_path_spiral(
        c2w_path, up, rads, focal_spiral, zdelta, zrate=0.5, rots=N_rots, N=N_views
    )
    render_poses = np.array(render_poses, dtype=np.float32)
    
    # Determinar vista de test (la más cercana al centro promedio)
    c2w_center = poses_avg(poses_llff)
    dists = np.sum(np.square(c2w_center[:3, 3] - poses_llff[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    
    print(f"Dimensiones finales:")
    print(f"Imágenes: {imgs.shape}")
    print(f"Poses: {poses_llff.shape}")
    print(f"Bounds: {bds.shape}")
    print(f"Render poses: {len(render_poses)}")
    print(f"Vista de test: {i_test}")
    
    return imgs, poses_llff, bds, render_poses, i_test