import numpy as np
import os
import time
from skimage.measure import marching_cubes
from sdf_glock import SDFNode, Vector3

class MeshExporter:
    def __init__(self, sdf_model: SDFNode, resolution: int = 50, grid_bounds: float = 12.0):
        self.sdf_model = sdf_model
        self.resolution = resolution
        self.grid_bounds = grid_bounds

    def generate_obj(self, output_path: str):
        print(f"Починаємо генерацію сітки (Marching Cubes). Роздільна здатність: {self.resolution}^3...")
        t0 = time.perf_counter()
        
        x_lin = np.linspace(-self.grid_bounds, self.grid_bounds, self.resolution)
        y_lin = np.linspace(-self.grid_bounds, self.grid_bounds, self.resolution)
        z_lin = np.linspace(-self.grid_bounds, self.grid_bounds, self.resolution)
        
        # ВЕКТОРИЗОВАНА оцінка всього об'єму за один виклик (замість Python-циклу O(N^3))
        # meshgrid повертає масиви форми (Rx, Ry, Rz) - точно та форма, що потрібна marching_cubes
        XX, YY, ZZ = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
        
        has_np = hasattr(self.sdf_model, 'evaluate_np')
        if has_np:
            print("  [fast] Numpy-векторизований шлях...")
            volume = self.sdf_model.evaluate_np(XX, YY, ZZ).astype(np.float32)
        else:
            # Fallback: стара побітова петля (залишена для сумісності)
            print("  [slow] Скалярний fallback (evaluate_np не знайдено)...")
            volume = np.zeros_like(XX, dtype=np.float32)
            for i in range(self.resolution):
                for j in range(self.resolution):
                    for k in range(self.resolution):
                        volume[i, j, k] = self.sdf_model.evaluate(Vector3(x_lin[i], y_lin[j], z_lin[k]))

        t1 = time.perf_counter()
        print(f"  SDF об'єм обчислено за {t1 - t0:.2f}с  min={volume.min():.3f} max={volume.max():.3f}")

        # Використовуємо алгоритм Marching Cubes на нульовому рівні (surface: level=0)
        try:
            verts, faces, normals, values = marching_cubes(volume, level=0.0)
        except ValueError as e:
            print(f"Помилка генерації Marching Cubes: {e}")
            return

        # Трансформуємо координати вертексів назад у масштаб світу (від сітки)
        scale_step = (self.grid_bounds * 2.0) / (self.resolution - 1)
        real_verts = []
        for v in verts:
            real_x = -self.grid_bounds + v[0] * scale_step
            real_y = -self.grid_bounds + v[1] * scale_step
            real_z = -self.grid_bounds + v[2] * scale_step
            real_verts.append((real_x, real_y, real_z))
            
        # Записуємо .obj файл (його буде їсти наш Rust-додаток)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Згенеровано через Glock Pipeline\n")
            for rv in real_verts:
                f.write(f"v {rv[0]:.6f} {rv[1]:.6f} {rv[2]:.6f}\n")
            # marching cubes normals є орієнтовними
            for vn in normals:
                f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
            # грані (.obj індексуються з 1, а не з 0)
            for face in faces:
                f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")

        t2 = time.perf_counter()
        print(f"Готово. Меш збережено у {output_path}")
        print(f"  Вершин: {len(real_verts)}, Граней: {len(faces)}, Запис: {t2-t1:.2f}с")