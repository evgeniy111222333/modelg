import math
import json
import os

class CameraConfig:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def generate_views(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Фокус посунуто по X/Y для ідеального центрування рамки та затвору Глоку 
        target = {"x": -0.5, "y": -0.7, "z": 0.0}
        
        # Змінено реальні мірки "кінокамери".
        # 2.5 - Macro Detail; 5.0 - Простір деталізації, 8.0 та 12.0 - Overview та Cinematic Zoom.
        distances = [2.5, 5.0, 8.0, 12.0]
        
        cameras =[]
        view_idx = 0
        
        for dist in distances:
            # Для кожної дистанції робимо ракурси з кроком 45 градусів
            for angle_deg in range(0, 360, 45):
                angle_rad = math.radians(angle_deg)
                
                # Нахил кіношний, знімаємо збоку та 45 градусів згори (вісь Y= 0.3*дист), шоб відчути верх затвора
                cam_x = target["x"] + dist * math.cos(angle_rad)
                cam_y = target["y"] + (dist * 0.35) 
                cam_z = target["z"] + dist * math.sin(angle_rad)
                
                cameras.append({
                    "view_id": f"dist_{dist}_ang_{angle_deg}",
                    "distance": dist,
                    "angle_deg": angle_deg,
                    "position": {"x": cam_x, "y": cam_y, "z": cam_z},
                    "target": target
                })
                view_idx += 1
                
        config_path = os.path.join(self.output_dir, "camera_config.json")
        with open(config_path, "w", encoding="utf-8") as file:
            json.dump({"views": cameras}, file, indent=4)
            
        print(f"Згенеровано конфіг з {len(cameras)} камерами зуму у файлі {config_path}")

if __name__ == "__main__":
    generator = CameraConfig("config_output")
    generator.generate_views()