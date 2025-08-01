# rsv
import os
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Пути
raster_paths = [
    r"C:\Users\123\Desktop\r_s_class\S1B_IW_SLC__1SDV_20210628T013430_20210628T013457_027550_0349E9_037C_deb_mat_Decomp_TC.tif",
    r"C:\Users\123\Desktop\r_s_class\S1B_IW_SLC__1SDV_20210722T013431_20210722T013458_027900_035441_242D_deb_mat_Decomp_TC.tif",
    r"C:\Users\123\Desktop\r_s_class\S1B_IW_SLC__1SDV_20210815T013433_20210815T013500_028250_035EC3_9790_deb_mat_Decomp_TC.tif"
]
shapefile_dir = r"C:\Users\123\Desktop\r_s_class\борисовка"
output_dir = r"C:\Users\123\Desktop\r_s_class\output"
os.makedirs(output_dir, exist_ok=True)
chunk_size = 512
model_path = os.path.join(output_dir, "rf_model.pkl")

# Чтение обучающих данных
def read_training_data(shapefile_dir, raster_path):
    crops = ['пшеница', 'ячмень', 'овес']
    X, y = [], []
    with rasterio.open(raster_path) as src:
        transform = src.transform
        for crop in crops:
            shp_path = os.path.join(shapefile_dir, f"{crop}.shp")
            gdf = gpd.read_file(shp_path).to_crs(src.crs)
            for geom in gdf.geometry:
                if geom.geom_type == 'Polygon':
                    for point in geom.exterior.coords:
                        px, py = ~transform * point
                        px, py = int(px), int(py)
                        if 0 <= px < src.width and 0 <= py < src.height:
                            sample = []
                            for path in raster_paths:
                                with rasterio.open(path) as r:
                                    val = r.read(window=Window(px, py, 1, 1)).flatten()
                                    sample.extend(val)
                            X.append(sample)
                            y.append(crop)
    return np.array(X), np.array(y)

# Обучение
def train_model(X, y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    joblib.dump((model, encoder), model_path)
    return model, encoder

# Генератор чанков
def generate_chunks(width, height, size):
    for y in range(0, height, size):
        for x in range(0, width, size):
            yield Window(x, y, min(size, width - x), min(size, height - y))

# Предсказание по чанкам
def predict_in_chunks(model, encoder, raster_paths, chunk_size, output_path):
    with rasterio.open(raster_paths[0]) as ref:
        meta = ref.meta.copy()
        width, height = ref.width, ref.height
        meta.update(count=1, dtype='uint8')

    with rasterio.open(output_path, 'w', **meta) as dst:
        for window in generate_chunks(width, height, chunk_size):
            chunk_data = []
            for path in raster_paths:
                with rasterio.open(path) as src:
                    data = src.read(window=window)
                    chunk_data.append(data)
            stack = np.concatenate(chunk_data, axis=0)  # (bands, h, w)
            h, w = stack.shape[1:]
            reshaped = stack.reshape(stack.shape[0], -1).T  # (pixels, bands)
            pred = model.predict(reshaped)
            pred_2d = pred.reshape(h, w).astype('uint8')
            dst.write(pred_2d, window=window, indexes=1)

# Главный блок
if __name__ == "__main__":
    print("[*] Чтение обучающих данных...")
    X, y = read_training_data(shapefile_dir, raster_paths[0])
    print(f"[*] Обучено на {len(X)} пикселях.")

    print("[*] Обучение модели...")
    model, encoder = train_model(X, y)

    print("[*] Классификация по чанкам...")
    output_path = os.path.join(output_dir, "classified.tif")
    predict_in_chunks(model, encoder, raster_paths, chunk_size, output_path)

    print(f"[+] Сохранено: {output_path}")
