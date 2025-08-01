# Улучшенная система классификации растровых данных

Система для обучения моделей машинного обучения на растровых данных и выполнения классификации больших растровых наборов данных с использованием чанковой обработки.

## Возможности

- 🏗️ **Объектно-ориентированная архитектура** - чистый, поддерживаемый код
- 🛡️ **Надежная обработка ошибок** - валидация входных данных и graceful degradation
- ⚡ **Оптимизированная производительность** - многопоточность и эффективная обработка памяти
- 📊 **Улучшенная выборка данных** - интеллектуальная выборка точек из полигонов
- 📈 **Прогресс-бары и логирование** - отслеживание выполнения операций
- 🔧 **Гибкая конфигурация** - поддержка JSON конфигурационных файлов
- 📝 **Подробная оценка модели** - метрики точности и отчеты классификации

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd rsv
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Быстрый старт

### Базовое использование

```python
from raster_classifier import RasterClassifier, Config

# Создайте конфигурацию
config = Config(
    raster_paths=[
        "path/to/raster1.tif",
        "path/to/raster2.tif",
        "path/to/raster3.tif"
    ],
    shapefile_dir="path/to/shapefiles",
    output_dir="output",
    crops=['пшеница', 'ячмень', 'овес']
)

# Инициализируйте классификатор
classifier = RasterClassifier(config)

# Запустите полный пайплайн
output_path = classifier.run_full_pipeline()
print(f"Классификация завершена: {output_path}")
```

### Использование конфигурационного файла

1. Создайте `config.json`:
```json
{
    "raster_paths": [
        "data/raster1.tif",
        "data/raster2.tif"
    ],
    "shapefile_dir": "data/shapefiles",
    "output_dir": "output",
    "chunk_size": 512,
    "crops": ["пшеница", "ячмень", "овес"],
    "n_estimators": 100
}
```

2. Используйте конфигурацию:
```python
from raster_classifier import load_config_from_file, RasterClassifier

config = load_config_from_file("config.json")
classifier = RasterClassifier(config)
output_path = classifier.run_full_pipeline()
```

## Структура проекта

```
rsv/
├── raster_classifier.py    # Основной модуль классификации
├── config.json            # Шаблон конфигурации
├── example_usage.py       # Примеры использования
├── requirements.txt       # Зависимости Python
└── README.md             # Документация
```

## Конфигурация

### Параметры Config

| Параметр | Тип | Описание | По умолчанию |
|----------|-----|----------|--------------|
| `raster_paths` | List[str] | Пути к растровым файлам | Обязательный |
| `shapefile_dir` | str | Директория с шейп-файлами | Обязательный |
| `output_dir` | str | Выходная директория | Обязательный |
| `chunk_size` | int | Размер чанка для обработки | 512 |
| `crops` | List[str] | Список классов культур | ['пшеница', 'ячмень', 'овес'] |
| `n_estimators` | int | Количество деревьев в Random Forest | 100 |
| `random_state` | int | Seed для воспроизводимости | 42 |
| `test_size` | float | Доля тестовых данных | 0.2 |

## Пошаговое использование

```python
from raster_classifier import RasterClassifier, Config

config = Config(
    raster_paths=["data/raster1.tif"],
    shapefile_dir="data/shapefiles", 
    output_dir="output"
)

classifier = RasterClassifier(config)

# Шаг 1: Валидация входных данных
if not classifier._validate_inputs():
    print("Ошибка валидации входных данных")
    exit(1)

# Шаг 2: Чтение обучающих данных
X, y = classifier.read_training_data(config.raster_paths[0])
print(f"Загружено обучающих данных: {X.shape}")

# Шаг 3: Обучение модели
model, encoder = classifier.train_model(X, y)
print("Модель обучена")

# Шаг 4: Классификация
output_path = "output/classified.tif"
classifier.predict_in_chunks(output_path)
print(f"Классификация завершена: {output_path}")
```

## Работа с существующими моделями

```python
# Загрузка существующей модели
config = Config(
    raster_paths=["data/raster1.tif"],
    shapefile_dir="data/shapefiles",
    output_dir="output",
    model_path="output/rf_model.pkl"
)

classifier = RasterClassifier(config)
model, encoder = classifier.load_model()

# Использование для классификации
classifier.predict_in_chunks("output/new_classification.tif")
```

## Улучшения по сравнению с оригинальным кодом

### 🏗️ Архитектура
- **Класс RasterClassifier**: Инкапсулирует всю функциональность
- **Dataclass Config**: Типизированная конфигурация
- **Разделение ответственности**: Каждый метод имеет четкую задачу

### 🛡️ Надежность
- **Валидация входных данных**: Проверка существования файлов
- **Обработка ошибок**: Try-catch блоки с информативными сообщениями
- **Обработка NoData**: Корректная работа с пропущенными значениями

### ⚡ Производительность
- **Многопоточность**: `n_jobs=-1` в Random Forest
- **Эффективная выборка**: Улучшенный алгоритм выборки точек из полигонов
- **Оптимизация памяти**: Обработка по чанкам с контролем памяти

### 📊 Мониторинг
- **Логирование**: Детальные логи в файл и консоль
- **Прогресс-бары**: tqdm для отслеживания прогресса
- **Метрики модели**: Accuracy, classification report

### 🔧 Удобство использования
- **Конфигурационные файлы**: JSON конфигурация
- **Примеры использования**: Подробные примеры в `example_usage.py`
- **Документация**: Полная типизация и docstrings

## Требования к данным

### Растровые файлы
- Формат: GeoTIFF (.tif)
- Проекция: Любая поддерживаемая GDAL
- Каналы: Любое количество

### Шейп-файлы
- Формат: ESRI Shapefile (.shp)
- Геометрия: Полигоны
- Именование: `{crop_name}.shp` (например, `пшеница.shp`)

## Логирование

Система создает логи в двух местах:
- **Консоль**: Основные сообщения о прогрессе
- **Файл**: Детальные логи в `{output_dir}/classifier.log`

## Производительность

Система оптимизирована для работы с большими растрами:
- **Чанковая обработка**: Обработка по частям для экономии памяти
- **Параллельные вычисления**: Использование всех доступных ядер CPU
- **LZW сжатие**: Сжатие выходных файлов

## Примеры

Запустите `example_usage.py` для просмотра различных способов использования:

```bash
python example_usage.py
```

## Лицензия

MIT License
