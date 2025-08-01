#!/usr/bin/env python3
"""
Improved Raster Classification System

This module provides functionality for training machine learning models on raster data
and performing classification on large raster datasets using chunked processing.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Generator, Union
from dataclasses import dataclass
import json

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioIOError
import geopandas as gpd
from shapely.geometry import Point, Polygon
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm


@dataclass
class Config:
    """Configuration class for the raster classifier."""
    raster_paths: List[str]
    shapefile_dir: str
    output_dir: str
    chunk_size: int = 512
    model_path: Optional[str] = None
    crops: List[str] = None
    n_estimators: int = 100
    random_state: int = 42
    test_size: float = 0.2
    
    def __post_init__(self):
        if self.crops is None:
            self.crops = ['пшеница', 'ячмень', 'овес']
        if self.model_path is None:
            self.model_path = os.path.join(self.output_dir, "rf_model.pkl")


class RasterClassifier:
    """
    A class for training and applying machine learning models to raster data.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the RasterClassifier.
        
        Args:
            config: Configuration object containing all necessary parameters
        """
        self.config = config
        self.model = None
        self.encoder = None
        self.logger = self._setup_logging()
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = os.path.join(self.config.output_dir, 'classifier.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _validate_inputs(self) -> bool:
        """
        Validate input files and directories.
        
        Returns:
            bool: True if all inputs are valid, False otherwise
        """
        # Check raster files
        for raster_path in self.config.raster_paths:
            if not os.path.exists(raster_path):
                self.logger.error(f"Raster file not found: {raster_path}")
                return False
            
            try:
                with rasterio.open(raster_path) as src:
                    if src.count == 0:
                        self.logger.error(f"Empty raster file: {raster_path}")
                        return False
            except RasterioIOError as e:
                self.logger.error(f"Cannot read raster file {raster_path}: {e}")
                return False
        
        # Check shapefile directory
        if not os.path.exists(self.config.shapefile_dir):
            self.logger.error(f"Shapefile directory not found: {self.config.shapefile_dir}")
            return False
        
        # Check for crop shapefiles
        for crop in self.config.crops:
            shp_path = os.path.join(self.config.shapefile_dir, f"{crop}.shp")
            if not os.path.exists(shp_path):
                self.logger.error(f"Shapefile not found: {shp_path}")
                return False
        
        return True
    
    def read_training_data(self, raster_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read training data from shapefiles and extract pixel values from rasters.
        
        Args:
            raster_path: Path to the reference raster file
            
        Returns:
            Tuple of feature matrix X and labels y
        """
        self.logger.info("Reading training data...")
        X, y = [], []
        
        try:
            with rasterio.open(raster_path) as src:
                transform = src.transform
                crs = src.crs
                
                for crop in tqdm(self.config.crops, desc="Processing crops"):
                    shp_path = os.path.join(self.config.shapefile_dir, f"{crop}.shp")
                    
                    try:
                        gdf = gpd.read_file(shp_path)
                        if gdf.crs != crs:
                            gdf = gdf.to_crs(crs)
                        
                        points_processed = 0
                        for geom in gdf.geometry:
                            if geom.geom_type == 'Polygon':
                                # Sample points from polygon boundary and interior
                                points = self._sample_points_from_polygon(geom, n_points=50)
                                
                                for point in points:
                                    px, py = ~transform * (point.x, point.y)
                                    px, py = int(px), int(py)
                                    
                                    if 0 <= px < src.width and 0 <= py < src.height:
                                        sample = self._extract_pixel_values(px, py)
                                        if sample is not None:
                                            X.append(sample)
                                            y.append(crop)
                                            points_processed += 1
                        
                        self.logger.info(f"Processed {points_processed} points for {crop}")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {crop}: {e}")
                        continue
                        
        except RasterioIOError as e:
            self.logger.error(f"Error reading raster {raster_path}: {e}")
            raise
        
        X_array, y_array = np.array(X), np.array(y)
        self.logger.info(f"Training data shape: {X_array.shape}")
        return X_array, y_array
    
    def _sample_points_from_polygon(self, polygon: Polygon, n_points: int = 50) -> List[Point]:
        """
        Sample points from polygon boundary and interior.
        
        Args:
            polygon: Shapely polygon geometry
            n_points: Number of points to sample
            
        Returns:
            List of Point geometries
        """
        points = []
        
        # Sample from boundary
        boundary_points = int(n_points * 0.3)
        for i in range(boundary_points):
            point = polygon.boundary.interpolate(i / boundary_points, normalized=True)
            points.append(point)
        
        # Sample from interior
        minx, miny, maxx, maxy = polygon.bounds
        interior_points = n_points - boundary_points
        attempts = 0
        max_attempts = interior_points * 10
        
        while len(points) < n_points and attempts < max_attempts:
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            point = Point(x, y)
            
            if polygon.contains(point):
                points.append(point)
            
            attempts += 1
        
        return points
    
    def _extract_pixel_values(self, px: int, py: int) -> Optional[List[float]]:
        """
        Extract pixel values from all raster files at given coordinates.
        
        Args:
            px, py: Pixel coordinates
            
        Returns:
            List of pixel values or None if extraction fails
        """
        sample = []
        
        try:
            for path in self.config.raster_paths:
                with rasterio.open(path) as r:
                    window = Window(px, py, 1, 1)
                    values = r.read(window=window)
                    
                    # Handle nodata values
                    if r.nodata is not None:
                        values = np.where(values == r.nodata, np.nan, values)
                    
                    sample.extend(values.flatten())
            
            # Check for NaN values
            if np.any(np.isnan(sample)):
                return None
                
            return sample
            
        except Exception as e:
            self.logger.warning(f"Error extracting pixel values at ({px}, {py}): {e}")
            return None
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[RandomForestClassifier, LabelEncoder]:
        """
        Train the Random Forest classifier.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Trained model and label encoder
        """
        self.logger.info("Training model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state, 
            stratify=y
        )
        
        # Encode labels
        encoder = LabelEncoder()
        y_train_encoded = encoder.fit_transform(y_train)
        y_test_encoded = encoder.transform(y_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            n_jobs=-1,  # Use all available cores
            verbose=1
        )
        
        model.fit(X_train, y_train_encoded)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        self.logger.info(f"Model accuracy: {accuracy:.4f}")
        self.logger.info("Classification report:")
        self.logger.info(f"\n{classification_report(y_test_encoded, y_pred, target_names=encoder.classes_)}")
        
        # Save model
        joblib.dump((model, encoder), self.config.model_path)
        self.logger.info(f"Model saved to: {self.config.model_path}")
        
        self.model = model
        self.encoder = encoder
        
        return model, encoder
    
    def load_model(self) -> Tuple[RandomForestClassifier, LabelEncoder]:
        """
        Load a previously trained model.
        
        Returns:
            Loaded model and encoder
        """
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
        
        self.logger.info(f"Loading model from: {self.config.model_path}")
        model, encoder = joblib.load(self.config.model_path)
        
        self.model = model
        self.encoder = encoder
        
        return model, encoder
    
    def generate_chunks(self, width: int, height: int, size: int) -> Generator[Window, None, None]:
        """
        Generate processing chunks for large rasters.
        
        Args:
            width, height: Raster dimensions
            size: Chunk size
            
        Yields:
            Window objects for each chunk
        """
        for y in range(0, height, size):
            for x in range(0, width, size):
                yield Window(x, y, min(size, width - x), min(size, height - y))
    
    def predict_in_chunks(self, output_path: str) -> None:
        """
        Perform classification on raster data using chunked processing.
        
        Args:
            output_path: Path for the output classified raster
        """
        if self.model is None or self.encoder is None:
            raise ValueError("Model not trained or loaded. Call train_model() or load_model() first.")
        
        self.logger.info("Starting chunked prediction...")
        
        try:
            with rasterio.open(self.config.raster_paths[0]) as ref:
                meta = ref.meta.copy()
                width, height = ref.width, ref.height
                meta.update(count=1, dtype='uint8', compress='lzw')
            
            total_chunks = ((width + self.config.chunk_size - 1) // self.config.chunk_size) * \
                          ((height + self.config.chunk_size - 1) // self.config.chunk_size)
            
            with rasterio.open(output_path, 'w', **meta) as dst:
                chunk_generator = self.generate_chunks(width, height, self.config.chunk_size)
                
                for window in tqdm(chunk_generator, total=total_chunks, desc="Processing chunks"):
                    try:
                        chunk_data = []
                        
                        # Read data from all rasters
                        for path in self.config.raster_paths:
                            with rasterio.open(path) as src:
                                data = src.read(window=window)
                                chunk_data.append(data)
                        
                        # Stack and reshape data
                        stack = np.concatenate(chunk_data, axis=0)  # (bands, h, w)
                        h, w = stack.shape[1:]
                        reshaped = stack.reshape(stack.shape[0], -1).T  # (pixels, bands)
                        
                        # Handle nodata values
                        valid_mask = ~np.any(np.isnan(reshaped), axis=1)
                        
                        # Predict
                        pred = np.full(reshaped.shape[0], 0, dtype='uint8')  # Default to 0
                        if np.any(valid_mask):
                            pred[valid_mask] = self.model.predict(reshaped[valid_mask])
                        
                        pred_2d = pred.reshape(h, w).astype('uint8')
                        dst.write(pred_2d, window=window, indexes=1)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing chunk {window}: {e}")
                        # Write zeros for failed chunks
                        zeros = np.zeros((window.height, window.width), dtype='uint8')
                        dst.write(zeros, window=window, indexes=1)
                        
        except Exception as e:
            self.logger.error(f"Error in chunked prediction: {e}")
            raise
        
        self.logger.info(f"Classification completed. Output saved to: {output_path}")
    
    def run_full_pipeline(self) -> str:
        """
        Run the complete classification pipeline.
        
        Returns:
            Path to the output classified raster
        """
        if not self._validate_inputs():
            raise ValueError("Input validation failed")
        
        # Read training data
        X, y = self.read_training_data(self.config.raster_paths[0])
        
        if len(X) == 0:
            raise ValueError("No training data found")
        
        # Train model
        self.train_model(X, y)
        
        # Perform classification
        output_path = os.path.join(self.config.output_dir, "classified.tif")
        self.predict_in_chunks(output_path)
        
        return output_path


def load_config_from_file(config_path: str) -> Config:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Config object
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    return Config(**config_dict)


def main():
    """Main execution function."""
    # Example configuration - replace with your paths
    config = Config(
        raster_paths=[
            r"C:\Users\123\Desktop\r_s_class\S1B_IW_SLC__1SDV_20210628T013430_20210628T013457_027550_0349E9_037C_deb_mat_Decomp_TC.tif",
            r"C:\Users\123\Desktop\r_s_class\S1B_IW_SLC__1SDV_20210722T013431_20210722T013458_027900_035441_242D_deb_mat_Decomp_TC.tif",
            r"C:\Users\123\Desktop\r_s_class\S1B_IW_SLC__1SDV_20210815T013433_20210815T013500_028250_035EC3_9790_deb_mat_Decomp_TC.tif"
        ],
        shapefile_dir=r"C:\Users\123\Desktop\r_s_class\борисовка",
        output_dir=r"C:\Users\123\Desktop\r_s_class\output",
        chunk_size=512,
        crops=['пшеница', 'ячмень', 'овес']
    )
    
    # Initialize classifier
    classifier = RasterClassifier(config)
    
    try:
        # Run full pipeline
        output_path = classifier.run_full_pipeline()
        print(f"[+] Classification completed successfully!")
        print(f"[+] Output saved to: {output_path}")
        
    except Exception as e:
        print(f"[-] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()