#!/usr/bin/env python3
"""
Example usage of the improved Raster Classification System
"""

from raster_classifier import RasterClassifier, Config, load_config_from_file


def example_with_config_object():
    """Example using a Config object directly."""
    print("=== Example 1: Using Config object ===")
    
    # Create configuration
    config = Config(
        raster_paths=[
            "data/raster1.tif",
            "data/raster2.tif", 
            "data/raster3.tif"
        ],
        shapefile_dir="data/shapefiles",
        output_dir="output",
        chunk_size=512,
        crops=['пшеница', 'ячмень', 'овес'],
        n_estimators=100,
        random_state=42
    )
    
    # Initialize classifier
    classifier = RasterClassifier(config)
    
    try:
        # Run full pipeline
        output_path = classifier.run_full_pipeline()
        print(f"✓ Classification completed!")
        print(f"✓ Output saved to: {output_path}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_with_config_file():
    """Example using a configuration file."""
    print("\n=== Example 2: Using config file ===")
    
    try:
        # Load configuration from file
        config = load_config_from_file("config.json")
        
        # Initialize classifier
        classifier = RasterClassifier(config)
        
        # Run full pipeline
        output_path = classifier.run_full_pipeline()
        print(f"✓ Classification completed!")
        print(f"✓ Output saved to: {output_path}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_step_by_step():
    """Example showing step-by-step execution."""
    print("\n=== Example 3: Step-by-step execution ===")
    
    config = Config(
        raster_paths=["data/raster1.tif", "data/raster2.tif"],
        shapefile_dir="data/shapefiles",
        output_dir="output",
        crops=['пшеница', 'ячмень']
    )
    
    classifier = RasterClassifier(config)
    
    try:
        # Step 1: Validate inputs
        if not classifier._validate_inputs():
            print("✗ Input validation failed")
            return
        print("✓ Inputs validated")
        
        # Step 2: Read training data
        X, y = classifier.read_training_data(config.raster_paths[0])
        print(f"✓ Training data loaded: {X.shape}")
        
        # Step 3: Train model
        model, encoder = classifier.train_model(X, y)
        print("✓ Model trained")
        
        # Step 4: Perform classification
        output_path = "output/classified_step_by_step.tif"
        classifier.predict_in_chunks(output_path)
        print(f"✓ Classification completed: {output_path}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_load_existing_model():
    """Example showing how to load and use an existing model."""
    print("\n=== Example 4: Using existing model ===")
    
    config = Config(
        raster_paths=["data/raster1.tif", "data/raster2.tif"],
        shapefile_dir="data/shapefiles",
        output_dir="output",
        model_path="output/rf_model.pkl"  # Path to existing model
    )
    
    classifier = RasterClassifier(config)
    
    try:
        # Load existing model
        model, encoder = classifier.load_model()
        print("✓ Model loaded successfully")
        
        # Perform classification with loaded model
        output_path = "output/classified_with_loaded_model.tif"
        classifier.predict_in_chunks(output_path)
        print(f"✓ Classification completed: {output_path}")
        
    except FileNotFoundError:
        print("✗ Model file not found. Train a model first.")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    print("Raster Classification System - Usage Examples")
    print("=" * 50)
    
    # Run examples
    example_with_config_object()
    example_with_config_file()
    example_step_by_step()
    example_load_existing_model()
    
    print("\n" + "=" * 50)
    print("All examples completed!")