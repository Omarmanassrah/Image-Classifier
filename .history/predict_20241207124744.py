import argparse
from utils import process_image, load_label_map
from tensorflow.keras.models import load_model

def predict(image_path, model, top_k):
    """Predict the top K classes for an input image."""
    processed_image = process_image(image_path)
    predictions = model.predict(processed_image)[0]
    top_indices = predictions.argsort()[-top_k:][::-1]
    top_probs = predictions[top_indices]
    return top_indices, top_probs

def main():
    parser = argparse.ArgumentParser(description="Predict flower class from an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("model_path", type=str, help="Path to the trained Keras model")
    parser.add_argument("--top_k", type=int, default=1, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to JSON file mapping labels to names")

    args = parser.parse_args()

    # Load the model
    model = load_model(args.model_path)

    # Make predictions
    top_indices, top_probs = predict(args.image_path, model, args.top_k)

    if args.category_names:
        label_map = load_label_map(args.category_names)
        top_classes = [label_map[str(idx)] for idx in top_indices]
    else:
        top_classes = [str(idx) for idx in top_indices]

    for cls, prob in zip(top_classes, top_probs):
        print(f"Class: {cls}, Probability: {prob:.4f}")

if __name__ == "__main__":
    main()
