import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

# Convolutional Layer
def conv2d(input_data, kernel, stride=1, padding=0):
    input_padded = np.pad(input_data, [(padding, padding), (padding, padding)], mode='constant', constant_values=0)
    kernel_height, kernel_width = kernel.shape
    output_height = (input_data.shape[0] - kernel_height + 2 * padding) // stride + 1
    output_width = (input_data.shape[1] - kernel_width + 2 * padding) // stride + 1
    output = np.zeros((output_height, output_width))

    for i in range(0, output_height):
        for j in range(0, output_width):
            output[i, j] = np.sum(input_padded[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width] * kernel)

    return output

# ReLU Activation Function
def relu(x):
    return np.maximum(0, x)

# Max Pooling Layer
def max_pooling(input_data, size=2, stride=2):
    output_height = (input_data.shape[0] - size) // stride + 1
    output_width = (input_data.shape[1] - size) // stride + 1
    output = np.zeros((output_height, output_width))
    
    for i in range(0, output_height):
        for j in range(0, output_width):
            output[i, j] = np.max(input_data[i*stride:i*stride+size, j*stride:j*stride+size])
    
    return output

# Load and process the image
def load_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((128, 128))  # Resize for simplicity, if needed
    img_array = np.array(img)
    return img_array

# Display Image
def display_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Main function to run the CNN pipeline
def cnn_pipeline(image_path, kernel):
    # Load and display the original image
    input_image = load_image(image_path)
    
    # Apply Convolution, ReLU, and Max Pooling
    conv_output = conv2d(input_image, kernel)
    relu_output = relu(conv_output)
    pooled_output = max_pooling(relu_output)

    return input_image, conv_output, relu_output, pooled_output

# Function to process multiple images
def process_multiple_images(image_paths, kernel):
    results = []
    for i, image_path in enumerate(image_paths):
        print(f"Processing Image {i+1}: {image_path}")
        input_image, conv_output, relu_output, pooled_output = cnn_pipeline(image_path, kernel)
        results.append({
            "input_image": input_image,
            "conv_output": conv_output,
            "relu_output": relu_output,
            "pooled_output": pooled_output
        })
        # Optionally display the images
        display_image(input_image, f"Original Image {i+1}")
        display_image(conv_output, f"Convolution Output {i+1}")
        display_image(relu_output, f"ReLU Output {i+1}")
        display_image(pooled_output, f"Max Pooling Output {i+1}")
    return results

# Save the results to a JSON file
def save_results_to_file(results, filename):
    with open(filename, 'w') as f:
        json.dump({k: v.tolist() for k, v in results.items()}, f)

# Load the expected results from a file
def load_results_from_file(filename):
    with open(filename, 'r') as f:
        loaded = json.load(f)
    return {k: np.array(v) for k, v in loaded.items()}

# Save actual results after running CNN pipeline
def save_actual_results(image_paths, kernel, filename):
    actual_results = process_multiple_images(image_paths, kernel)
    # Save each image's result separately
    for i, result in enumerate(actual_results):
        save_results_to_file(result, f"{filename}_image_{i+1}.json")
    print("Actual results saved.")

# Testing function: Compares the result with expected outputs
def test_cnn_pipeline_with_saved_results(image_paths, kernel, expected_result_files):
    actual_results = process_multiple_images(image_paths, kernel)
    
    # Loop through each result and compare with expected results
    for i, actual in enumerate(actual_results):
        expected = load_results_from_file(expected_result_files[i])
        # Compare Convolution Output
        assert np.allclose(actual["conv_output"], expected["conv_output"]), f"Test failed at convolution for image {i+1}"
        # Compare ReLU Output
        assert np.allclose(actual["relu_output"], expected["relu_output"]), f"Test failed at ReLU for image {i+1}"
        # Compare Max Pooling Output
        assert np.allclose(actual["pooled_output"], expected["pooled_output"]), f"Test failed at max pooling for image {i+1}"
        print(f"Image {i+1} passed all tests.")
    
    print("All tests passed!")

# Example Usage: Testing CNN with expected results
if __name__ == "__main__":
    # Define image paths
    image_paths = [
        "",  # Replace with your actual image file paths
        ""    # Replace with your actual image file paths 
    ]
    
    # Define a simple kernel for edge detection (Sobel filter)
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])
    
    # Save actual results for later comparison (only needs to be done once)
    save_actual_results(image_paths, kernel, "actual_results")

    # Define expected result files (previously saved actual results)
    expected_result_files = [
        "actual_results_image_1.json",
        "actual_results_image_2.json"
    ]

    # Run the test against saved expected results
    test_cnn_pipeline_with_saved_results(image_paths, kernel, expected_result_files)
