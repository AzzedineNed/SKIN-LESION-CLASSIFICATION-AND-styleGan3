# Skin Lesion Classification and Synthesis

This project consists of two parts:

## Part 1: Skin Lesion Classification

This part of the project involves a comparative study for a binary classification problem using the ISIC dataset, specifically focusing on distinguishing between Melanoma (MEL) and Nevus (NEV) skin lesions. We employ two popular convolutional neural network architectures: ResNet50 and EfficientNetB4.

### Data Preprocessing

For data preprocessing, we applied two techniques:
- **Hair Removal**: To remove hair artifacts from the images.
- **Color Constancy**: To ensure consistent color representation across different images.

### Usage

To replicate the experiments and evaluate the classification models:

1. Download the ISIC dataset.
2. Preprocess the data by removing hair artifacts and applying color constancy.
3. Train ResNet50 and EfficientNetB4 models on the preprocessed dataset.
4. Evaluate the models' performance using appropriate metrics such as accuracy, precision, recall, and F1-score.

### Results

- **ResNet50**: Achieved an accuracy of X%, precision of Y%, recall of Z%, and F1-score of W%.
- **EfficientNetB4**: Achieved an accuracy of X%, precision of Y%, recall of Z%, and F1-score of W%.

## Part 2: Skin Lesion Synthesis

In this part, we focus on generating synthetic Nevus skin lesion images using StyleGAN3, a state-of-the-art generative adversarial network (GAN) architecture.

### Data Preprocessing

We applied hair removal as the only preprocessing step to prepare the input data for StyleGAN3.

### Usage

To generate synthetic Nevus skin lesion images:

1. Prepare a dataset of Nevus skin lesion images.
2. Preprocess the data by removing hair artifacts.
3. Train a StyleGAN3 model on the preprocessed dataset to generate synthetic images.
4. Fine-tune the model as necessary to improve the quality and diversity of generated images.

### Results

- **StyleGAN3**: Generated synthetic Nevus skin lesion images with high fidelity and diversity. Qualitative evaluation showed that the generated images exhibit realistic lesion characteristics.

## Contributing

Contributions to this project are welcome! If you have any suggestions, improvements, or bug fixes, please feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
