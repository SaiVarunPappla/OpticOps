# 📷 OpticOps

**OpticOps** is a PyTorch-based Optical Character Recognition (OCR) pipeline designed to recognize multi-digit numeric sequences from LCD screen images. It uses a Convolutional Neural Network (CNN) combined with Connectionist Temporal Classification (CTC) loss to learn and decode sequential digit patterns — ideal for reading utility meters, counters, and digital displays.

---

## 🚀 Features

- 📸 LCD image digit sequence recognition  
- 🧠 CNN + CTC loss for sequence prediction  
- 🛠️ Albumentations for data augmentation  
- 🧪 Accuracy-based model checkpointing  
- 📊 Integrated train/test evaluation with duplicate removal logic  

---

## 🗂️ Project Structure

OpticOps/
│<br>
├── train.py # Training script with model saving<br>
├── model.py # CNN model with CTC output<br>
├── config.py # Configuration constants (paths, hyperparameters)<br>
├── engine.py # Train and evaluation logic<br>
├── dataset.py # Custom Dataset class for image/label loading<br>
├── saved_models/ # Output directory for saved models<br>
├── labels.csv # CSV file: image_name,label<br>
├── images/ # Directory containing all input .jpg images<br>



🤝 Contributing
If you have suggestions or improvements, feel free to open an issue or submit a pull request.

📄 License
Licensed under the MIT License.
