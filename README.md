**Disclaimer:** I intend to write my master's thesis on "Aircraft Region Detection using Deep Learning." However, I am uncertain whether the dataset will be ready by the deadline required for this course. This will become clear shortly. Therefore, I have outlined an alternative plan below, which involves selecting a different project type. My preferred choice is Topic 1; if that is not feasible, I will pursue Topic 2 instead.

# Aircraft Region Detection using Deep Learning

## 1. References to scientific papers:
a) Deep Learning for Scene Classification: A Survey by  Zeng et. al.
b) Scene classification with Convolutional Neural Networks King, Kishore, Ranalli

## 2. Topic:
Aircraft Region Detection using Deep Learning 

## 3. Project Type:
Bring your own data 

## 4. Written Summary: 

### a) Short description of project idea and approach:
The project aims to develop a deep learning-based system for detecting and classifying different regions of an aircraft to support visual inspections. The approach will involve: 

- Collecting a dataset of aircraft images from various angles and regions
- Developing a convolutional neural network (CNN) or Vision Transformer for region classification
- Evaluating the model's performance on a test set

### b) Description of the dataset:
The dataset will consist of high-resolution images of various aircraft regions, including: 

- Fuselage sections
- Wings (leading edge, trailing edge, winglets)
- Empennage (vertical and horizontal stabilizers)
- Landing gear
- Engine nacelles and pylons

Images will be collected from multiple aircraft types to ensure diversity. Each image will be labeled with the corresponding region classification. The goal is to collect enough pictures so that the prediction is reliable 

### c) Work-breakdown structure with time estimates: 

1. **Dataset Collection and Preparation (4 weeks) (Oct 21 – 17 Nov)**
   - Coordinate with Professor about access to the pictures (14 days)
   - Organize and label images (13 days)
   - Preprocess and augment data (2 days)

2. **Network Design and Implementation (2 weeks) (18 Nov – 1 Dez)**
   - Research and select appropriate CNN architecture (3 days)
   - Implement chosen architecture using PyTorch (4 days)
   - Set up data pipeline and training environment (3 days)

3. **Model Training and Fine-tuning (2 weeks) (2 Dez – 15 Dez)**
   - Initial model training (5 days)
   - Hyperparameter tuning and optimization (5 days)
   - Validation and testing (4 days)

4. **Performance Evaluation and Analysis (1 week) (12 Dez-17 Dez)**
   - Conduct comprehensive evaluation on test set (3 days)
   - Analyze results and generate performance metrics (2 days)

*Christmas Break*

5. **Application Development and Integration (27 Dez – 12 Jan):**
   - Build a small application to run the model (14 days)
   - Prepare deliverables (2 days)

6. **Showcase Application and Finalize Presentation (13 Jan – Jan 21):**
   - Any other leftover things (3 days)
   - Make the presentation (3 days)


# Plan B
# Vision Transformer for Scene Classification using AID Dataset

## 1. References to Scientific Papers
- **a)** "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (2021)
- **b)** "Image Classification Based on Vision Transformer" by Scirp Journal (2023)

## 2. Topic
**Scene Classification using Vision Transformer (ViT) on the AID (Aerial Image Dataset) dataset**

## 3. Project Type
**Bring your own method**

## 4. Written Summary

### a) Short Description of Project Idea and Approach
This project aims to implement a Vision Transformer (ViT) model for scene classification using the AID dataset. The approach will involve:
- Adapting the ViT architecture for scene classification tasks
- Training the ViT model on the AID dataset
- Comparing the performance of the ViT model with traditional CNN approaches
- Analyzing the model's attention mechanisms for interpretability

### b) Description of the Dataset
The AID dataset is a large-scale aerial image dataset for scene classification. It contains:
- 10,000 aerial images
- 30 scene classes (e.g., airport, bare land, beach, bridge, commercial, desert)
- Images with a fixed size of 600x600 pixels
- High intra-class variations and small inter-class dissimilarities

### c) Work-Breakdown Structure with Time Estimates
- **Dataset Preparation and Exploration (2 weeks)** (Oct 22 - Nov 4)
  - Download and organize the AID dataset (4 days)
  - Perform exploratory data analysis (5 days)
  - Preprocess and augment data (5 days)

- **ViT Model Design and Implementation (2 weeks)** (Nov 5 - Nov 18)
  - Study ViT architecture and adapt for scene classification (5 days)
  - Implement ViT model using PyTorch or TensorFlow (6 days)
  - Set up data pipeline and training environment (3 days)

- **Model Training and Optimization (2 weeks)** (Nov 19 - Dec 2)
  - Initial model training (5 days)
  - Hyperparameter tuning and optimization (5 days)
  - Validation and testing (4 days)

- **Performance Evaluation and Comparison (1 week)** (Dec 3 - Dec 9)
  - Evaluate ViT model on test set (2 days)
  - Compare with CNN baselines (2 days)
  - Analyze attention mechanisms for interpretability (3 days)

- **Application Development and Integration (2 weeks)** (Dec 10 - Dec 23)
  - Develop a simple web or desktop application to demonstrate the model (8 days)
  - Integrate the trained ViT model into the application (6 days)

**Christmas Break**

- **Documentation and Reporting (1 week)** (Jan 2 - Jan 8)
  - Write a comprehensive report on the methodology and results (4 days)
  - Prepare a presentation summarizing the project (3 days)

- **Final Review and Submission (2 weeks)** (Jan 9 - Jan 22)
  - Review and refine all deliverables (8 days)
  - Prepare for final submission (6 days)
