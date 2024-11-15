# AI-Driven Plant Disease Detection System

## Group Information
- **Group Name:** Leaf PlantPatrol
- **Supervisor:** Dr. Stanley Mwangi Chege, PhD (Email: stanley.mwangichege@gmail.com)
- **Group Members:**
  - Isaac Ngugi (Group Lead: itsngugiisaackinyanjui@gmail.com)
  - Naftali Koome
  - Serena Waithera
  - Wangeci Njiru
  - Mark Wema
    
- **Group Slogan:** Your Plants, Our Priority
- **Topic:** AI for Climate Change, Agriculture, and Food Security
- **Project Title:** AI-Driven Plant Disease Detection System

## Table of Contents
1. Project Overview
2. Project Understanding
   - 2.1 Problem Statement
   - 2.2 Stakeholders
3. Installation Instructions
4. Project Structure
5. Dataset Description
   - 5.1 Key Features of the Dataset
   - 5.2 Target Variable
6. Objectives
   - 6.1 Main Objective
   - 6.2 Specific Objectives
7. Exploratory Data Analysis (EDA)
8. Modeling Overview
9. Validation Results
10. Model Explainability
11. Validation Strategy
12. Conclusion
    - 12.1 Insights
    - 12.2 Limitations
    - 12.3 Recommendations
    - 12.4 Future Work

## 1. Project Overview
The PlantPatrol project is an innovative initiative that leverages artificial intelligence (AI) to address critical issues in agriculture and food security. By creating an AI-driven plant disease detection system, PlantPatrol aims to provide farmers with an accessible, real-time tool for diagnosing plant diseases, thereby enhancing crop health and yield.

## 2. Project Understanding

### 2.1 Problem Statement
Plant diseases pose a significant threat to global food security, resulting in substantial crop losses and adversely affecting farmers, particularly those in developing regions. Traditional detection methods often require expert knowledge, are time-consuming, and are inaccessible to small-scale farmers. This delay exacerbates the spread of diseases and reduces crop yields. The PlantPatrol project proposes an AI-driven plant disease detection system that leverages image recognition technology to offer real-time, accurate disease diagnoses.

### 2.2 Stakeholders
- **Farmers:** Primary users of the system who will benefit from timely disease detection and management.
- **Agricultural Experts:** Provide insights and validation for the AI model and its recommendations.
- **Local Agricultural Organizations:** Support the dissemination of the technology and its adoption among farmers.
- **Government Agencies:** Interested in improving food security and agricultural productivity.
- **Researchers and Academics:** May use the data and findings for further studies in agriculture and AI applications.

## 3. Installation Instructions
To set up the AI-Driven Plant Disease Detection System, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/PlantPatrol.git
   ```
2. Navigate to the project directory:
   ```bash
   cd PlantPatrol
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure you have the necessary environment variables set up for your API keys and configurations.

## 4. Project Structure
1. [Notebook](https://github.com/iamisaackn/AI-Driven-Plant-Disease-Detection-System/blob/main/notebook.ipynb)
2. [Notebook PDF](https://github.com/iamisaackn/AI-Driven-Plant-Disease-Detection-System/blob/main/notebook.pdf)
3. [Power Point Presentation](https://github.com/iamisaackn/AI-Driven-Plant-Disease-Detection-System/blob/main/Power%20Point%20Plant%20Patrol%20Slide%20Presentation.pptx)
4. [Presentation PDF](https://github.com/iamisaackn/AI-Driven-Plant-Disease-Detection-System/blob/main/Presentation.pdf)
5. [Requirements File](https://github.com/iamisaackn/AI-Driven-Plant-Disease-Detection-System/blob/main/requirements.txt)
6. [Read Me](https://github.com/iamisaackn/AI-Driven-Plant-Disease-Detection-System/blob/main/README.md)
7. [Plant Patrol Streamlit App](https://plantpatrol.streamlit.app/)
8. [LICENSE](https://github.com/iamisaackn/AI-Driven-Plant-Disease-Detection-System/blob/main/LICENSE)

## 5. Dataset Description

### 5.1 Key Features of the Dataset
- **Plant Species:** Different types of plants included in the dataset (e.g., Strawberry, Apple, Tomato).
- **Disease Categories:** Various diseases affecting the plants (e.g., Apple Scab, Leaf Mold).
- **Image Count:** The number of images available for each plant and disease category.

### 5.2 Target Variable
The target variable for the model is the classification of plant diseases based on the input images. The model aims to predict the disease category for a given plant image.

## 6. Objectives

### 6.1 Main Objective
Develop an AI-based image recognition system for real-time plant disease diagnosis.

### 6.2 Specific Objectives
- Design an intuitive WhatsApp chatbot for farmers of varying technical backgrounds.
- Ensure rapid, accessible disease detection through mobile and cloud integration.
- Adapt the system for regional crops and diseases.
- Incorporate educational resources for disease prevention and treatment.
- Offer multilingual support for diverse regions.
- Track environmental and economic impact by monitoring pesticide use reduction and crop health improvements.
- Ensure data privacy and security.
- Establish a continuous feedback loop for system improvements based on user input.

## 7. Exploratory Data Analysis (EDA)
The EDA process involved analyzing the dataset to understand the distribution of plant species and diseases, identifying any imbalances, and visualizing the data to inform model development.

## 8. Modeling Overview
| Model Type                | Description                                      |
|---------------------------|--------------------------------------------------|
| Convolutional Neural Network (CNN) | Used for image classification tasks. |

## 9. Validation Results
| Metric                    | Value        |
|---------------------------|--------------|
| Accuracy                  | 95%          |
| Precision                 | 93%          |
| Recall                    | 92%          |
| F1 Score                  | 92.5%        |

## 10. Model Explainability
The model's predictions can be explained using techniques such as Grad-CAM, which highlights the regions of the input image that contributed most to the prediction, providing insights into the model's decision-making process.

## 11. Validation Strategy
The model was validated using a separate validation dataset to assess its performance and ensure it generalizes well to unseen data. Cross-validation techniques were employed to enhance reliability.

## 12. Conclusion

### 12.1 Insights
- **AI Integration in Agriculture:** The project demonstrates how AI can significantly enhance agricultural practices by providing real-time disease detection, crucial for improving crop health and yield.
- **User Accessibility:** The development of a WhatsApp chatbot makes the technology accessible to farmers with varying levels of technical expertise.
- **Data-Driven Decisions:** Farmers can make informed decisions quickly, potentially reducing crop losses and improving food security.
- **Sustainability Focus:** The project promotes sustainable farming practices by reducing reliance on pesticides through timely disease detection and management.
- **Community Empowerment:** The initiative empowers local farmers by providing them with tools and knowledge to combat plant diseases.

### 12.2 Limitations
- **Data Quality and Diversity:** The model’s performance is reliant on the quality and diversity of the training dataset.
- **Environmental Variability:** The model may struggle with variations in environmental conditions not present in the training data.
- **Complexity of Diseases:** Some plant diseases may exhibit similar symptoms, making differentiation challenging.
- **User Dependency:** The effectiveness of the system relies on users accurately capturing and submitting images.
- **Technical Limitations:** Deployment may face challenges related to internet connectivity and smartphone access in rural areas.

### 12.3 Recommendations
- **Data Augmentation:** Improve model robustness by augmenting the dataset with diverse images.
- **User Training:** Provide training sessions for farmers on how to use the system effectively.
- **Regular Model Updates:** Continuously update the model with new data to improve accuracy.
- **Feedback Mechanism:** Implement a feedback system for users to report inaccuracies.
- **Expand Disease Coverage:** Gradually expand the model to cover more plant species and diseases.

### 12.4 Future Work
- **Integration with IoT Devices:** Explore integration for real-time monitoring of plant health.
- **Multilingual Support:** Develop the system to support multiple languages.
- **Collaboration with Agricultural Experts:** Partner with experts to validate predictions and improve accuracy.
- **Mobile App Development:** Consider developing a dedicated mobile application with additional features.
- **Longitudinal Studies:** Conduct studies to assess the long-term impact of the system on crop yields and farmer practices.
