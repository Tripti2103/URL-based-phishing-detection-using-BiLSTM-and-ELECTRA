# URL-based-phishing-detection-using-BiLSTM-and-ELECTRA
Phishing URL Detection using Deep Learning

Phishing attacks have become increasingly sophisticated, tricking users into revealing sensitive information or downloading malware through deceptive URLs that mimic legitimate websites.  
This project presents a **robust, multi-layered approach** to detecting phishing URLs using deep learning techniques.  

## **Project Overview**  
![Front Page](https://github.com/Tripti2103/URL-based-phishing-detection-using-BiLSTM-and-ELECTRA/blob/main/Front%20Page.png?raw=true)  

## **Methodology**  
### **1. URL Standardization**  
- The URLs are preprocessed and converted into a numerical format.  

### **2. Feature Extraction with ELECTRA**  
- A transformer-based model extracts meaningful textual patterns from URLs.  

### **3. BiLSTM for Sequential Patterns**  
- A Bidirectional Long Short-Term Memory (BiLSTM) layer captures sequential dependencies.  

### **4. Hierarchical Attention-Based CNN (HABCNN)**  
- A combination of convolutional layers and attention mechanisms refines feature extraction.  

### **5. Gating Mechanism & Classification**  
- The model filters critical features and classifies URLs as either phishing or legitimate.  

## **Classification Output**  
**Legitimate URL Example:**  
![Legitimate URL](https://github.com/Tripti2103/URL-based-phishing-detection-using-BiLSTM-and-ELECTRA/blob/main/legitimate.png?raw=true)  

**Phishing URL Example:**  
![Phishing URL](https://github.com/Tripti2103/URL-based-phishing-detection-using-BiLSTM-and-ELECTRA/blob/main/phishing.png?raw=true)  

## **Key Benefits**  
  ->**High Accuracy** – Utilizes advanced NLP and deep learning techniques for precise classification.  
  ->**Robust & Scalable** – Effective against evolving phishing strategies.  
  ->**Contributes to Cybersecurity** – Provides a reliable approach to detecting malicious URLs.  

## **Future Improvements**  
This project aims to enhance cybersecurity by detecting phishing attempts with high efficiency. Future improvements may include integrating real-time threat analysis and expanding the dataset for better generalization.  
