# Artificial Neural Network (ANN) - Video Game Success Prediction for Billing Games 

## Project Overview:
The **Video Game Success Prediction** project uses an **Artificial Neural Network (ANN)** to predict the success of a video game based on features such as player details, playtime, and various other in-game metrics. The model classifies the likelihood of a video game being successful as a binary outcome: either successful (1) or unsuccessful (0). The model helps understand the key factors that influence the success of games, assisting game developers at **Billing Games** in optimizing their game strategies and improving product offerings.

## Objective:
The primary goal of this project is to build an **ANN model** that predicts the success of a video game based on a range of factors such as player engagement, in-game behavior, and product attributes. The system helps Billing's game development team gain insights into which game features are most likely to lead to commercial success, thereby improving future game development and marketing strategies.

## Key Features:
- **Data Preprocessing**: The dataset is cleaned and transformed, including handling categorical variables (e.g., player clan), feature scaling, and splitting the data into training and testing sets.
- **ANN Architecture**: The model uses a simple **Fully Connected (Dense) Neural Network** with **ReLU** activations for hidden layers and a **sigmoid activation** for the output layer, suitable for binary classification tasks.
- **Model Training**: The network is trained using **binary cross-entropy** loss and the **Adam optimizer**, ensuring an efficient learning process for classifying game success.
- **Performance Visualization**: Training and validation accuracy, along with loss, are plotted to visualize model performance over epochs.

## Methods & Techniques:

### **1. Data Preprocessing**:
The raw data includes player information and game performance metrics. The preprocessing steps involve:
- **Dropping redundant columns**: Unnecessary columns, such as `player_id`, are removed.
- **Handling Categorical Data**: Categorical variables (e.g., `clan`) are encoded using **OneHotEncoder** to make them compatible with the neural network.
- **Feature Scaling**: **MinMaxScaler** is used to scale numerical features into the range [0, 1], which helps the model train more efficiently.
- **Train-Test Split**: The data is split into **training** (80%) and **testing** (20%) sets, ensuring proper model validation.

### **2. Neural Network Architecture**:
The model uses a **fully connected neural network** with the following layers:
- **Input Layer**: The number of nodes corresponds to the number of input features.
- **Hidden Layers**: Two **Dense layers** with 32 units each, activated by **ReLU** functions to introduce non-linearity.
- **Output Layer**: A single neuron with **sigmoid activation** to predict a binary output (successful or unsuccessful).

### **3. Model Training**:
The model is trained using the **Adam optimizer** and **binary cross-entropy loss function**, which are standard choices for binary classification tasks. We train the network over 50 epochs, using a batch size of 32. A **ModelCheckpoint** callback is used to save the best model based on validation accuracy during training.

### **4. Evaluation and Visualization**:
During the training, the model’s performance is tracked with the following metrics:
- **Loss**: Measures the error between predicted and true values.
- **Accuracy**: Tracks how often the model correctly predicts the outcome.
The results are plotted as graphs of **loss** and **accuracy** for both training and validation datasets, helping us understand model convergence.

### **5. Making Predictions**:
After training, the model can predict the success of new game data:
- **New Player Data**: Using the trained model, we predict the success of a game for new player data (e.g., `player_a` and `player_b`).
- The prediction returns a probability value, which is then converted into a binary classification (`1` for success, `0` for failure) based on a threshold of **0.5**.

## Technologies Used:
- **Python**: Programming language for implementing the neural network and data preprocessing.
- **Pandas**: For data manipulation and handling the dataset.
- **TensorFlow/Keras**: For building, training, and evaluating the ANN model.
- **scikit-learn**: For preprocessing tasks such as splitting data and encoding categorical variables.
- **Matplotlib**: For visualizing the training and validation performance over epochs.

## Key Results & Outcomes:
- The model successfully predicts the success of a video game based on input features.
- The training and validation accuracy improved over epochs, showing the model's ability to generalize.
- **Cosine Similarity** was used to calculate the success prediction, ensuring efficient and reliable results.

## Lessons Learned:
- Data preprocessing, such as **feature scaling** and **encoding categorical variables**, is crucial to making the model work efficiently.
- **Transfer learning** is an effective technique to leverage previously learned patterns, saving time on model training and improving accuracy.
- **Hyperparameter tuning** plays an essential role in optimizing the performance of the neural network.

## Future Enhancements:
- **Hyperparameter Tuning**: Further optimization of the ANN architecture, including exploring different activation functions, hidden layer sizes, and regularization techniques.
- **Advanced Models**: Implementing more advanced neural network architectures like **Convolutional Neural Networks (CNNs)** or **Recurrent Neural Networks (RNNs)** could improve predictions for sequential data (e.g., player behavior over time).
- **Model Deployment**: The trained model can be deployed in a production environment for real-time predictions for ABC Grocery's video games.

