# Air Quality Prediction using Machine Learning

## Overview

This project uses a machine learning model to predict the Air Quality Index (AQI) based on various gas levels such as CO2, CH4, CO, H, N2O, O3, and CFC. The model is trained using a linear regression approach, and its performance is evaluated using the R² score. The project also includes checks for multi-collinearity and normality of residuals to ensure the validity of the model.

## Features

- **Input:** The user provides gas level values (CO2, CH4, CO, H, N2O, O3, CFC).
- **Prediction:** The model predicts the AQI value based on the inputted gas levels.
- **Model Performance:** The model evaluates its performance using the R² score (accuracy).
- **Visualization:** 
  - Pairplot to check the linearity between different gases and AQI.
  - Heatmap to check multi-collinearity between different gas levels.
  - Residuals distribution to check for normality.

## Requirements
- Libraries:
  - numpy
  - pandas
  - seaborn
  - matplotlib
  - scikit-learn

3. **Input values**: The script will prompt you to enter values for the following gases:
   - CO2
   - CH4
   - CO
   - H
   - N2O
   - O3
   - CFC
   
4. **Output**: The script will display the predicted AQI value and model accuracy (R² score as a percentage).

## Results

- The predicted AQI value is displayed after inputting the gas levels.
- The R² score is calculated to assess the model's accuracy.

## Example Output

Enter CO2 level: 450
Enter CH4 level: 1900
Enter CO level: 8
Enter H level: 0.5
Enter N2O level: 300
Enter O3 level: 0.05
Enter CFC level: 0.05
Predicted AQI Value: 75.23
R2 score (accuracy as percentage) = 85.24%

## Visualizations

- **Pairplot:** Shows relationships between different gas levels and AQI.
- **Heatmap:** Visualizes the correlation between different gas levels.
- **Residuals Distribution:** Helps in checking the normality of errors.

## Conclusion

This machine learning model helps predict AQI based on the levels of several gases. By analyzing the R² score, you can assess the model's performance. Future improvements could include experimenting with other machine learning algorithms and refining the model further.
