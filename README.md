# Student Performance Analysis

Data analysis project analyzing factors affecting student exam scores using Python.

## Dataset
6607 students with 20 features from Kaggle

## Questions Analyzed
1. Does attendance affect exam score?
2. Do students with internet access score better?
3. Does studying more hours = better exam score?
4. Top factors influencing exam score

## Key Findings
- Attendance is the strongest factor (0.58 correlation)
- Internet access has minimal impact
- Hours studied has moderate impact

## Technologies Used
- Python, Pandas, NumPy, Matplotlib

## Machine Learning Model
Built a Linear Regression model to predict student exam scores.

### Model Performance
- Mean Absolute Error: 1.27 marks
- R2 Score: 0.64 (64% accuracy)

### How it works
- Features used: Attendance, Hours Studied, Previous Scores, Tutoring Sessions
- Split data: 80% training, 20% testing
- Model predicts exam score within ~1.27 marks of actual score
