import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("StudentPerformanceFactors.csv")

print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())                                # checks for missing value in the dataset
print(df.duplicated().sum())                            # checks for the repeated values in the dataset 


#  Question 1 — Does attendance affect exam score?

df["Attendance_Range"] = pd.cut(df["Attendance"],
                                 bins = [0,60,70,80,90,100],
                                   labels=['0-60','60-70','70-80','80-90','90-100'])
#avg exam score for each attendance range 
avg_exam_score = df.groupby("Attendance_Range")["Exam_Score"].mean()
print("\nAverage Exam Score by Attendance Range:")
print(avg_exam_score)

# Bar chart 
avg_exam_score.plot(kind='bar', color='steelblue')
plt.title("Average Exam Score by Attendance Range")     # Gives title to the graph
plt.xlabel("Attendance Range")                          # Gives label to the x-axis
plt.ylabel("Average Exam Score")                        # Gives label to the y-axis 
plt.xticks(rotation=0)                                  # Rotates x-axis labels

plt.tight_layout()                                      # To ensure everything fits without overlapping
plt.show()                                              # To display the Graph 

# Box plot Representation 
df.boxplot(column= "Exam_Score", by = "Attendance_Range")
plt.title("Exam Score Distribution by Attendance Range")
plt.suptitle("")                                        # Removes the default title to avoid overlap
plt.xlabel("Attendance Range")
plt.ylabel("Exam Score")
plt.show()

#  Question 2 — Do students with internet access score better?

avg_exam_score_internet = df.groupby("Internet_Access")["Exam_Score"].mean()
print("\nAverage Exam Score by Internet Access:")
print(avg_exam_score_internet)

# Bar Chart
avg_exam_score_internet.plot(kind='bar' , color='steelblue')
plt.title("Average Exam Score by Internet Access")
plt.xlabel("Internet Access")
plt.ylabel("Average Exam Score")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# Box PLot Representation
df.boxplot(column= "Exam_Score", by = "Internet_Access")
plt.title("Exam Score Distribution by Internet Access")
plt.suptitle("")                                       
plt.xlabel("Internet Access")
plt.ylabel("Exam Score")
plt.show()

#  Question 3 — Does studying more hours = better exam score?

plt.scatter(df["Hours_Studied"], df["Exam_Score"], alpha=0.3) # plt.scatter(x, y)
plt.title("Exam Score vs Hours Studied")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")

# Adding a Regression Line
m, b = np.polyfit(df["Hours_Studied"], df["Exam_Score"], 1)
plt.plot(df["Hours_Studied"], m * df["Hours_Studied"] + b, color='red')

plt.show()

#  Question 4 — Top factors influencing exam score

numeric_cols = df[['Hours_Studied', 'Attendance', 'Sleep_Hours', 
                    'Previous_Scores', 'Tutoring_Sessions', 
                    'Physical_Activity', 'Exam_Score']]

correlation = numeric_cols.corr()['Exam_Score'].sort_values(ascending=False)
print("\nCorrelation with Exam Score:")
print(correlation)

correlation.drop("Exam_Score").plot(kind='bar', color='steelblue') # "drop" removes the Exam_Score 
plt.title("Correlation of Factors with Exam Score")
plt.xlabel("Factors")
plt.ylabel("Correlation Coefficient")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
