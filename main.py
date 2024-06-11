import calendar
import math
from datetime import datetime
import random
import pandas as pd

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def cosine_distance(point1, point2):
    dot_product = point1[0]*point2[0] + point1[1]*point2[1]
    magnitude1 = math.sqrt(point1[0]**2 + point1[1]**2)
    magnitude2 = math.sqrt(point2[0]**2 + point2[1]**2)
    return 1 - (dot_product / (magnitude1 * magnitude2))
def main():
    df = pd.read_csv("data/laptop_cleaned2.csv")
    print(df.head().keys())
    # Question 1: Write a program to find Perimeter and Area of a Rectangle with integers of Width and Height
    # w = int(input("enter w: "))
    # l = int(input("enter l: "))
    # print("Question 1: ")
    # print(f"Perimeter: {(w+l)/2}")
    # print(f"Area: {w*l}")
    # # print("-------------------------")
    #
    # # Question 2: Write a program to convert centimeter to decimeter and inch with float centimeter
    # print("Question 2: ")
    # cm = float(input("enter cm : "))
    # print(f"{cm} cm to decimeter =  {cm/10}")
    # print(f"{cm} cm to inch =  {cm *0.3937 }")
    # # print("-------------------------")
    #
    # # Question 3: Write a program to random an integer number and check whether this number having two or three digits
    # print("Question 3: ")
    # ramdom = random.randint(10,999)
    # if 10 <= ramdom <= 99:
    #     print(f"The random number {ramdom} has two digits.")
    # elif 100 <= ramdom <= 999:
    #     print(f"The random number {ramdom} has three digits.")
    # # print("-------------------------")
    #
    # # Question 4: Write a program to random an integer number in range [-100, 100]
    # # check whether this number positive or negative number and have two digits
    # print("Question 4: ")
    # ramdom = random.randint(-100,100)
    # if ramdom < 0:
    #     print(f"The random number {ramdom} is positive.")
    # elif ramdom > 0:
    #     print(f"The random number {ramdom} is negative.")
    # else:
    #     print(f"The random number {ramdom} is neutral number")
    # # print("-------------------------")
    #
    # # Question 5: Write a program to random an integer number in range [10, 150] and normalize it into range[0, 1]
    # print("Question 5: ")
    # x = random.randint(10,150)
    # y = (x-10)/140
    # print(f"{x} tranforms to {y:0.2} in [0,1]")
    # # print("-------------------------")
    #
    # # Question 6: Write a program to find degree and radian angle between hours and minute hands
    # # with integers hour and minute
    # print("Question 6: ")
    # minute = int(input("minute = "))
    # hours = int(input("hours = "))
    # # 30h - 5.5m
    # print(abs(30*hours - 5.5*minute))
    # # print("-------------------------")
    #
    # # Question 7: Write a program to Solve the quadratic equation x^2 + 5*x + 6 = 0
    # # and print step by step (discriminant, check condition of discriminant, solutions)
    # print("Question 7: ")
    # a = float(input("a = "))
    # b = float(input("b = "))
    # c = float(input("c = "))
    # # Calculate the discriminant
    # discriminant = b ** 2 - 4 * a * c
    # print(f"Step 1: Discriminant = {discriminant}")
    # # Check the condition of the discriminant
    # if discriminant > 0:
    #     print("Step 2: Discriminant is positive. Two real solutions exist.")
    #     x1 = (-b + math.sqrt(discriminant)) / (2 * a)
    #     x2 = (-b - math.sqrt(discriminant)) / (2 * a)
    #     print(f"Step 3: Solutions are x1 = {x1} and x2 = {x2}")
    # elif discriminant == 0:
    #     print("Step 2: Discriminant is zero. One real solution exists.")
    #     x = -b / (2 * a)
    #     print(f"Step 3: The solution is x = {x}")
    # else:
    #     print("Step 2: Discriminant is negative. No real solutions exist.")
    # # print("-------------------------")
    #
    # # Question 8: Assume we have a string "Today is Sunday and we don't need to wake up at 6 am".
    # # Print how many words in the string and check whether number in string. Print position of that
    # #number in string
    # print("Question 8 : ")
    # str = "Today is Sunday and we don't need to wake up at 6 am"
    # print(f"Số lượng từ trong câu là: {len(str)}")
    # for index,word in enumerate(str):
    #     if word.isdigit():
    #         print(f"Number {word} find at {index + 1} in string.")
    # # print("-------------------------")
    #
    # # Question 9: From the keyboard input the student profile including :
    # # Name of Subject 1,2,3 and Mark of Subject 1,2,3.
    # # Name of Student and Date of Birth
    # print("Question 9 : ")
    # # Nhập thông tin về môn học và điểm số từ bàn phím
    # n = int(input("Enter number of student = "))
    # for i in range(n):
    #     print(f"Học sinh {i}")
    #     subject1 = input("Enter subject 1: ")
    #     subject2 = input("Enter subject 2: ")
    #     subject3 = input("Enter subject 3: ")
    #     mark1 = float(input("Enter mark 1: "))
    #     mark2 = float(input("Enter mark 2: "))
    #     mark3 = float(input("Enter mark 3: "))
    #     name = input("Enter name of student: ")
    #     birth = input("Enter date of birth: ")
    #     averageMark = (mark1 + mark2 + mark3) / 3
    #     print(f"Name of student: {name}")
    #     print(f"Date of birth student: {birth}")
    #     print(f"Name of Subject 1,2,3 = {subject1,subject2,subject3}")
    #     print(f"Name of mark 1,2,3 = {subject1,subject2,subject3}")
    #     print(f"average mark = {averageMark}")
    # # print("-------------------------")
    #
    # # Question 10: From keyboard input two points with (x,y) and print Euclidean, Manhattan, Cosine
    # #distance of points
    # print("Question 10: ")
    # x1, y1 = map(float, input("Enter coordinates of point 1 (x y): ").split())
    # x2, y2 = map(float, input("Enter coordinates of point 2 (x y): ").split())
    # # Calculate distances
    # point1 = (x1, y1)
    # point2 = (x2, y2)
    # euclidean = euclidean_distance(point1, point2)
    # manhattan = manhattan_distance(point1, point2)
    # cosine = cosine_distance(point1, point2)
    #
    # # Print distances
    # print(f"Euclidean Distance: {euclidean}")
    # print(f"Manhattan Distance: {manhattan}")
    # print(f"Cosine Distance: {cosine}")
    # # print("-------------------------")
    #
    # # Question 11: Input from keyboards your birthday with day, month year
    # # and print information about weekday name, month name, and your age now
    # print("Question 11: ")
    # day = int(input("Enter your birthday (day): "))
    # month = int(input("Enter your birthday (month): "))
    # year = int(input("Enter your birthday (year): "))
    # thu = calendar.weekday(year, month, day)
    # dateNow = datetime.now()
    # nameThu = calendar.day_name[thu]
    # nameMonth = calendar.month_name[month]
    # age = dateNow.year - year - ((month,day) < (dateNow.month,dateNow.day))
    # print(f"ngày thứ {nameThu} tháng {nameMonth} và tuổi là = {age}")
    # # print("-------------------------")

    # print("Question 12: ")
    # CustomerList = ["John", "John", "Marry", "Marry", "Marry"]
    # ProductList = ["Beer", "Pork", "Milk", "Vegetable", "Pork"]
    # QuantityList = ["2 Bottles", "1 kg", "5 boxes", "2 bunches", "3 kg"]
    # # A. Create a DataFrame
    # df = pd.DataFrame({"Customer": CustomerList, "Product": ProductList, "Quantity": QuantityList})
    # # B. Separate column QuantityList to Quantity and Unit
    # df[['Quantity', 'Unit']] = df['Quantity'].str.split(' ', expand=True)
    # # C. Find customer information who bought Pork over 2kg
    # result = df[(df['Product'] == 'Pork') & (df['Unit'] == 'kg') & (df['Quantity'].astype(int) > 2)]
    # # result = df[(df['Product'] == 'Pork') & (df['Unit'] == 'kg') & (df['Quantity'].astype(int) > 2)]
    # print("Customer information who bought Pork over 2kg:")
    # print(result[['Customer', 'Product', 'Quantity', 'Unit']])
    # print("-------------------------")

if __name__ == '__main__':
    main()


