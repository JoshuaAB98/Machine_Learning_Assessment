import tkinter as tk
from tkinter import filedialog, Text
import os

root = tk.Tk()

def calcBmi():

    try:
        w = float(entryW.get())
        h = float(entryH.get())
    except:
        label3.config(text='Incorrect Inputs')
        label4.config(text='')

    try:
        bmi = (w/h/h)*10000
        label3.config(text='Your BMI is:')
        label4.config(text=round(bmi, 2))
    except:
        print("No values to perform operation!")
        bmi = -1

    if(bmi>29 and bmi<40):
        label5.config(text="Your BMI is in the overweight range.")
    elif(bmi<18.5 and bmi > 0):
        label5.config(text="Your BMI is in the underweight range.")
    elif(bmi>39.9):
        label5.config(text="You are morbidly obese!")
    elif (bmi < 0):
        label5.config(text=" ")
    else:
        label5.config(text="Your BMI is in the healthy range.")

canvas = tk.Canvas(root, height=1000, width=1900, bg="white")
canvas.pack()

# frame = tk.Canvas(root, bg="white")
# frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

entryH = tk.Entry (root)
canvas.create_window(300, 40, window=entryH)

entryW = tk.Entry (root)
canvas.create_window(300, 100, window=entryW)

labelH = tk.Label(root, text='Enter your height (cm)', bg="white")
labelH.config(font=('helvetica', 14))
canvas.create_window(100, 40, window=labelH)

labelW = tk.Label(root, text='Enter your weight (kg)', bg="white")
labelW.config(font=('helvetica', 14))
canvas.create_window(100, 100, window=labelW)

label3 = tk.Label(root, text='Your BMI is:', font=('helvetica', 10), bg="white")
canvas.create_window(200, 170, window=label3)

label4 = tk.Label(root, font=('helvetica', 10), bg="white")
canvas.create_window(200, 190, window=label4)

label5 = tk.Label(root, font=('helvetica', 10), bg="white")
canvas.create_window(200, 210, window=label5)

calc = tk.Button(root, text="Calculate!", padx=10, pady=5, fg="white", bg="#263D42", command=calcBmi)
calc.pack()

root.mainloop()