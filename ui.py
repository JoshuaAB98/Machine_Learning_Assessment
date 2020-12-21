import tkinter as tk
from tkinter import filedialog, Text
import os

root = tk.Tk()

variable = tk.StringVar(root)
variable.set("1 Day")
w = tk.OptionMenu(root, variable, '1 Day', '2 Days', '3 Days', '4 Days', '5 Days', '6 Days', '1 Week', '2 Weeks', '3 Weeks', '1 Month', '1 Year')


def func():

    try:
        print("Congrats")
    except:
        print("fail")



canvas = tk.Canvas(root, height=1000, width=1900, bg="white")
canvas.pack()

# frame = tk.Canvas(root, bg="white")
# frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

entryP = tk.Entry (root)
canvas.create_window(420, 40, window=entryP)

entryT = tk.OptionMenu(root, variable, '1 Day', '2 Days', '3 Days', '4 Days', '5 Days', '6 Days', '1 Week', '2 Weeks', '3 Weeks', '1 Month', '1 Year')
canvas.create_window(420, 100, window=entryT)

labelP = tk.Label(root, text='Enter the amount of desired profit (£): ', bg="white")
labelP.config(font=('helvetica', 14))
canvas.create_window(180, 40, window=labelP)

labelT = tk.Label(root, text='Enter the desired wait time: ', bg="white")
labelT.config(font=('helvetica', 14))
canvas.create_window(140, 100, window=labelT)

label3 = tk.Label(root, text='Your BMI is:', font=('helvetica', 10), bg="white")
canvas.create_window(220, 170, window=label3)

label4 = tk.Label(root, font=('helvetica', 10), bg="white")
canvas.create_window(220, 190, window=label4)

label5 = tk.Label(root, font=('helvetica', 10), bg="white")
canvas.create_window(220, 210, window=label5)

calc = tk.Button(root, text="Calculate!", padx=10, pady=5, fg="white", bg="#263D42", command=func())
calc.pack()

root.mainloop()