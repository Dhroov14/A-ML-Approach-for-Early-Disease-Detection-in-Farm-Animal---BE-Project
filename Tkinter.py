import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
from PIL import Image, ImageTk

def show_page1():
    # Hide other frames and show this one
    page2_frame.pack_forget()
    page3_frame.pack_forget()
    page1_frame.pack(fill="both", expand=True)

def show_page2():
    # Hide other frames and show this one
    page1_frame.pack_forget()
    page3_frame.pack_forget()
    page2_frame.pack(fill="both", expand=True)

def show_page3():
    # Hide other frames and show this one
    page1_frame.pack_forget()
    page2_frame.pack_forget()
    page3_frame.pack(fill="both", expand=True)

def upload_image():
    pass  # Your image upload code here

def send_message():
    pass  # Your send message code here

def change_language(*args):
    pass  # Your language change code here

root = tk.Tk()
root.title("Lumpy Skin Disease Detection & Chatbot")
root.geometry("1200x780")
root.resizable(False, False)

# Background image setup
bg_image = Image.open("9background.jpg").resize((1200, 780), Image.Resampling.LANCZOS)
bg_image = ImageTk.PhotoImage(bg_image)

# Main canvas
canvas = tk.Canvas(root, width=1200, height=780)
canvas.create_image(0, 0, image=bg_image, anchor="nw")
canvas.pack(fill="both", expand=True)

# Frame 1 (Home page)
page1_frame = tk.Frame(root, bg="#f4f4f4")
title_label = tk.Label(page1_frame, text="Welcome to Lumpy Skin Disease Detection & Chatbot", font=("Arial", 24, "bold"), bg="#f4f4f4")
title_label.pack(pady=20)
info_button = tk.Button(page1_frame, text="Learn More", command=show_page2, font=("Arial", 14), bg="#4CAF50", fg="white", relief="flat", width=15, height=2)
info_button.pack(pady=10)
upload_button = tk.Button(page1_frame, text="Upload Image", command=upload_image, font=("Arial", 14), bg="#4CAF50", fg="white", relief="flat", width=15, height=2)
upload_button.pack(pady=10)
chat_button = tk.Button(page1_frame, text="Chat with Bot", command=show_page3, font=("Arial", 14), bg="#4CAF50", fg="white", relief="flat", width=15, height=2)
chat_button.pack(pady=10)

# Frame 2 (Information page)
page2_frame = tk.Frame(root, bg="#f4f4f4")
info_label = tk.Label(page2_frame, text="Lumpy Skin Disease (LSD) is a viral disease affecting cattle, causing skin lumps and other symptoms. Early detection is crucial for effective management.", font=("Arial", 14), bg="#f4f4f4")
info_label.pack(pady=20)
back_button = tk.Button(page2_frame, text="Back to Home", command=show_page1, font=("Arial", 14), bg="#4CAF50", fg="white", relief="flat", width=15, height=2)
back_button.pack(pady=10)

# Frame 3 (Chatbot page)
page3_frame = tk.Frame(root, bg="#f4f4f4")
chat_label = tk.Label(page3_frame, text="Chatbot", font=("Arial", 24, "bold"), bg="#f4f4f4")
chat_label.pack(pady=20)
chat_display = tk.Text(page3_frame, width=85, height=25, font=("Arial", 12), bg="#f4f4f4", wrap="word")
chat_display.pack(padx=20, pady=10)
chat_input = tk.Text(page3_frame, width=85, height=6, font=("Arial", 12), bg="#f4f4f4", wrap="word")
chat_input.pack(padx=20, pady=10)
send_button = tk.Button(page3_frame, text="Send", command=send_message, font=("Arial", 14), bg="#4CAF50", fg="white", relief="flat", width=15, height=2)
send_button.pack(pady=10)
back_button2 = tk.Button(page3_frame, text="Back to Home", command=show_page1, font=("Arial", 14), bg="#4CAF50", fg="white", relief="flat", width=15, height=2)
back_button2.pack(pady=10)

# Language selection menu
language_var = tk.StringVar()
language_var.set("Language select")
language_menu = ttk.Combobox(root, textvariable=language_var, values=["English", "Marathi"], state="readonly", font=("Arial", 12))
language_menu.place(x=820, y=730, width=150, height=30)
language_var.trace("w", change_language)

# Start with the Home page
show_page1()

root.mainloop()
